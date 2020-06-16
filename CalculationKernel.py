import numpy as np
import pandas as pd
from agent.ExchangeAgent import ExchangeAgent

import os, queue, sys
from message.Message import MessageType

from util.util import log_print
from Kernel import Kernel


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


class CalculationKernel(Kernel):

    def writeLog(self, sender, dfLog, filename=None):
        return

    def appendSummaryLog(self, sender, eventType, event):
        return

    def writeSummaryLog(self):
        return

    def getOrderbook(self):
        pass

    def runner(self, agents=[], startTime=None, stopTime=None,
               num_simulations=1, defaultComputationDelay=1,
               defaultLatency=1, agentLatency=None, latencyNoise=[1.0],
               agentLatencyModel=None,
               seed=None, oracle=None, log_dir=None, return_value=None):

        # agents must be a list of agents for the simulation,
        #        based on class agent.Agent
        self.agents = agents

        # Simulation custom state in a freeform dictionary.  Allows config files
        # that drive multiple simulations, or require the ability to generate
        # special logs after simulation, to obtain needed output without special
        # case code in the Kernel.  Per-agent state should be handled using the
        # provided updateAgentState() method.
        self.custom_state = {}

        # The kernel start and stop time (first and last timestamp in
        # the simulation, separate from anything like exchange open/close).
        self.startTime = startTime
        self.stopTime = stopTime

        # The global seed, NOT used for anything agent-related.
        self.seed = seed

        # The data oracle for this simulation, if needed.
        self.oracle = oracle

        # If a log directory was not specified, use the initial wallclock.
        if log_dir:
            self.log_dir = log_dir
        else:
            self.log_dir = str(int(self.kernelWallClockStart.timestamp()))

        # The kernel maintains a current time for each agent to allow
        # simulation of per-agent computation delays.  The agent's time
        # is pushed forward (see below) each time it awakens, and it
        # cannot receive new messages/wakeups until the global time
        # reaches the agent's time.  (i.e. it cannot act again while
        # it is still "in the future")

        # This also nicely enforces agents being unable to act before
        # the simulation startTime.
        self.agentCurrentTimes = [self.startTime] * len(agents)

        # agentComputationDelays is in nanoseconds, starts with a default
        # value from config, and can be changed by any agent at any time
        # (for itself only).  It represents the time penalty applied to
        # an agent each time it is awakened  (wakeup or recvMsg).  The
        # penalty applies _after_ the agent acts, before it may act again.
        # TODO: this might someday change to pd.Timedelta objects.
        self.agentComputationDelays = [defaultComputationDelay] * len(agents)

        # If an agentLatencyModel is defined, it will be used instead of
        # the older, non-model-based attributes.
        self.agentLatencyModel = agentLatencyModel

        # If an agentLatencyModel is NOT defined, the older parameters:
        # agentLatency (or defaultLatency) and latencyNoise should be specified.
        # These should be considered deprecated and will be removed in the future.

        # If agentLatency is not defined, define it using the defaultLatency.
        # This matrix defines the communication delay between every pair of
        # agents.
        if agentLatency is None:
            self.agentLatency = [[defaultLatency] * len(agents)] * len(agents)
        else:
            self.agentLatency = agentLatency

        # There is a noise model for latency, intended to be a one-sided
        # distribution with the peak at zero.  By default there is no noise
        # (100% chance to add zero ns extra delay).  Format is a list with
        # list index = ns extra delay, value = probability of this delay.
        self.latencyNoise = latencyNoise

        # The kernel maintains an accumulating additional delay parameter
        # for the current agent.  This is applied to each message sent
        # and upon return from wakeup/receiveMessage, in addition to the
        # agent's standard computation delay.  However, it never carries
        # over to future wakeup/receiveMessage calls.  It is useful for
        # staggering of sent messages.
        self.currentAgentAdditionalDelay = 0

        # Note that num_simulations has not yet been really used or tested
        # for anything.  Instead we have been running multiple simulations
        # with coarse parallelization from a shell script.
        for sim in range(num_simulations):

            # Event notification for kernel init (agents should not try to
            # communicate with other agents, as order is unknown).  Agents
            # should initialize any internal resources that may be needed
            # to communicate with other agents during agent.kernelStarting().
            # Kernel passes self-reference for agents to retain, so they can
            # communicate with the kernel in the future (as it does not have
            # an agentID).
            for agent in self.agents:
                agent.kernelInitializing(self)

            # Event notification for kernel start (agents may set up
            # communications or references to other agents, as all agents
            # are guaranteed to exist now).  Agents should obtain references
            # to other agents they require for proper operation (exchanges,
            # brokers, subscription services...).  Note that we generally
            # don't (and shouldn't) permit agents to get direct references
            # to other agents (like the exchange) as they could then bypass
            # the Kernel, and therefore simulation "physics" to send messages
            # directly and instantly or to perform disallowed direct inspection
            # of the other agent's state.  Agents should instead obtain the
            # agent ID of other agents, and communicate with them only via
            # the Kernel.  Direct references to utility objects that are not
            # agents are acceptable (e.g. oracles).
            for agent in self.agents:
                agent.kernelStarting(self.startTime)

            # Set the kernel to its startTime.
            self.currentTime = self.startTime

            # Track starting wall clock time and total message count for stats at the end.
            eventQueueWallClockStart = pd.Timestamp('now')
            ttl_messages = 0

            # Process messages until there aren't any (at which point there never can
            # be again, because agents only "wake" in response to messages), or until
            # the kernel stop time is reached.
            while not self.messages.empty() and self.currentTime and (self.currentTime <= self.stopTime):
                # Get the next message in timestamp order (delivery time) and extract it.
                self.currentTime, event = self.messages.get()
                msg_recipient, msg_type, msg = event

                ttl_messages += 1

                # In between messages, always reset the currentAgentAdditionalDelay.
                self.currentAgentAdditionalDelay = 0

                # Dispatch message to agent.
                if msg_type == MessageType.WAKEUP:

                    # Who requested this wakeup call?
                    agent = msg_recipient

                    # Test to see if the agent is already in the future.  If so,
                    # delay the wakeup until the agent can act again.
                    if self.agentCurrentTimes[agent] > self.currentTime:
                        # Push the wakeup call back into the PQ with a new time.
                        self.messages.put((self.agentCurrentTimes[agent],
                                           (msg_recipient, msg_type, msg)))
                        continue

                    # Set agent's current time to global current time for start
                    # of processing.
                    self.agentCurrentTimes[agent] = self.currentTime

                    # Wake the agent.
                    agents[agent].wakeup(self.currentTime)

                    # Delay the agent by its computation delay plus any transient additional delay requested.
                    self.agentCurrentTimes[agent] += pd.Timedelta(self.agentComputationDelays[agent] +
                                                                  self.currentAgentAdditionalDelay)


                elif msg_type == MessageType.MESSAGE:

                    # Who is receiving this message?
                    agent = msg_recipient

                    # Test to see if the agent is already in the future.  If so,
                    # delay the message until the agent can act again.
                    if self.agentCurrentTimes[agent] > self.currentTime:
                        # Push the message back into the PQ with a new time.
                        self.messages.put((self.agentCurrentTimes[agent],
                                           (msg_recipient, msg_type, msg)))
                        continue

                    # Set agent's current time to global current time for start
                    # of processing.
                    self.agentCurrentTimes[agent] = self.currentTime

                    # Deliver the message.
                    agents[agent].receiveMessage(self.currentTime, msg)

                    # Delay the agent by its computation delay plus any transient additional delay requested.
                    self.agentCurrentTimes[agent] += pd.Timedelta(self.agentComputationDelays[agent] +
                                                                  self.currentAgentAdditionalDelay)

                else:
                    raise ValueError("Unknown message type found in queue",
                                     "currentTime:", self.currentTime,
                                     "messageType:", self.msg.type)

            # Record wall clock stop time and elapsed time for stats at the end.
            eventQueueWallClockStop = pd.Timestamp('now')

            eventQueueWallClockElapsed = eventQueueWallClockStop - eventQueueWallClockStart

            # Event notification for kernel end (agents may communicate with
            # other agents, as all agents are still guaranteed to exist).
            # Agents should not destroy resources they may need to respond
            # to final communications from other agents.
            for agent in agents:
                agent.kernelStopping()

            # Event notification for kernel termination (agents should not
            # attempt communication with other agents, as order of termination
            # is unknown).  Agents should clean up all used resources as the
            # simulation program may not actually terminate if num_simulations > 1.
            for agent in agents:
                agent.kernelTerminating()

        if return_value is None:
            return

        for agent in self.agents:
            if isinstance(agent, ExchangeAgent):
                symbol_dict = dict()
                for symbol in return_value:
                    if return_value[symbol] == "orderbook":
                        symbol_dict[symbol] = agent.order_books_log[symbol]
                    if return_value[symbol] == "midprices":
                        orderbook = agent.order_books_log[symbol]
                        symbol_dict[symbol] = self._get_midprices_from_orderbook(orderbook)
                return symbol_dict

    def _get_midprices_from_orderbook(self,df_book):

        df_book = df_book.unstack(1)
        df_book.columns = df_book.columns.droplevel(0)

        # Now row (single) index is time.  Column (single) index is quote price.

        # In temporary data frame, find best bid per (time) row.
        # Copy bids only.
        best_bid = df_book[df_book < 0].copy()

        # Replace every non-zero bid volume with the column header (quote price) instead.
        for col in best_bid.columns:
            c = best_bid[col]
            c[c < 0] = col

        # Copy asks only.
        best_ask = df_book[df_book > 0].copy()

        # Replace every non-zero ask volume with the column header (quote price) instead.
        for col in best_ask.columns:
            c = best_ask[col]
            c[c > 0] = col

        # In a new column in each temporary data frame, compute the best bid or ask.
        best_bid['best'] = best_bid.idxmax(axis=1)
        best_ask['best'] = best_ask.idxmin(axis=1)

        # Iterate over the index (all three DF have the same index) and set the special
        # best bid/ask value in the correct column(s) per row.  Also compute and include
        # the midpoint where possible.
        mid_price, mid_time = [], []
        for idx in df_book.index:
            bb = best_bid.loc[idx, 'best']
            ba = best_ask.loc[idx, 'best']
            mid_time.append(idx)
            mid_price.append(round((ba + bb) / 2))
        df_midprices=pd.DataFrame({"price": mid_price}, index=mid_time)
        df_midprices.fillna(method="ffill", inplace=True)
        df_midprices.fillna(method="bfill", inplace=True)
        return df_midprices