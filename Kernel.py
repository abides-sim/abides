import numpy as np
import pandas as pd

import os, queue, sys
from message.Message import MessageType

from util.util import log_print


class Kernel:

  def __init__(self, kernel_name, random_state = None):
    # kernel_name is for human readers only.
    self.name = kernel_name
    self.random_state = random_state

    if not random_state:
      raise ValueError("A valid, seeded np.random.RandomState object is required " +
             "for the Kernel", self.name)
      sys.exit()

    # A single message queue to keep everything organized by increasing
    # delivery timestamp.
    self.messages = queue.PriorityQueue()

    # currentTime is None until after kernelStarting() event completes
    # for all agents.  This is a pd.Timestamp that includes the date.
    self.currentTime = None

    # Timestamp at which the Kernel was created.  Primarily used to
    # create a unique log directory for this run.  Also used to
    # print some elapsed time and messages per second statistics.
    self.kernelWallClockStart = pd.Timestamp('now')

    # TODO: This is financial, and so probably should not be here...
    self.meanResultByAgentType = {}
    self.agentCountByType = {}

    # The Kernel maintains a summary log to which agents can write
    # information that should be centralized for very fast access
    # by separate statistical summary programs.  Detailed event
    # logging should go only to the agent's individual log.  This
    # is for things like "final position value" and such.
    self.summaryLog = []

    log_print ("Kernel initialized: {}", self.name)


  # This is called to actually start the simulation, once all agent
  # configuration is done.
  def runner(self, agents = [], startTime = None, stopTime = None,
             num_simulations = 1, defaultComputationDelay = 1,
             defaultLatency = 1, agentLatency = None, latencyNoise = [ 1.0 ],
             agentLatencyModel = None, skip_log = False,
             seed = None, oracle = None, log_dir = None):

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

    # Should the Kernel skip writing agent logs?
    self.skip_log = skip_log

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

    log_print ("Kernel started: {}", self.name)
    log_print ("Simulation started!")

    # Note that num_simulations has not yet been really used or tested
    # for anything.  Instead we have been running multiple simulations
    # with coarse parallelization from a shell script.
    for sim in range(num_simulations):
      log_print ("Starting sim {}", sim)

      # Event notification for kernel init (agents should not try to
      # communicate with other agents, as order is unknown).  Agents
      # should initialize any internal resources that may be needed
      # to communicate with other agents during agent.kernelStarting().
      # Kernel passes self-reference for agents to retain, so they can
      # communicate with the kernel in the future (as it does not have
      # an agentID).
      log_print ("\n--- Agent.kernelInitializing() ---")
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
      log_print ("\n--- Agent.kernelStarting() ---")
      for agent in self.agents:
        agent.kernelStarting(self.startTime)

      # Set the kernel to its startTime.
      self.currentTime = self.startTime
      log_print ("\n--- Kernel Clock started ---")
      log_print ("Kernel.currentTime is now {}", self.currentTime)

      # Start processing the Event Queue.
      log_print ("\n--- Kernel Event Queue begins ---")
      log_print ("Kernel will start processing messages.  Queue length: {}", len(self.messages.queue))

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

        # Periodically print the simulation time and total messages, even if muted.
        if ttl_messages % 100000 == 0:
          print ("\n--- Simulation time: {}, messages processed: {}, wallclock elapsed: {} ---\n".format(
                         self.fmtTime(self.currentTime), ttl_messages, pd.Timestamp('now') - eventQueueWallClockStart))

        log_print ("\n--- Kernel Event Queue pop ---")
        log_print ("Kernel handling {} message for agent {} at time {}", 
                   msg_type, msg_recipient, self.fmtTime(self.currentTime))

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
            log_print ("Agent in future: wakeup requeued for {}",
                       self.fmtTime(self.agentCurrentTimes[agent]))
            continue
            
          # Set agent's current time to global current time for start
          # of processing.
          self.agentCurrentTimes[agent] = self.currentTime

          # Wake the agent.
          agents[agent].wakeup(self.currentTime)

          # Delay the agent by its computation delay plus any transient additional delay requested.
          self.agentCurrentTimes[agent] += pd.Timedelta(self.agentComputationDelays[agent] +
                                                        self.currentAgentAdditionalDelay)

          log_print ("After wakeup return, agent {} delayed from {} to {}",
                     agent, self.fmtTime(self.currentTime), self.fmtTime(self.agentCurrentTimes[agent]))

        elif msg_type == MessageType.MESSAGE:

          # Who is receiving this message?
          agent = msg_recipient

          # Test to see if the agent is already in the future.  If so,
          # delay the message until the agent can act again.
          if self.agentCurrentTimes[agent] > self.currentTime:
            # Push the message back into the PQ with a new time.
            self.messages.put((self.agentCurrentTimes[agent],
                              (msg_recipient, msg_type, msg)))
            log_print ("Agent in future: message requeued for {}",
                       self.fmtTime(self.agentCurrentTimes[agent]))
            continue

          # Set agent's current time to global current time for start
          # of processing.
          self.agentCurrentTimes[agent] = self.currentTime

          # Deliver the message.
          agents[agent].receiveMessage(self.currentTime, msg)

          # Delay the agent by its computation delay plus any transient additional delay requested.
          self.agentCurrentTimes[agent] += pd.Timedelta(self.agentComputationDelays[agent] +
                                                        self.currentAgentAdditionalDelay)

          log_print ("After receiveMessage return, agent {} delayed from {} to {}",
                     agent, self.fmtTime(self.currentTime), self.fmtTime(self.agentCurrentTimes[agent]))

        else:
          raise ValueError("Unknown message type found in queue",
                           "currentTime:", self.currentTime,
                           "messageType:", self.msg.type)

      if self.messages.empty():
        log_print ("\n--- Kernel Event Queue empty ---")

      if self.currentTime and (self.currentTime > self.stopTime):
        log_print ("\n--- Kernel Stop Time surpassed ---")

      # Record wall clock stop time and elapsed time for stats at the end.
      eventQueueWallClockStop = pd.Timestamp('now')

      eventQueueWallClockElapsed = eventQueueWallClockStop - eventQueueWallClockStart

      # Event notification for kernel end (agents may communicate with
      # other agents, as all agents are still guaranteed to exist).
      # Agents should not destroy resources they may need to respond
      # to final communications from other agents.
      log_print ("\n--- Agent.kernelStopping() ---")
      for agent in agents:
        agent.kernelStopping()

      # Event notification for kernel termination (agents should not
      # attempt communication with other agents, as order of termination
      # is unknown).  Agents should clean up all used resources as the
      # simulation program may not actually terminate if num_simulations > 1.
      log_print ("\n--- Agent.kernelTerminating() ---")
      for agent in agents:
        agent.kernelTerminating()

      print ("Event Queue elapsed: {}, messages: {}, messages per second: {:0.1f}".format(
             eventQueueWallClockElapsed, ttl_messages, 
             ttl_messages / (eventQueueWallClockElapsed / (np.timedelta64(1, 's')))))
      log_print ("Ending sim {}", sim)


    # The Kernel adds a handful of custom state results for all simulations,
    # which configurations may use, print, log, or discard.
    self.custom_state['kernel_event_queue_elapsed_wallclock'] = eventQueueWallClockElapsed
    self.custom_state['kernel_slowest_agent_finish_time'] = max(self.agentCurrentTimes)

    # Agents will request the Kernel to serialize their agent logs, usually
    # during kernelTerminating, but the Kernel must write out the summary
    # log itself.
    self.writeSummaryLog()

    # This should perhaps be elsewhere, as it is explicitly financial, but it
    # is convenient to have a quick summary of the results for now.
    print ("Mean ending value by agent type:")
    for a in self.meanResultByAgentType:
      value = self.meanResultByAgentType[a]
      count = self.agentCountByType[a]
      print ("{}: {:d}".format(a, int(round(value / count))))

    print ("Simulation ending!")

    return self.custom_state


  def sendMessage(self, sender = None, recipient = None, msg = None, delay = 0):
    # Called by an agent to send a message to another agent.  The kernel
    # supplies its own currentTime (i.e. "now") to prevent possible
    # abuse by agents.  The kernel will handle computational delay penalties
    # and/or network latency.  The message must derive from the message.Message class.
    # The optional delay parameter represents an agent's request for ADDITIONAL
    # delay (beyond the Kernel's mandatory computation + latency delays) to represent
    # parallel pipeline processing delays (that should delay the transmission of messages
    # but do not make the agent "busy" and unable to respond to new messages).
    
    if sender is None:
      raise ValueError("sendMessage() called without valid sender ID",
                       "sender:", sender, "recipient:", recipient,
                       "msg:", msg)

    if recipient is None:
      raise ValueError("sendMessage() called without valid recipient ID",
                       "sender:", sender, "recipient:", recipient,
                       "msg:", msg)

    if msg is None:
      raise ValueError("sendMessage() called with message == None",
                       "sender:", sender, "recipient:", recipient,
                       "msg:", msg)

    # Apply the agent's current computation delay to effectively "send" the message
    # at the END of the agent's current computation period when it is done "thinking".
    # NOTE: sending multiple messages on a single wake will transmit all at the same
    # time, at the end of computation.  To avoid this, use Agent.delay() to accumulate
    # a temporary delay (current cycle only) that will also stagger messages.

    # The optional pipeline delay parameter DOES push the send time forward, since it
    # represents "thinking" time before the message would be sent.  We don't use this
    # for much yet, but it could be important later.

    # This means message delay (before latency) is the agent's standard computation delay
    # PLUS any accumulated delay for this wake cycle PLUS any one-time requested delay
    # for this specific message only.
    sentTime = self.currentTime + pd.Timedelta(self.agentComputationDelays[sender] + 
                                               self.currentAgentAdditionalDelay + delay)

    # Apply communication delay per the agentLatencyModel, if defined, or the
    # agentLatency matrix [sender][recipient] otherwise.
    if self.agentLatencyModel is not None:
      latency = self.agentLatencyModel.get_latency(sender_id = sender, recipient_id = recipient)
      deliverAt = sentTime + pd.Timedelta(latency)
      log_print ("Kernel applied latency {}, accumulated delay {}, one-time delay {} on sendMessage from: {} to {}, scheduled for {}",
                 latency, self.currentAgentAdditionalDelay, delay, self.agents[sender].name, self.agents[recipient].name,
                 self.fmtTime(deliverAt))
    else:
      latency = self.agentLatency[sender][recipient]
      noise = self.random_state.choice(len(self.latencyNoise), 1, self.latencyNoise)[0]
      deliverAt = sentTime + pd.Timedelta(latency + noise)
      log_print ("Kernel applied latency {}, noise {}, accumulated delay {}, one-time delay {} on sendMessage from: {} to {}, scheduled for {}",
                 latency, noise, self.currentAgentAdditionalDelay, delay, self.agents[sender].name, self.agents[recipient].name,
                 self.fmtTime(deliverAt))

    # Finally drop the message in the queue with priority == delivery time.
    self.messages.put((deliverAt, (recipient, MessageType.MESSAGE, msg)))

    log_print ("Sent time: {}, current time {}, computation delay {}", sentTime, self.currentTime, self.agentComputationDelays[sender])
    log_print ("Message queued: {}", msg)



  def setWakeup(self, sender = None, requestedTime = None):
    # Called by an agent to receive a "wakeup call" from the kernel
    # at some requested future time.  Defaults to the next possible
    # timestamp.  Wakeup time cannot be the current time or a past time.
    # Sender is required and should be the ID of the agent making the call.
    # The agent is responsible for maintaining any required state; the
    # kernel will not supply any parameters to the wakeup() call.

    if requestedTime is None:
        requestedTime = self.currentTime + pd.TimeDelta(1)

    if sender is None:
      raise ValueError("setWakeup() called without valid sender ID",
                       "sender:", sender, "requestedTime:", requestedTime)

    if self.currentTime and (requestedTime < self.currentTime):
      raise ValueError("setWakeup() called with requested time not in future",
                       "currentTime:", self.currentTime,
                       "requestedTime:", requestedTime)

    log_print ("Kernel adding wakeup for agent {} at time {}",
               sender, self.fmtTime(requestedTime))

    self.messages.put((requestedTime,
                      (sender, MessageType.WAKEUP, None)))


  def getAgentComputeDelay(self, sender = None):
    # Allows an agent to query its current computation delay.
    return self.agentComputationDelays[sender]


  def setAgentComputeDelay(self, sender = None, requestedDelay = None):
    # Called by an agent to update its computation delay.  This does
    # not initiate a global delay, nor an immediate delay for the
    # agent.  Rather it sets the new default delay for the calling
    # agent.  The delay will be applied upon every return from wakeup
    # or recvMsg.  Note that this delay IS applied to any messages
    # sent by the agent during the current wake cycle (simulating the
    # messages popping out at the end of its "thinking" time).

    # Also note that we DO permit a computation delay of zero, but this should
    # really only be used for special or massively parallel agents.

    # requestedDelay should be in whole nanoseconds.
    if not type(requestedDelay) is int:
      raise ValueError("Requested computation delay must be whole nanoseconds.",
                       "requestedDelay:", requestedDelay)

    # requestedDelay must be non-negative.
    if not requestedDelay >= 0:
      raise ValueError("Requested computation delay must be non-negative nanoseconds.",
                       "requestedDelay:", requestedDelay)

    self.agentComputationDelays[sender] = requestedDelay



  def delayAgent(self, sender = None, additionalDelay = None):
    # Called by an agent to accumulate temporary delay for the current wake cycle.
    # This will apply the total delay (at time of sendMessage) to each message,
    # and will modify the agent's next available time slot.  These happen on top
    # of the agent's compute delay BUT DO NOT ALTER IT.  (i.e. effects are transient)
    # Mostly useful for staggering outbound messages.

    # additionalDelay should be in whole nanoseconds.
    if not type(additionalDelay) is int:
      raise ValueError("Additional delay must be whole nanoseconds.",
                       "additionalDelay:", additionalDelay)

    # additionalDelay must be non-negative.
    if not additionalDelay >= 0:
      raise ValueError("Additional delay must be non-negative nanoseconds.",
                       "additionalDelay:", additionalDelay)

    self.currentAgentAdditionalDelay += additionalDelay



  def findAgentByType(self, type = None):
    # Called to request an arbitrary agent ID that matches the class or base class
    # passed as "type".  For example, any ExchangeAgent, or any NasdaqExchangeAgent.
    # This method is rather expensive, so the results should be cached by the caller!

    for agent in self.agents:
      if isinstance(agent, type):
        return agent.id


  def writeLog (self, sender, dfLog, filename=None):
    # Called by any agent, usually at the very end of the simulation just before
    # kernel shutdown, to write to disk any log dataframe it has been accumulating
    # during simulation.  The format can be decided by the agent, although changes
    # will require a special tool to read and parse the logs.  The Kernel places
    # the log in a unique directory per run, with one filename per agent, also
    # decided by the Kernel using agent type, id, etc.

    # If there are too many agents, placing all these files in a directory might
    # be unfortunate.  Also if there are too many agents, or if the logs are too
    # large, memory could become an issue.  In this case, we might have to take
    # a speed hit to write logs incrementally.

    # If filename is not None, it will be used as the filename.  Otherwise,
    # the Kernel will construct a filename based on the name of the Agent
    # requesting log archival.

    if self.skip_log: return

    path = os.path.join(".", "log", self.log_dir)

    if filename:
      file = "{}.bz2".format(filename)
    else:
      file = "{}.bz2".format(self.agents[sender].name.replace(" ",""))

    if not os.path.exists(path):
      os.makedirs(path)

    dfLog.to_pickle(os.path.join(path, file), compression='bz2')


  def appendSummaryLog (self, sender, eventType, event):
    # We don't even include a timestamp, because this log is for one-time-only
    # summary reporting, like starting cash, or ending cash.
    self.summaryLog.append({ 'AgentID' : sender,
                             'AgentStrategy' : self.agents[sender].type,
                             'EventType' : eventType, 'Event' : event })


  def writeSummaryLog (self):
    path = os.path.join(".", "log", self.log_dir)
    file = "summary_log.bz2"

    if not os.path.exists(path):
      os.makedirs(path)

    dfLog = pd.DataFrame(self.summaryLog)

    dfLog.to_pickle(os.path.join(path, file), compression='bz2')


  def updateAgentState (self, agent_id, state):
    """ Called by an agent that wishes to replace its custom state in the dictionary
        the Kernel will return at the end of simulation.  Shared state must be set directly,
        and agents should coordinate that non-destructively.

        Note that it is never necessary to use this kernel state dictionary for an agent
        to remember information about itself, only to report it back to the config file.
    """

    if 'agent_state' not in self.custom_state: self.custom_state['agent_state'] = {}
    self.custom_state['agent_state'][agent_id] = state

 
  @staticmethod
  def fmtTime(simulationTime):
    # The Kernel class knows how to pretty-print time.  It is assumed simulationTime
    # is in nanoseconds since midnight.  Note this is a static method which can be
    # called either on the class or an instance.

    # Try just returning the pd.Timestamp now.
    return (simulationTime)

    ns = simulationTime
    hr = int(ns / (1000000000 * 60 * 60))
    ns -= (hr * 1000000000 * 60 * 60)
    m = int(ns / (1000000000 * 60))
    ns -= (m * 1000000000 * 60)
    s = int(ns / 1000000000)
    ns = int(ns - (s * 1000000000))

    return "{:02d}:{:02d}:{:02d}.{:09d}".format(hr, m, s, ns)

