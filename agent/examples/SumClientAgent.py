from agent.Agent import Agent
from agent.examples.SumServiceAgent import SumServiceAgent
from message.Message import Message
from util.util import log_print

import pandas as pd


# The SumClientAgent class inherits from the base Agent class.  It is intended
# to serve as an example in which a service agent performs some aggregated
# computation for multiple clients and returns the result to all clients.

class SumClientAgent(Agent):

  def __init__(self, id, name, type, peer_list=None, random_state=None):
    # Base class init.
    super().__init__(id, name, type, random_state)

    self.peer_list = peer_list
    self.peer_exchange_complete = False

    self.peers_received = {}
    self.peer_sum = 0


  ### Simulation lifecycle messages.

  def kernelStarting(self, startTime):
    # self.kernel is set in Agent.kernelInitializing()

    # Find an SumServiceAgent which can answer our queries.  It is guaranteed
    # to exist by now (if there is one).
    self.serviceAgentID = self.kernel.findAgentByType(SumServiceAgent)

    log_print ("Agent {} requested agent of type Agent.SumServiceAgent.  Given Agent ID: {}",
               self.id, self.serviceAgentID)

    # Request a wake-up call as in the base Agent (spread across five seconds).
    super().kernelStarting(startTime + pd.Timedelta(self.random_state.randint(low = 0, high = 5000000000), unit='ns'))


  ### Simulation participation messages.

  def wakeup (self, currentTime):
    # Allow the base Agent to do whatever it needs to.
    super().wakeup(currentTime)

    # This agent only needs one wakeup call at simulation start.  At this time,
    # each client agent will send a number to each agent in its peer list.
    # Each number will be sampled independently.  That is, client agent 1 will
    # send n2 to agent 2, n3 to agent 3, and so forth.

    # Once a client agent has received these initial random numbers from all
    # agents in the peer list, it will make its first request from the sum
    # service.  Afterwards, it will simply request new sums when answers are
    # delivered to previous queries.

    # At the first wakeup, initiate peer exchange.
    if not self.peer_exchange_complete:
      n = [self.random_state.randint(low = 0, high = 100) for i in range(len(self.peer_list))]
      log_print ("agent {} peer list: {}", self.id, self.peer_list)
      log_print ("agent {} numbers to exchange: {}", self.id, n)

      for idx, peer in enumerate(self.peer_list):
        self.sendMessage(peer, Message({ "msg" : "PEER_EXCHANGE", "sender": self.id, "n" : n[idx] }))

    else:
      # For subsequent (self-induced) wakeups, place a sum query.
      n1, n2 = [self.random_state.randint(low = 0, high = 100) for i in range(2)]

      log_print ("agent {} transmitting numbers {} and {} with peer sum {}", self.id, n1, n2, self.peer_sum)

      # Add the sum of the peer exchange values to both numbers.
      n1 += self.peer_sum
      n2 += self.peer_sum

      self.sendMessage(self.serviceAgentID, Message({ "msg" : "SUM_QUERY", "sender": self.id,
                                                      "n1" : n1, "n2" : n2 })) 

    return


  def receiveMessage (self, currentTime, msg):
    # Allow the base Agent to do whatever it needs to.
    super().receiveMessage(currentTime, msg)

    if msg.body['msg'] == "PEER_EXCHANGE":

      # Ensure we don't somehow record the same peer twice.
      if msg.body['sender'] not in self.peers_received:
        self.peers_received[msg.body['sender']] = True
        self.peer_sum += msg.body['n']

        if len(self.peers_received) == len(self.peer_list):
          # We just heard from the final peer.  Initiate our first sum request.
          log_print ("agent {} heard from final peer.  peers_received = {}, peer_sum = {}",
                     self.id, self.peers_received, self.peer_sum)

          self.peer_exchange_complete = True
          self.setWakeup(currentTime + pd.Timedelta('1ns'))

    elif msg.body['msg'] == "SUM_QUERY_RESPONSE":
      log_print("Agent {} received sum query response: {}", self.id, msg)

      # Now schedule a new query.
      self.setWakeup(currentTime + pd.Timedelta('1m'))

