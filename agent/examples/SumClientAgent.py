from agent.Agent import Agent
from agent.examples.SumServiceAgent import SumServiceAgent
from message.Message import Message
from util.util import log_print

import pandas as pd


# The SumClientAgent class inherits from the base Agent class.  It is intended
# to serve as an example in which a service agent performs some aggregated
# computation for multiple clients and returns the result to all clients.

class SumClientAgent(Agent):

  def __init__(self, id, name, type, random_state=None):
    # Base class init.
    super().__init__(id, name, type, random_state)


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

    # This agent only needs one wakeup call at simulation start.  Afterwards,
    # it will simply request new sums when answers are delivered to previous
    # queries.

    # At that wakeup, it places its first sum query.
    n1, n2 = [self.random_state.randint(low = 0, high = 100) for i in range(2)]

    self.sendMessage(self.serviceAgentID, Message({ "msg" : "SUM_QUERY", "sender": self.id,
                                                    "n1" : n1, "n2" : n2 })) 

    return


  def receiveMessage (self, currentTime, msg):
    # Allow the base Agent to do whatever it needs to.
    super().receiveMessage(currentTime, msg)

    if msg.body['msg'] == "SUM_QUERY_RESPONSE":
      log_print("Agent {} received sum query response: {}", self.id, msg)

      # Now schedule a new query.
      self.setWakeup(currentTime + pd.Timedelta('1m'))

