from agent.Agent import Agent
from message.Message import Message
from util.util import log_print


# The SumServiceAgent class inherits from the base Agent class.  It is intended
# to serve as an example in which a service agent performs some aggregated
# computation for multiple clients and returns the result to all clients.

class SumServiceAgent(Agent):

  def __init__(self, id, name, type, random_state=None, num_clients=10):
    # Base class init.
    super().__init__(id, name, type, random_state)

    # How many clients should we wait for?
    self.num_clients = num_clients

    # A list of the numbers to sum: dictionary keyed by agentID.
    self.numbers = {}

    # We track the total sum for the entire day to print at the end.
    self.total = 0


  ### Simulation lifecycle messages.
  def kernelStarting(self, startTime):
    # self.kernel is set in Agent.kernelInitializing()

    # This agent should have negligible computation delay.
    self.setComputationDelay(1000000)    # 1 ms

    # Request a wake-up call as in the base Agent.
    super().kernelStarting(startTime)


  def kernelStopping(self):
    # Print the total sum for the day, only for completed sum requests.
    print("Agent {} reports total sum: {}".format(self.id, self.total))

    # Allow the base class to perform stopping activities.
    super().kernelStopping()


  ### Simulation participation messages.

  # The service agent does not require wakeup calls.

  def receiveMessage (self, currentTime, msg):
    # Allow the base Agent to do whatever it needs to.
    super().receiveMessage(currentTime, msg)

    if msg.body['msg'] == "SUM_QUERY":
      log_print("Agent {} received sum query: {}", self.id, msg)
      self.numbers[msg.body['sender']] = (msg.body['n1'], msg.body['n2'])

    if len(self.numbers.keys()) >= self.num_clients:
      # It is time to sum the numbers.
      self.processSum()

      # Then clear the pending queries.
      self.numbers = {}


  ### Sum client numbers and respond to each client.
  def processSum (self):

    current_sum = sum([ x[0] + x[1] for x in self.numbers.values() ])
    self.total += current_sum

    log_print("Agent {} computed sum: {}", self.id, current_sum)

    for sender in self.numbers.keys():
      self.sendMessage(sender, Message({ "msg" : "SUM_QUERY_RESPONSE", "sender": self.id,
                                         "sum" : current_sum }))


