from agent.Agent import Agent
from message.Message import Message
from util.util import log_print
from util.crypto.logReg import getWeights

from copy import deepcopy
import numpy as np
import pandas as pd

### NEW MESSAGES: CLIENT_WEIGHTS : weights, SHARED_WEIGHTS : weights

# The PPFL_ServiceAgent class inherits from the base Agent class.  It provides
# the simple shared service necessary for model combination under secure
# federated learning.

class PPFL_ServiceAgent(Agent):

  def __init__(self, id, name, type, random_state=None, msg_fwd_delay=1000000,
               iterations=4, num_clients=10):

    # Base class init.
    super().__init__(id, name, type, random_state)

    # From how many clients do we expect to hear in each protocol iteration?
    self.num_clients = num_clients

    # How long does it take us to forward a peer-to-peer client relay message?
    self.msg_fwd_delay = msg_fwd_delay
    
    # Agent accumulation of elapsed times by category of task.
    self.elapsed_time = { 'STORE_MODEL' : pd.Timedelta(0), 'COMBINE_MODEL' : pd.Timedelta(0) }

    # How many iterations of the protocol should be run?
    self.no_of_iterations = iterations

    # Create a dictionary keyed by agentID to record which clients we have
    # already heard from during the current protocol iteration.  This can
    # also guard against double accumulation from duplicate messages.
    self.received = {}

    # Create a list to accumulate received values.  We don't need to know which came from which
    # client, what the values mean, or indeed anything about them.
    self.total = []

    # Track the current iteration of the protocol.
    self.current_iteration = 0


  ### Simulation lifecycle messages.
  def kernelStarting(self, startTime):
    # self.kernel is set in Agent.kernelInitializing()

    # Initialize custom state properties into which we will accumulate results later.
    self.kernel.custom_state['srv_store_model'] = pd.Timedelta(0)
    self.kernel.custom_state['srv_combine_model'] = pd.Timedelta(0)

    # This agent should have negligible (or no) computation delay until otherwise specified.
    self.setComputationDelay(0)

    # Request a wake-up call as in the base Agent.
    super().kernelStarting(startTime)


  def kernelStopping(self):
    # Add the server time components to the custom state in the Kernel, for output to the config.
    # Note that times which should be reported in the mean per iteration are already so computed.
    self.kernel.custom_state['srv_store_model'] += (self.elapsed_time['STORE_MODEL'] / self.no_of_iterations)
    self.kernel.custom_state['srv_combine_model'] += (self.elapsed_time['COMBINE_MODEL'] / self.no_of_iterations)

    # Allow the base class to perform stopping activities.
    super().kernelStopping()
    

  ### Simulation participation messages.

  # The service agent does not require wakeup calls.

  def receiveMessage (self, currentTime, msg):
    # Allow the base Agent to do whatever it needs to.
    super().receiveMessage(currentTime, msg)

    # Logic for receiving weights from client agents.  The weights are almost certainly
    # noisy and encrypted, but that doesn't matter to us.
    if msg.body['msg'] == "CLIENT_WEIGHTS":
      
      # Start wallclock timing for message handling.
      dt_combine_complete = None
      dt_start_rcv = pd.Timestamp('now')

      sender = msg.body['sender']
      if sender in self.received: return

      self.received[sender] = True
      self.total.append(msg.body['weights'].tolist())

      # Capture elapsed wallclock for model storage.
      dt_store_complete = pd.Timestamp('now')

      log_print ("Server received {} from {}.", msg.body['weights'], msg.body['sender'])

      if len(self.received.keys()) >= self.num_clients:
        # This is the last client on whom we were waiting.
        self.combineWeights()

        # Capture elapsed wallclock for model combination.
        dt_combine_complete = pd.Timestamp('now')

        # Then clear the protocol attributes for the next round.
        self.received = {}
        self.total = []

      # Capture elapsed wallclock at end of CLIENT_WEIGHTS.
      dt_end_rcv = pd.Timestamp('now')

      # Compute time deltas only after all elapsed times are captured.
      if dt_combine_complete is not None: self.elapsed_time['COMBINE_MODEL'] += dt_combine_complete - dt_store_complete
      self.elapsed_time['STORE_MODEL'] += dt_store_complete - dt_start_rcv
      elapsed_total = int((dt_end_rcv - dt_start_rcv).to_timedelta64())

      # Use total elapsed wallclock as computation delay.
      self.setComputationDelay(elapsed_total)

    elif msg.body['msg'] == "FWD_MSG":
      # In our star topology, all client messages are forwarded securely through the server.
      sender = msg.body['sender']
      recipient = msg.body['recipient']
      msg.body['msg'] = msg.body['msgToForward']

      self.setComputationDelay(self.msg_fwd_delay)

      # Normally not advisable, but here we need to fix up the sender so the
      # server can be a silent proxy.
      self.kernel.sendMessage(sender, recipient, msg)


  ### Combine client weights and respond to each client.
  def combineWeights (self):

    log_print ("total: {}", self.total)

    # Don't respond after the final iteration.
    if (self.current_iteration < self.no_of_iterations):

      # Take the mean weights across the clients.
      self.total = np.array(self.total)
      totals = np.mean(self.total, axis=0)

      # Send the combined weights back to each client who participated.
      for sender in self.received.keys():
        log_print ("Sending {} to {}", totals, sender)
        self.sendMessage(sender, Message({ "msg" : "SHARED_WEIGHTS", "sender": self.id, "weights" : deepcopy(totals) }))

      # This is the end of one round of the protocol.
      self.current_iteration += 1

