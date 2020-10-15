from agent.Agent import Agent
from agent.examples.crypto.PPFL_ServiceAgent import PPFL_ServiceAgent
from message.Message import Message
from util.util import log_print

from util.crypto.logReg import getWeights, reportStats
import util.crypto.diffieHellman as dh

import numpy as np
from os.path import exists
import pandas as pd
import random


# The PPFL_TemplateClientAgent class inherits from the base Agent class.  It has the
# structure of a secure federated learning protocol with secure multiparty communication,
# but without any particular learning or noise methods.  That is, this is a template in
# which the client parties simply pass around arbitrary data.  Sections that would need
# to be completed are clearly marked.

class PPFL_TemplateClientAgent(Agent):

  def __init__(self, id, name, type, peer_list=None, iterations=4, multiplier=10000, secret_scale = 100000,
               X_train = None, y_train = None, X_test = None, y_test = None, split_size = None,
               num_clients = None, num_subgraphs = None, random_state=None):

    # Base class init.
    super().__init__(id, name, type, random_state)


    # Store the client's peer list (subgraph, neighborhood) with which it should communicate.
    self.peer_list = peer_list

    # Initialize a tracking attribute for the initial peer exchange and record the subgraph size.
    self.peer_exchange_complete = False
    self.num_peers = len(self.peer_list)

    # Record the total number of clients participating in the protocol and the number of subgraphs.
    # Neither of these are part of the protocol, or necessary for real-world implementation, but do
    # allow for convenient logging of progress and results in simulation.
    self.num_clients = num_clients
    self.num_subgraphs = num_subgraphs

    # Record the number of protocol (federated learning) iterations the clients will perform.
    self.no_of_iterations = iterations

    # Record the multiplier that will be used to protect against floating point accuracy loss and
    # the scale of the client shared secrets.
    self.multiplier = multiplier
    self.secret_scale = secret_scale

    # Record the training and testing splits for the data set to be learned.
    self.X_train = X_train
    self.y_train = y_train

    self.X_test = X_test
    self.y_test = y_test

    # Record the number of features in the data set.
    self.no_of_weights = X_train.shape[1]
    
    # Initialize an attribute to remember the shared weights returned from the server.
    self.prevWeight = np.zeros(self.no_of_weights)

    # Each client receives only a portion of the training data each protocol iteration.
    self.split_size = split_size

    # Initialize a dictionary to remember which peers we have heard from during peer exchange.
    self.peers_received = {}

    # Initialize a dictionary to accumulate this client's timing information by task.
    self.elapsed_time = { 'DH_OFFLINE' : pd.Timedelta(0), 'DH_ONLINE' : pd.Timedelta(0),
                          'TRAINING' : pd.Timedelta(0), 'ENCRYPTION' : pd.Timedelta(0) }


    # Pre-generate this client's local training data for each iteration (for the sake of simulation speed).
    self.trainX = []
    self.trainY = []

    # This is a faster PRNG than the default, for times when we must select a large quantity of randomness.
    self.prng = np.random.Generator(np.random.SFC64())

    ### Data randomly selected from total training set each iteration, simulating online behavior.
    for i in range(iterations):
      slice = self.prng.choice(range(self.X_train.shape[0]), size = split_size, replace = False)

      # Pull together the current local training set.
      self.trainX.append(self.X_train[slice].copy())
      self.trainY.append(self.y_train[slice].copy())


    # Create dictionaries to hold the public and secure keys for this client, and the public keys shared
    # by its peers.
    self.pubkeys = {}
    self.seckeys = {}
    self.peer_public_keys = {}

    # Create dictionaries to hold the shared key for each peer each iteration and the seed for the
    # following iteration.
    self.r = {}
    self.R = {}


    ### ADD DIFFERENTIAL PRIVACY CONSTANTS AND CONFIGURATION HERE, IF NEEDED.
    #
    #


    # Iteration counter.
    self.current_iteration = 0




  ### Simulation lifecycle messages.

  def kernelStarting(self, startTime):

    # Initialize custom state properties into which we will later accumulate results.
    # To avoid redundancy, we allow only the first client to handle initialization.
    if self.id == 1:
      self.kernel.custom_state['dh_offline'] = pd.Timedelta(0)
      self.kernel.custom_state['dh_online'] = pd.Timedelta(0)
      self.kernel.custom_state['training'] = pd.Timedelta(0)
      self.kernel.custom_state['encryption'] = pd.Timedelta(0)

    # Find the PPFL service agent, so messages can be directed there.
    self.serviceAgentID = self.kernel.findAgentByType(PPFL_ServiceAgent)

    # Request a wake-up call as in the base Agent.  Noise is kept small because
    # the overall protocol duration is so short right now.  (up to one microsecond)
    super().kernelStarting(startTime + pd.Timedelta(self.random_state.randint(low = 0, high = 1000), unit='ns'))


  def kernelStopping(self):

    # Accumulate into the Kernel's "custom state" this client's elapsed times per category.
    # Note that times which should be reported in the mean per iteration are already so computed.
    # These will be output to the config (experiment) file at the end of the simulation.

    self.kernel.custom_state['dh_offline'] += self.elapsed_time['DH_OFFLINE']
    self.kernel.custom_state['dh_online'] += (self.elapsed_time['DH_ONLINE'] / self.no_of_iterations)
    self.kernel.custom_state['training'] += (self.elapsed_time['TRAINING'] / self.no_of_iterations)
    self.kernel.custom_state['encryption'] += (self.elapsed_time['ENCRYPTION'] / self.no_of_iterations)

    super().kernelStopping()


  ### Simulation participation messages.

  def wakeup (self, currentTime):
    super().wakeup(currentTime)

    # Record start of wakeup for real-time computation delay..
    dt_wake_start = pd.Timestamp('now')

    # Check if the clients are still performing the one-time peer exchange.
    if not self.peer_exchange_complete:    

      # Generate DH keys.
      self.pubkeys, self.seckeys = dh.dict_keygeneration( self.peer_list )

      # Record elapsed wallclock for Diffie Hellman offline.
      dt_wake_end = pd.Timestamp('now')
      self.elapsed_time['DH_OFFLINE'] += dt_wake_end - dt_wake_start

      # Set computation delay to elapsed wallclock time.
      self.setComputationDelay(int((dt_wake_end - dt_wake_start).to_timedelta64()))

      # Send generated values to peers.
      for idx, peer in enumerate(self.peer_list):
        # We assume a star network configuration where all messages between peers must be forwarded
        # through the server.
        self.sendMessage(self.serviceAgentID, Message({ "msg" : "FWD_MSG", "msgToForward" : "PEER_EXCHANGE",
                        "sender": self.id, "recipient": peer, "pubkey" : self.pubkeys[peer] }))

    else: 
       
      # We are waking up to start a new iteration of the protocol.
      # (Peer exchange is done before all this.)

      if (self.current_iteration == 0):
        # During iteration 0 (only) we complete the key exchange and prepare the
        # common key list, because at this point we know we have received keys
        # from all peers.

        # R is the common key dictionary (by peer agent id).
        dh.dict_keyexchange(self.peer_list, self.id, self.pubkeys, self.seckeys, self.peer_public_keys)


        # CREATE AND CACHE LOCAL DIFFERENTIAL PRIVACY NOISE HERE, IF NEEDED.
        #
        #


      # Diffie Hellman is done in every iteration.
      for peer_id, common_key in self.R.items():

        random.seed(common_key)
        rand = random.getrandbits(512)

        rand_b_raw  = format(rand, '0512b')
        rand_b_rawr = rand_b_raw[:256]
        rand_b_rawR = rand_b_raw[256:]


        # Negate offsets below this agent's id.  This ensures each offset will be
        # added once and subtracted once.
        r = int(rand_b_rawr,2) % (2**32)

        # Update dictionary of shared secrets for this iteration.
        self.r[peer_id] = r if peer_id < self.id else -r

        # Store the shared seeds for the next iteration.
        self.R[peer_id] = int(rand_b_rawR,2)


      # Record elapsed wallclock for Diffie Hellman online.
      dt_online_complete = pd.Timestamp('now')

      # For convenience of things indexed by iteration...
      i = self.current_iteration


      ### ADD LOCAL LEARNING METHOD HERE, IF NEEDED.
      #
      #

      weight = np.random.normal (loc = self.prevWeight, scale = self.prevWeight / 10, size = self.prevWeight.shape)


      # Record elapsed wallclock for training model.
      dt_training_complete = pd.Timestamp('now')


      ### ADD NOISE TO THE WEIGHTS HERE, IF NEEDED.
      #
      #

      n = np.array(weight) * self.multiplier
      weights_to_send = n + sum(self.r.values())

      
      # Record elapsed wallclock for encryption.
      dt_encryption_complete = pd.Timestamp('now')

      # Set computation delay to elapsed wallclock time.
      self.setComputationDelay(int((dt_encryption_complete - dt_wake_start).to_timedelta64()))

      # Send the message to the server.
      self.sendMessage(self.serviceAgentID, Message({ "msg" : "CLIENT_WEIGHTS", "sender": self.id,
                                                      "weights" : weights_to_send }))
    
      self.current_iteration += 1

      # Store elapsed times by category.
      self.elapsed_time['DH_ONLINE'] += dt_online_complete - dt_wake_start
      self.elapsed_time['TRAINING'] += dt_training_complete - dt_online_complete
      self.elapsed_time['ENCRYPTION'] += dt_encryption_complete - dt_training_complete



  def receiveMessage (self, currentTime, msg):
    super().receiveMessage(currentTime, msg)

    if msg.body['msg'] == "PEER_EXCHANGE":

      # Record start of message processing.
      dt_rcv_start = pd.Timestamp('now')

      # Ensure we don't somehow record the same peer twice.  These all come from the
      # service provider, relayed from other clients, but are "fixed up" to appear
      # as if they come straight from the relevant peer.
      if msg.body['sender'] not in self.peers_received:

        # Record the content of the message and that we received it.
        self.peers_received[msg.body['sender']] = True
        self.peer_public_keys[msg.body['sender']] = msg.body['pubkey']

        # Record end of message processing.
        dt_rcv_end = pd.Timestamp('now')

        # Store elapsed times by category.
        self.elapsed_time['DH_OFFLINE'] += dt_rcv_end - dt_rcv_start

        # Set computation delay to elapsed wallclock time.
        self.setComputationDelay(int((dt_rcv_end - dt_rcv_start).to_timedelta64()))

        # If this is the last peer from whom we expect to hear, move on with the protocol.
        if len(self.peers_received) == self.num_peers:
          self.peer_exchange_complete = True
          self.setWakeup(currentTime + pd.Timedelta('1ns'))

    elif msg.body['msg'] == "SHARED_WEIGHTS":
      # Reset computation delay.
      self.setComputationDelay(0)

      # Extract the shared weights from the message.
      self.prevWeight = msg.body['weights']

      # Remove the multiplier that was helping guard against floating point error.
      self.prevWeight /= self.multiplier

      log_print ("Client weights received for iteration {} by {}: {}", self.current_iteration, self.id, self.prevWeight)

      if self.id == 1: print (f"Protocol iteration {self.current_iteration} complete.")

      # Start a new iteration if we are not at the end of the protocol.
      if self.current_iteration < self.no_of_iterations:
        self.setWakeup(currentTime + pd.Timedelta('1ns'))

