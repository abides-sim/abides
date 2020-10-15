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


# The PPFL_ClientAgent class inherits from the base Agent class.  It implements
# a secure federated learning protocol with basic differential privacy plus
# secure multiparty communication.

class PPFL_ClientAgent(Agent):

  def __init__(self, id, name, type, peer_list=None, iterations=4, multiplier=10000, secret_scale = 100000,
               X_train = None, y_train = None, X_test = None, y_test = None, split_size = None,
               learning_rate = None, clear_learning = None, num_clients = None, num_subgraphs = None,
               epsilon = None, max_logreg_iterations = None, collusion = False, random_state=None):

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

    # Record whether the clients should be recording information about the potential accuracy of
    # peer data reconstruction via collusion among the clients.
    self.collusion = collusion

    # Record the number of protocol (federated learning) iterations the clients will perform.
    self.no_of_iterations = iterations

    # Record the multiplier that will be used to protect against floating point accuracy loss and
    # the scale of the client shared secrets.
    self.multiplier = multiplier
    self.secret_scale = secret_scale

    # Record the number of local iterations of logistic regression each client will run during
    # each protocol iteration and what local learning rate will be used.
    self.max_logreg_iterations = max_logreg_iterations
    self.learning_rate = learning_rate

    # Record whether clients will do federated learning in the clear (no privacy, no encryption)
    # and, if needed, the epsilon value for differential privacy.
    self.clear_learning = clear_learning
    self.epsilon = epsilon
    
    # Record the training and testing splits for the data set to be learned.
    self.X_train = X_train
    self.y_train = y_train

    self.X_test = X_test
    self.y_test = y_test

    # Record the number of features in the data set.
    self.no_of_weights = X_train.shape[1]
    
    # Initialize an attribute to remember the shared weights returned from the server.
    self.prevWeight = None

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


    # Specify the parameters used for generation of randomness.
    self.px_reg = 1
    self.px_epsilon = epsilon
    self.px_min_rows = self.split_size

    self.px_shape = 1 / ( self.num_peers + 1)
    self.px_scale = 2 / (( self.num_peers + 1 ) * self.px_min_rows * self.px_reg * self.px_epsilon )

    if self.id == 1: print (f"px_shape is {self.px_shape}")
    if self.id == 1: print (f"px_scale is {self.px_scale}")

    # Specify the required shape for vectorized generation of randomness.
    self.px_dims = ( self.num_peers, self.no_of_iterations, self.no_of_weights )


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
      if not self.clear_learning: self.pubkeys, self.seckeys = dh.dict_keygeneration( self.peer_list )

      # Record elapsed wallclock for Diffie Hellman offline.
      dt_wake_end = pd.Timestamp('now')
      self.elapsed_time['DH_OFFLINE'] += dt_wake_end - dt_wake_start

      # Set computation delay to elapsed wallclock time.
      self.setComputationDelay(int((dt_wake_end - dt_wake_start).to_timedelta64()))

      # Send generated values to peers.
      if not self.clear_learning:
        for idx, peer in enumerate(self.peer_list):
          # We assume a star network configuration where all messages between peers must be forwarded
          # through the server.
          self.sendMessage(self.serviceAgentID, Message({ "msg" : "FWD_MSG", "msgToForward" : "PEER_EXCHANGE",
                          "sender": self.id, "recipient": peer, "pubkey" : self.pubkeys[peer] }))

      if self.clear_learning:
        self.peer_exchange_complete = True
        self.setWakeup(currentTime + pd.Timedelta('1ns'))
      
    else: 
       
      # We are waking up to start a new iteration of the protocol.
      # (Peer exchange is done before all this.)

      if (self.current_iteration == 0):
        # During iteration 0 (only) we complete the key exchange and prepare the
        # common key list, because at this point we know we have received keys
        # from all peers.

        # R is the common key dictionary (by peer agent id).
        if not self.clear_learning: self.R = dh.dict_keyexchange(self.peer_list, self.id, self.pubkeys,
                                                                 self.seckeys, self.peer_public_keys)

        # Pre-generate all of this client's local differential privacy noise (for simulation speed).
        # We will need one per weight per protocol iteration.
        self.my_noise = np.random.laplace(scale = self.px_scale, size = (self.no_of_iterations, self.no_of_weights))


      # Diffie Hellman is done in every iteration.
      if not self.clear_learning:
        for peer_id, common_key in self.R.items():

          random.seed(common_key)
          rand = random.getrandbits(512)

          rand_b_raw  = format(rand, '0512b')
          rand_b_rawr = rand_b_raw[:256]
          rand_b_rawR = rand_b_raw[256:]


          # Negate offsets below this agent's id.  This ensures each offset will be
          # added once and subtracted once.
          r = int(rand_b_rawr,2) % (2**32)

          log_print ("SELECTED r: {}", r)

          # Update dictionary of shared secrets for this iteration.
          self.r[peer_id] = r if peer_id < self.id else -r

          # Store the shared seeds for the next iteration.
          self.R[peer_id] = int(rand_b_rawR,2)


      # Record elapsed wallclock for Diffie Hellman online.
      dt_online_complete = pd.Timestamp('now')

      # For convenience of things indexed by iteration...
      i = self.current_iteration

      # Perform the local training for this client, using only its local (private) data.  The configured learning
      # rate might need to be increased if there are very many clients, each with very little data, otherwise
      # convergence may take a really long time.
      #
      # max_iter controls how many iterations of gradient descent to perform on the logistic
      # regression model.  previous_weight should be passed as None for the first iteration.
      weight = getWeights(previous_weight = self.prevWeight, max_iter = self.max_logreg_iterations, lr = self.learning_rate,
                          trainX = self.trainX[i], trainY = self.trainY[i], self_id = self.id)

      # If in collusion analysis mode, write out the weights we will need to evaluate reconstruction.
      if self.collusion:
        with open('results/collusion_weights.csv', 'a') as results_file:
          results_file.write(f"{self.id},{self.current_iteration},{','.join([str(x) for x in weight])}\n")

      # Record elapsed wallclock for training model.
      dt_training_complete = pd.Timestamp('now')

      if not self.clear_learning:
        # Add a random sample from Laplace to each of the weights.
        noise = self.my_noise[i]

        if self.collusion:
          with open('results/collusion_selected_noise.csv', 'a') as results_file:
            # Write out the noise added to each weight by this client.
            results_file.write(f"{self.id},{self.current_iteration},{','.join([str(x) for x in noise])}\n")

        log_print ("weight {}", weight)
        log_print ("noise {}", noise)

      if self.clear_learning: n = np.array(weight) * self.multiplier
      else: n = (np.array(weight) + noise) * self.multiplier

      log_print ("n {}", n)
      log_print ("r {}", self.r)

      weights_to_send = n + sum(self.r.values())

      log_print ("weights_to_send {}", weights_to_send)

      
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

      # Client number 1 (arbitrary choice) records the shared learning progress each iteration
      # for later visualization and analysis.
      if self.id == 1:
        is_acc, is_mcc, is_f1, is_mse, is_auprc, oos_acc, oos_mcc, oos_f1, oos_mse, oos_auprc = reportStats(self.prevWeight, self.current_iteration, self.X_train, self.y_train, self.X_test, self.y_test)

        if not exists("results/all_results.csv"):
          with open('results/all_results.csv', 'a') as results_file:
            # Write out the header.
            results_file.write(f"Clients,Peers,Subgraphs,Iterations,Train Rows,Learning Rate,In The Clear?,Local Iterations,Epsilon,Iteration,IS ACC,OOS ACC,IS MCC,OOS MCC,IS MSE,OOS MSE,IS F1,OOS F1,IS AUPRC,OOS AUPRC\n")

        with open('results/all_results.csv', 'a') as results_file:
          # Write out the current protocol iteration weights and metadata.
          results_file.write(f"{self.num_clients},{self.num_peers},{self.num_subgraphs},{self.no_of_iterations},{self.split_size},{self.learning_rate},{self.clear_learning},{self.max_logreg_iterations},{self.epsilon},{self.current_iteration},{is_acc},{oos_acc},{is_mcc},{oos_mcc},{is_mse},{oos_mse},{is_f1},{oos_f1},{is_auprc},{oos_auprc}\n")

        if self.collusion:
          with open('results/collusion_consensus.csv', 'a') as results_file:
            # Agent 1 also writes out the consensus weights each iteration (for collusion analysis).
            results_file.write(f"{self.current_iteration},{','.join([str(x) for x in self.prevWeight])}\n")


      # Start a new iteration if we are not at the end of the protocol.
      if self.current_iteration < self.no_of_iterations:
        self.setWakeup(currentTime + pd.Timedelta('1ns'))

