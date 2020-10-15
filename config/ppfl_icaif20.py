# Our custom modules.
from Kernel import Kernel
from agent.examples.crypto.PPFL_ClientAgent import PPFL_ClientAgent
from agent.examples.crypto.PPFL_ServiceAgent import PPFL_ServiceAgent
from model.LatencyModel import LatencyModel
from util import util
from util.crypto import logReg

# Standard modules.
from datetime import timedelta
from math import floor
import numpy as np
from os.path import exists
import pandas as pd
from sklearn.model_selection import train_test_split
from sys import exit
from time import time


# Some config files require additional command line parameters to easily
# control agent or simulation hyperparameters during coarse parallelization.
import argparse

parser = argparse.ArgumentParser(description='Detailed options for PPFL config.')
parser.add_argument('-a', '--clear_learning', action='store_true',
                    help='Learning in the clear (vs SMP protocol)')
parser.add_argument('-c', '--config', required=True,
                    help='Name of config file to execute')
parser.add_argument('-e', '--epsilon', type=float, default=1.0,
                    help='Privacy loss epsilon')
parser.add_argument('-g', '--num_subgraphs', type=int, default=1,
                    help='Number of connected subgraphs into which to place client agents')
parser.add_argument('-i', '--num_iterations', type=int, default=5,
                    help='Number of iterations for the secure multiparty protocol)')
parser.add_argument('-k', '--skip_log', action='store_true',
                    help='Skip writing agent logs to disk')
parser.add_argument('-l', '--log_dir', default=None,
                    help='Log directory name (default: unix timestamp at program start)')
parser.add_argument('-m', '--max_logreg_iterations', type=int, default=50,
                    help='Number of iterations for client local LogReg'),
parser.add_argument('-n', '--num_clients', type=int, default=5,
                    help='Number of clients for the secure multiparty protocol)')
parser.add_argument('-o', '--collusion', action='store_true',
                    help='Compute collusion analysis (big and slow!)')
parser.add_argument('-p', '--split_size', type=int, default=20,
                    help='Local training size per client per iteration')
parser.add_argument('-r', '--learning_rate', type=float, default=10.0,
                    help='Local learning rate for training on client data')
parser.add_argument('-s', '--seed', type=int, default=None,
                    help='numpy.random.seed() for simulation')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='Maximum verbosity!')
parser.add_argument('--config_help', action='store_true',
                    help='Print argument options for this config file')

args, remaining_args = parser.parse_known_args()

if args.config_help:
  parser.print_help()
  exit()

# Historical date to simulate.  Required even if not relevant.
historical_date = pd.to_datetime('2014-01-28')

# Requested log directory.
log_dir = args.log_dir
skip_log = args.skip_log

# Random seed specification on the command line.  Default: None (by clock).
# If none, we select one via a specific random method and pass it to seed()
# so we can record it for future use.  (You cannot reasonably obtain the
# automatically generated seed when seed() is called without a parameter.)

# Note that this seed is used to (1) make any random decisions within this
# config file itself and (2) to generate random number seeds for the
# (separate) Random objects given to each agent.  This ensure that when
# the agent population is appended, prior agents will continue to behave
# in the same manner save for influences by the new agents.  (i.e. all prior
# agents still have their own separate PRNG sequence, and it is the same as
# before)

seed = args.seed
if not seed: seed = int(pd.Timestamp.now().timestamp() * 1000000) % (2**32 - 1)
np.random.seed(seed)

# Config parameter that causes util.util.print to suppress most output.
util.silent_mode = not args.verbose

num_clients = args.num_clients
num_iterations = args.num_iterations
split_size = args.split_size
max_logreg_iterations = args.max_logreg_iterations
epsilon = args.epsilon
learning_rate = args.learning_rate
clear_learning = args.clear_learning
collusion = args.collusion

### How many client agents will there be?   1000 in 125 subgraphs of 8 fits ln(n), for example
num_subgraphs = args.num_subgraphs

print ("Silent mode: {}".format(util.silent_mode))
print ("Configuration seed: {}\n".format(seed))



# Since the simulator often pulls historical data, we use a real-world
# nanosecond timestamp (pandas.Timestamp) for our discrete time "steps",
# which are considered to be nanoseconds.  For other (or abstract) time
# units, one can either configure the Timestamp interval, or simply
# interpret the nanoseconds as something else.

# What is the earliest available time for an agent to act during the
# simulation?
midnight = historical_date
kernelStartTime = midnight

# When should the Kernel shut down?
kernelStopTime = midnight + pd.to_timedelta('17:00:00')

# This will configure the kernel with a default computation delay
# (time penalty) for each agent's wakeup and recvMsg.  An agent
# can change this at any time for itself.  (nanoseconds)

defaultComputationDelay = 1000000000 * 5   # five seconds

# IMPORTANT NOTE CONCERNING AGENT IDS: the id passed to each agent must:
#    1. be unique
#    2. equal its index in the agents list
# This is to avoid having to call an extra getAgentListIndexByID()
# in the kernel every single time an agent must be referenced.


### Configure the Kernel.
kernel = Kernel("Base Kernel", random_state = np.random.RandomState(seed=np.random.randint(low=0,high=2**32)))

### Obtain random state for whatever latency model will be used.
latency_rstate = np.random.RandomState(seed=np.random.randint(low=0,high=2**32))

### Obtain a seed for the train-test split shuffling.
shuffle_seed = np.random.randint(low=0,high=2**32)

### Configure the agents.  When conducting "agent of change" experiments, the
### new agents should be added at the END only.
agent_count = 0
agents = []
agent_types = []

### What accuracy multiplier will be used?
accy_multiplier = 100000

### What will be the scale of the shared secret?
secret_scale = 1000000

### For now, non-integer sizes are NOT handled.  Please choose an even divisor.
subgraph_size = int(floor(num_clients / num_subgraphs))

logReg.number_of_parties = num_clients


# Load in the data the clients must learn, once.

# Note that time and amount columns are not preprocessed, which might affect
# how we would like to approach them.  We exclude Time (time since first record)
# since we are not using a time-sensitive method.
nd1 = np.loadtxt('util/crypto/datasets/creditcard/creditcard.csv', delimiter=',', skiprows=1)
X_data = nd1[:,1:-1]
y_data = nd1[:,-1]

# We add a feature zero, always with value 1, to allow the intercept to be just
# another weight for purposes of vector math.
X_data = np.insert(X_data, 0, 1.0, axis=1)

print (X_data.shape,y_data.shape)
print (np.unique(y_data))


# Randomly shuffle and split the data for training and testing.
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.25, random_state = shuffle_seed)


### Configure a service agent.

agents.extend([ PPFL_ServiceAgent(0, "PPFL Service Agent 0", "PPFL_ServiceAgent",
                random_state = np.random.RandomState(seed=np.random.randint(low=0,high=2**32)),
                msg_fwd_delay=0, iterations = num_iterations, num_clients = num_clients) ])
agent_types.extend(["PPFL_ServiceAgent"])
agent_count += 1


### Configure a population of cooperating learning client agents.
a, b = agent_count, agent_count + num_clients

client_init_start = time()

# Iterate over all client IDs.
for i in range (a, b):

  # Determine subgraph.
  subgraph = int(floor((i - a) / subgraph_size))

  #print ("Neighborhood for agent {} is {}".format(i, subgraph))

  # Determine agents in subgraph.
  subgraph_start = a + (subgraph * subgraph_size)
  subgraph_end = a + ((subgraph + 1) * subgraph_size)

  neighbors = range(subgraph_start, subgraph_end)

  #print ("Peers for {} are {}".format(i, [x for x in neighbors if x != i]))

  # Peer list is all agents in subgraph except self.
  agents.append(PPFL_ClientAgent(i, "PPFL Client Agent {}".format(i), "PPFL_ClientAgent",
                peer_list = [ x for x in neighbors if x != i ], iterations = num_iterations,
                max_logreg_iterations = max_logreg_iterations, epsilon = epsilon, learning_rate = learning_rate,
                clear_learning = clear_learning, num_clients = num_clients, num_subgraphs = num_subgraphs,
                multiplier = accy_multiplier, X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test,
                split_size = split_size, secret_scale = secret_scale, collusion = collusion,
                random_state = np.random.RandomState(seed=np.random.randint(low=0,high=2**32))))

agent_types.extend([ "PPFL_ClientAgent" for i in range(a,b) ])
agent_count += num_clients

client_init_end = time()
init_seconds = client_init_end - client_init_start
td_init = timedelta(seconds = init_seconds)
print (f"Client init took {td_init}")


### Configure a latency model for the agents.

# Get a new-style cubic LatencyModel from the networking literature.
pairwise = (len(agent_types),len(agent_types))

model_args = { 'connected'   : True,

               # All in NYC.  Only matters for evaluating "real world" protocol duration,
               # not for accuracy, collusion, or reconstruction.
               'min_latency' : np.random.uniform(low = 21000, high = 100000, size = pairwise),
               'jitter'      : 0.3,
               'jitter_clip' : 0.05,
               'jitter_unit' : 5,
             }

latency_model = LatencyModel ( latency_model = 'cubic', random_state = latency_rstate, kwargs = model_args )


# Start the kernel running.
results = kernel.runner(agents = agents, startTime = kernelStartTime, stopTime = kernelStopTime,
                        agentLatencyModel = latency_model,
                        defaultComputationDelay = defaultComputationDelay,
                        skip_log = skip_log, log_dir = log_dir)


# Print parameter summary and elapsed times by category for this experimental trial.
print ()
print (f"Protocol Iterations: {num_iterations}, Clients: {num_clients}, Split Size: {split_size}, " \
       f"Local Iterations {max_logreg_iterations}, Learning Rate: {learning_rate}.")
print (f"Learning in the clear? {clear_learning}, Privacy Epsilon: {epsilon}.")
print ()
print ("Service Agent mean time per iteration...")
print (f"    Storing models:   {results['srv_store_model'] / num_iterations}")
print (f"    Combining models: {results['srv_combine_model'] / num_iterations}")
print ()
print ("Client Agent mean time per iteration (except DH Offline)...")
print (f"    DH Offline: {results['dh_offline'] / num_clients}")
print (f"    DH Online:  {results['dh_online'] / num_clients}")
print (f"    Training:   {results['training'] / num_clients}")
print (f"    Encryption: {results['encryption'] / num_clients}")
print ()
print (f"Slowest agent simulated time: {results['kernel_slowest_agent_finish_time']}")


# Write out the timing log to disk.
if not exists("results/timing_log.csv"):
  with open('results/timing_log.csv', 'a') as results_file:
    results_file.write(f"Clients,Peers,Subgraphs,Iterations,Train Rows,Learning Rate,In The Clear?,Local Iterations,Epsilon,DH Offline,DH Online,Training,Encryption,Store Model,Combine Model,Last Agent Finish,Time to Simulate\n")

  with open('results/timing_log.csv', 'a') as results_file:
    results_file.write(f"{num_clients},{subgraph_size-1},{num_subgraphs},{num_iterations},{split_size},{learning_rate},{clear_learning},{max_logreg_iterations},{epsilon},{results['dh_offline'] / num_clients},{results['dh_online'] / num_clients},{results['training'] / num_clients},{results['encryption'] / num_clients},{results['srv_store_model']},{results['srv_combine_model']},{results['kernel_event_queue_elapsed_wallclock']},{results['kernel_slowest_agent_finish_time']}\n")


