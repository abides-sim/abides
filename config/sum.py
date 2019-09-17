from Kernel import Kernel
from agent.examples.SumClientAgent import SumClientAgent
from agent.examples.SumServiceAgent import SumServiceAgent
from util import util

import numpy as np
import pandas as pd
import sys


# Some config files require additional command line parameters to easily
# control agent or simulation hyperparameters during coarse parallelization.
import argparse

parser = argparse.ArgumentParser(description='Detailed options for example sum config.')
parser.add_argument('-c', '--config', required=True,
                    help='Name of config file to execute')
parser.add_argument('-l', '--log_dir', default=None,
                    help='Log directory name (default: unix timestamp at program start)')
parser.add_argument('-s', '--seed', type=int, default=None,
                    help='numpy.random.seed() for simulation')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='Maximum verbosity!')
parser.add_argument('--config_help', action='store_true',
                    help='Print argument options for this config file')

args, remaining_args = parser.parse_known_args()

if args.config_help:
  parser.print_help()
  sys.exit()

# Historical date to simulate.  Required even if not relevant.
historical_date = pd.to_datetime('2014-01-28')

# Requested log directory.
log_dir = args.log_dir

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


### Configure the agents.  When conducting "agent of change" experiments, the
### new agents should be added at the END only.
agent_count = 0
agents = []
agent_types = []


### How many client agents will there be?
num_clients = 10


### Configure a sum service agent.

agents.extend([ SumServiceAgent(0, "Sum Service Agent 0", "SumServiceAgent",
                random_state = np.random.RandomState(seed=np.random.randint(low=0,high=2**32)),
                num_clients = num_clients) ])
agent_types.extend(["SumServiceAgent"])
agent_count += 1



### Configure a population of sum client agents.
a, b = agent_count, agent_count + num_clients

agents.extend([ SumClientAgent(i, "Sum Client Agent {}".format(i), "SumClientAgent", peer_list = [ x for x in range(a,b) if x != i ], random_state = np.random.RandomState(seed=np.random.randint(low=0,high=2**32))) for i in range(a,b) ])
agent_types.extend([ "SumClientAgent" for i in range(a,b) ])
agent_count += num_clients



### Configure a simple message latency matrix for the agents.  Each entry is the minimum
### nanosecond delay on communication [from][to] agent ID.

# Square numpy array with dimensions equal to total agent count.  Most agents are handled
# at init, drawn from a uniform distribution from:
# Times Square (3.9 miles from NYSE, approx. 21 microseconds at the speed of light) to:
# Pike Place Starbucks in Seattle, WA (2402 miles, approx. 13 ms at the speed of light).
# Other agents can be explicitly set afterward (and the mirror half of the matrix is also).

# This configures all agents to a starting latency as described above.
latency = np.random.uniform(low = 21000, high = 13000000, size=(len(agent_types),len(agent_types)))

# Overriding the latency for certain agent pairs happens below, as does forcing mirroring
# of the matrix to be symmetric.
for i, t1 in zip(range(latency.shape[0]), agent_types):
  for j, t2 in zip(range(latency.shape[1]), agent_types):
    # Three cases for symmetric array.  Set latency when j > i, copy it when i > j, same agent when i == j.
    # The j > i case is handled in the initialization above, unless we need to override specific agents.
    if i > j:
      # This "bottom" half of the matrix simply mirrors the top.
      latency[i,j] = latency[j,i]
    elif i == j:
      # This is the same agent.  How long does it take to reach localhost?  In our data center, it actually
      # takes about 20 microseconds.
      latency[i,j] = 20000


# Configure a simple latency noise model for the agents.
# Index is ns extra delay, value is probability of this delay being applied.
noise = [ 0.25, 0.25, 0.20, 0.15, 0.10, 0.05 ]



# Start the kernel running.
kernel.runner(agents = agents, startTime = kernelStartTime, stopTime = kernelStopTime,
              agentLatency = latency, latencyNoise = noise,
              defaultComputationDelay = defaultComputationDelay,
              log_dir = log_dir)

