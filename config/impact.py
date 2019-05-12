from Kernel import Kernel
from agent.ExchangeAgent import ExchangeAgent
from agent.HeuristicBeliefLearningAgent import HeuristicBeliefLearningAgent
from agent.ImpactAgent import ImpactAgent
from agent.ZeroIntelligenceAgent import ZeroIntelligenceAgent
from util.order import LimitOrder
from util.oracle.MeanRevertingOracle import MeanRevertingOracle
from util import util

import datetime as dt
import numpy as np
import pandas as pd
import sys


DATA_DIR = "~/data"


# Some config files require additional command line parameters to easily
# control agent or simulation hyperparameters during coarse parallelization.
import argparse

parser = argparse.ArgumentParser(description='Detailed options for momentum config.')
parser.add_argument('-b', '--book_freq', default=None,
                    help='Frequency at which to archive order book for visualization')
parser.add_argument('-c', '--config', required=True,
                    help='Name of config file to execute')
parser.add_argument('-g', '--greed', type=float, default=0.25,
                    help='Impact agent greed')
parser.add_argument('-i', '--impact', action='store_false',
                    help='Do not actually fire an impact trade.')
parser.add_argument('-l', '--log_dir', default=None,
                    help='Log directory name (default: unix timestamp at program start)')
parser.add_argument('-n', '--obs_noise', type=float, default=1000000,
                    help='Observation noise variance for zero intelligence agents (sigma^2_n)')
parser.add_argument('-r', '--shock_variance', type=float, default=500000,
                    help='Shock variance for mean reversion process (sigma^2_s)')
parser.add_argument('-o', '--log_orders', action='store_true',
                    help='Log every order-related action by every agent.')
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

# Requested order book snapshot archive frequency.
book_freq = args.book_freq

# Observation noise variance for zero intelligence agents.
sigma_n = args.obs_noise

# Shock variance of mean reversion process.
sigma_s = args.shock_variance

# Impact agent greed.
greed = args.greed

# Should the impact agent actually trade?
impact = args.impact

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
# Also suppresses formatting of limit orders (which is time consuming).
util.silent_mode = not args.verbose
LimitOrder.silent_mode = not args.verbose

# Config parameter that causes every order-related action to be logged by
# every agent.  Activate only when really needed as there is a significant
# time penalty to all that object serialization!
log_orders = args.log_orders


print ("Silent mode: {}".format(util.silent_mode))
print ("Logging orders: {}".format(log_orders))
print ("Book freq: {}".format(book_freq))
print ("ZeroIntelligenceAgent noise: {:0.4f}".format(sigma_n))
print ("ImpactAgent greed: {:0.2f}".format(greed))
print ("ImpactAgent firing: {}".format(impact))
print ("Shock variance: {:0.4f}".format(sigma_s))
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

# When should the Kernel shut down?  (This should be after market close.)
# Here we go for 5 PM the same day.
kernelStopTime = midnight + pd.to_timedelta('17:00:00')

# This will configure the kernel with a default computation delay
# (time penalty) for each agent's wakeup and recvMsg.  An agent
# can change this at any time for itself.  (nanoseconds)
defaultComputationDelay = 0        # no delay for this config


# IMPORTANT NOTE CONCERNING AGENT IDS: the id passed to each agent must:
#    1. be unique
#    2. equal its index in the agents list
# This is to avoid having to call an extra getAgentListIndexByID()
# in the kernel every single time an agent must be referenced.


# This is a list of symbols the exchange should trade.  It can handle any number.
# It keeps a separate order book for each symbol.  The example data includes
# only IBM.  This config uses generated data, so the symbol doesn't really matter.

# If shock variance must differ for each traded symbol, it can be overridden here.
symbols = { 'IBM' : { 'r_bar' : 100000, 'kappa' : 0.05, 'sigma_s' : sigma_s } }



### Configure the Kernel.
kernel = Kernel("Base Kernel", random_state = np.random.RandomState(seed=np.random.randint(low=0,high=2**32)))



### Configure the agents.  When conducting "agent of change" experiments, the
### new agents should be added at the END only.
agent_count = 0
agents = []
agent_types = []


### Configure an exchange agent.

# Let's open the exchange at 9:30 AM.
mkt_open = midnight + pd.to_timedelta('09:30:00')

# And close it at 9:30:00.000001 (i.e. 1,000 nanoseconds or "time steps")
mkt_close = midnight + pd.to_timedelta('09:30:00.000001')


# Configure an appropriate oracle for all traded stocks.
# All agents requiring the same type of Oracle will use the same oracle instance.
oracle = MeanRevertingOracle(mkt_open, mkt_close, symbols)


# Create the exchange.
num_exchanges = 1
agents.extend([ ExchangeAgent(j, "Exchange Agent {}".format(j), "ExchangeAgent", mkt_open, mkt_close, [s for s in symbols], log_orders=log_orders, book_freq=book_freq, pipeline_delay = 0, computation_delay = 0, stream_history = 10, random_state = np.random.RandomState(seed=np.random.randint(low=0,high=2**32)))
                for j in range(agent_count, agent_count + num_exchanges) ])
agent_types.extend(["ExchangeAgent" for j in range(num_exchanges)])
agent_count += num_exchanges



### Configure some zero intelligence agents.

# Cash in this simulator is always in CENTS.
starting_cash = 10000000

# Here are the zero intelligence agents.
symbol = 'IBM'
s = symbols[symbol]

# Tuples are: (# agents, R_min, R_max, eta, L).  L for HBL only.

# Some configs for ZI agents only (among seven parameter settings).

# 4 agents
#zi = [ (1, 0, 250, 1), (1, 0, 500, 1), (1, 0, 1000, 0.8), (1, 0, 1000, 1), (0, 0, 2000, 0.8), (0, 250, 500, 0.8), (0, 250, 500, 1) ]
#hbl = []

# 28 agents
#zi = [ (4, 0, 250, 1), (4, 0, 500, 1), (4, 0, 1000, 0.8), (4, 0, 1000, 1), (4, 0, 2000, 0.8), (4, 250, 500, 0.8), (4, 250, 500, 1) ]
#hbl = []

# 65 agents
#zi = [ (10, 0, 250, 1), (10, 0, 500, 1), (9, 0, 1000, 0.8), (9, 0, 1000, 1), (9, 0, 2000, 0.8), (9, 250, 500, 0.8), (9, 250, 500, 1) ]
#hbl = []

# 100 agents
#zi = [ (15, 0, 250, 1), (15, 0, 500, 1), (14, 0, 1000, 0.8), (14, 0, 1000, 1), (14, 0, 2000, 0.8), (14, 250, 500, 0.8), (14, 250, 500, 1) ]
#hbl = []

# 1000 agents
#zi = [ (143, 0, 250, 1), (143, 0, 500, 1), (143, 0, 1000, 0.8), (143, 0, 1000, 1), (143, 0, 2000, 0.8), (143, 250, 500, 0.8), (142, 250, 500, 1) ]
#hbl = []

# 10000 agents
#zi = [ (1429, 0, 250, 1), (1429, 0, 500, 1), (1429, 0, 1000, 0.8), (1429, 0, 1000, 1), (1428, 0, 2000, 0.8), (1428, 250, 500, 0.8), (1428, 250, 500, 1) ]
#hbl = []


# Some configs for HBL agents only (among four parameter settings).

# 4 agents
#zi = []
#hbl = [ (1, 250, 500, 1, 2), (1, 250, 500, 1, 3), (1, 250, 500, 1, 5), (1, 250, 500, 1, 8) ] 

# 28 agents
#zi = []
#hbl = [ (7, 250, 500, 1, 2), (7, 250, 500, 1, 3), (7, 250, 500, 1, 5), (7, 250, 500, 1, 8) ] 

# 1000 agents
#zi = []
#hbl = [ (250, 250, 500, 1, 2), (250, 250, 500, 1, 3), (250, 250, 500, 1, 5), (250, 250, 500, 1, 8) ]


# Some configs that mix both types of agents.

# 28 agents
#zi = [ (3, 0, 250, 1), (3, 0, 500, 1), (3, 0, 1000, 0.8), (3, 0, 1000, 1), (3, 0, 2000, 0.8), (3, 250, 500, 0.8), (2, 250, 500, 1) ]
#hbl = [ (2, 250, 500, 1, 2), (2, 250, 500, 1, 3), (2, 250, 500, 1, 5), (2, 250, 500, 1, 8) ]

# 65 agents
#zi = [ (7, 0, 250, 1), (7, 0, 500, 1), (7, 0, 1000, 0.8), (7, 0, 1000, 1), (7, 0, 2000, 0.8), (7, 250, 500, 0.8), (7, 250, 500, 1) ]
#hbl = [ (4, 250, 500, 1, 2), (4, 250, 500, 1, 3), (4, 250, 500, 1, 5), (4, 250, 500, 1, 8) ] 

# 1000 agents
zi = [ (100, 0, 250, 1), (100, 0, 500, 1), (100, 0, 1000, 0.8), (100, 0, 1000, 1), (100, 0, 2000, 0.8), (100, 250, 500, 0.8), (100, 250, 500, 1) ]
hbl = [ (75, 250, 500, 1, 2), (75, 250, 500, 1, 3), (75, 250, 500, 1, 5), (75, 250, 500, 1, 8) ] 



# ZI strategy split.
for i,x in enumerate(zi):
  strat_name = "Type {} [{} <= R <= {}, eta={}]".format(i+1, x[1], x[2], x[3])
  agents.extend([ ZeroIntelligenceAgent(j, "ZI Agent {} {}".format(j, strat_name), "ZeroIntelligenceAgent {}".format(strat_name), random_state = np.random.RandomState(seed=np.random.randint(low=0,high=2**32)),log_orders=log_orders, symbol=symbol, starting_cash=starting_cash, sigma_n=sigma_n, r_bar=s['r_bar'], kappa=s['kappa'], sigma_s=s['sigma_s'], q_max=10, sigma_pv=5000000, R_min=x[1], R_max=x[2], eta=x[3], lambda_a=0.005) for j in range(agent_count,agent_count+x[0]) ])
  agent_types.extend([ "ZeroIntelligenceAgent {}".format(strat_name) for j in range(x[0]) ])
  agent_count += x[0]

# HBL strategy split.
for i,x in enumerate(hbl):
  strat_name = "Type {} [{} <= R <= {}, eta={}, L={}]".format(i+1, x[1], x[2], x[3], x[4])
  agents.extend([ HeuristicBeliefLearningAgent(j, "HBL Agent {} {}".format(j, strat_name), "HeuristicBeliefLearningAgent {}".format(strat_name), random_state = np.random.RandomState(seed=np.random.randint(low=0,high=2**32)), log_orders=log_orders, symbol=symbol, starting_cash=starting_cash, sigma_n=sigma_n, r_bar=s['r_bar'], kappa=s['kappa'], sigma_s=s['sigma_s'], q_max=10, sigma_pv=5000000, R_min=x[1], R_max=x[2], eta=x[3], lambda_a=0.005, L=x[4]) for j in range(agent_count,agent_count+x[0]) ])
  agent_types.extend([ "HeuristicBeliefLearningAgent {}".format(strat_name) for j in range(x[0]) ])
  agent_count += x[0]



# Impact agent.

# 200 time steps in...
impact_time = midnight + pd.to_timedelta('09:30:00.0000002')

i = agent_count
agents.append(ImpactAgent(i, "Impact Agent {}".format(i), "ImpactAgent", symbol = "IBM", starting_cash = starting_cash, greed = greed, impact = impact, impact_time = impact_time, random_state = np.random.RandomState(seed=np.random.randint(low=0,high=2**32))))
agent_types.append("Impact Agent {}".format(i))
agent_count += 1


### Configure a simple message latency matrix for the agents.  Each entry is the minimum
# nanosecond delay on communication [from][to] agent ID.

# Square numpy array with dimensions equal to total agent count.  In this config,
# there should not be any communication delay.
latency = np.zeros((len(agent_types),len(agent_types)))

# Configure a simple latency noise model for the agents.
# Index is ns extra delay, value is probability of this delay being applied.
# In this config, there is no latency (noisy or otherwise).
noise = [ 1.0 ]



# Start the kernel running.
kernel.runner(agents = agents, startTime = kernelStartTime,
              stopTime = kernelStopTime, agentLatency = latency,
              latencyNoise = noise,
              defaultComputationDelay = defaultComputationDelay,
              oracle = oracle, log_dir = log_dir)

