from Kernel import Kernel
from agent.ExchangeAgent import ExchangeAgent
from agent.HeuristicBeliefLearningAgent import HeuristicBeliefLearningAgent
from agent.ZeroIntelligenceAgent import ZeroIntelligenceAgent
from util.order import LimitOrder
from util.oracle.MeanRevertingOracle import MeanRevertingOracle
from util import util

import datetime as dt
import numpy as np
import pandas as pd
import sys


# Some config files require additional command line parameters to easily
# control agent or simulation hyperparameters during coarse parallelization.
import argparse

parser = argparse.ArgumentParser(description='Detailed options for momentum config.')
parser.add_argument('-b', '--book_freq', default='10N',
                    help='Frequency at which to archive order book for visualization')
parser.add_argument('-c', '--config', required=True,
                    help='Name of config file to execute')
parser.add_argument('-l', '--log_dir', default=None,
                    help='Log directory name (default: unix timestamp at program start)')
parser.add_argument('-n', '--obs_noise', type=float, default=1000,
                    help='Observation noise variance for zero intelligence agents (sigma^2_n)')
parser.add_argument('-r', '--shock_variance', type=float, default=100000,
                    help='Shock variance for mean reversion process (sigma^2_s)')
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

# Historical date to simulate.  Log file directory if not default.
# Not relevant for SRG config, but one is required.
historical_date = pd.to_datetime('2014-01-28')
log_dir = args.log_dir
book_freq = args.book_freq if args.book_freq.lower() != 'none' else None

# Observation noise variance for zero intelligence agents.
sigma_n = args.obs_noise

# Shock variance of mean reversion process.
sigma_s = args.shock_variance

# Random seed specification on the command line.  Default: None (by clock).
# If none, we select one via a specific random method and pass it to seed()
# so we can record it for future use.  (You cannot reasonably obtain the
# automatically generated seed when seed() is called without a parameter.)
seed = args.seed
if not seed: seed = int(pd.Timestamp.now().timestamp() * 1000000) % (2**32 - 1)
np.random.seed(seed)

# Config parameter that causes util.util.print to suppress most output.
# Also suppresses formatting of limit orders (which is time consuming).
util.silent_mode = not args.verbose
LimitOrder.silent_mode = not args.verbose


print ("Silent mode: {}".format(util.silent_mode))
print ("Book freq: {}".format(book_freq))
print ("ZeroIntelligenceAgent noise: {:0.4f}".format(sigma_n))
print ("Shock variance: {:0.4f}".format(sigma_s))
print ("Configuration seed: {}\n".format(seed))



### Required parameters for all simulations.

# Since the simulator often pulls historical data, we use a real-world
# nanosecond timestamp (pandas.Timestamp) for our discrete time "steps",
# which are considered to be nanoseconds.  For other (or abstract) time
# units, one can either configure the Timestamp interval, or simply
# interpret the nanoseconds as something else.

# What is the earliest available time for an agent to act during the
# simulation?

# Timestamp will default to midnight, as desired.
midnight = historical_date
kernelStartTime = midnight

# When should the Kernel shut down?  (This should be after market close.)
# Here we go for 5 PM the same day.
kernelStopTime = midnight + pd.to_timedelta('17:00:00')

# This will configure the kernel with a default computation delay
# (time penalty) for each agent's wakeup and recvMsg.  An agent
# can change this at any time for itself.  (nanoseconds)
defaultComputationDelay = 0        # no delay for SRG config


# IMPORTANT NOTE CONCERNING AGENT IDS: the id passed to each agent must:
#    1. be unique
#    2. equal its index in the agents list
# This is to avoid having to call an extra getAgentListIndexByID()
# in the kernel every single time an agent must be referenced.


# This is a list of symbols the exchange should trade.  It can handle any number.
# It keeps a separate order book for each symbol.  The example data includes
# only IBM.

# If shock variance must differ for each traded symbol, it can be overridden here.
symbols = { 'IBM' : { 'r_bar' : 100000, 'kappa' : 0.05, 'sigma_s' : sigma_s } }


### Configure some zero intelligence agents.
num_agents = 100

# Cash in this simulator is always in CENTS.
starting_cash = 10000000

agent_count = 0

# Here are the zero intelligence agents.
symbol = 'IBM'
s = symbols[symbol]
agents = [ ZeroIntelligenceAgent(i, "ZI Agent {}".format(i), symbol, starting_cash, sigma_n=sigma_n, r_bar=s['r_bar'], kappa=s['kappa'], sigma_s=s['sigma_s'], q_max=10, sigma_pv=5000000, R_min=250, R_max=500, eta=0.8, lambda_a=0.005) for i in range(0,num_agents) ]
agent_types = ["ZeroIntelligenceAgent" for i in range(num_agents)]
agent_count = num_agents


# Here are the heuristic belief learning agents.
num_hbl_agents = 10
agents.extend([ HeuristicBeliefLearningAgent(i, "HBL Agent {}".format(i), symbol, starting_cash, sigma_n=sigma_n, r_bar=s['r_bar'], kappa=s['kappa'], sigma_s=s['sigma_s'], q_max=10, sigma_pv=5000000, R_min=250, R_max=500, eta=0.8, lambda_a=0.005, L=8) for i in range(agent_count, agent_count + num_hbl_agents) ])
agent_types.extend(["HeuristicBeliefLearningAgent" for i in range(num_hbl_agents)])
agent_count += num_hbl_agents



### Configure an exchange agent.

# Let's open the exchange at 9:30 AM.
mkt_open = midnight + pd.to_timedelta('09:30:00')

# And close it at 9:30:00.00001 (i.e. 10,000 nanoseconds or "time steps")
mkt_close = midnight + pd.to_timedelta('09:30:00.00001')


num_exchanges = 1
agents.extend([ ExchangeAgent(i, "Exchange Agent {}".format(i), mkt_open, mkt_close, [s for s in symbols], book_freq=book_freq, pipeline_delay = 0, computation_delay = 0, stream_history = 10)
                for i in range(agent_count, agent_count + num_exchanges) ])
agent_types.extend(["ExchangeAgent" for i in range(num_exchanges)])
agent_count += num_exchanges



### Configure a simple message latency matrix for the agents.  Each entry is the minimum
# nanosecond delay on communication [from][to] agent ID.

# Square numpy array with dimensions equal to total agent count.  In the SRG config,
# there should not be any communication delay.
latency = np.zeros((len(agent_types),len(agent_types)))

# Configure a simple latency noise model for the agents.
# Index is ns extra delay, value is probability of this delay being applied.
# In the SRG config, there is no latency (noisy or otherwise).
noise = [ 1.0 ]


# Create the data oracle for this experiment.  All agents will use the same one.
oracle = MeanRevertingOracle(mkt_open, mkt_close, symbols)


# Start a basic kernel.
kernel = Kernel("Base Kernel")
kernel.runner(agents = agents, startTime = kernelStartTime,
              stopTime = kernelStopTime, agentLatency = latency,
              latencyNoise = noise,
              defaultComputationDelay = defaultComputationDelay,
              oracle = oracle, log_dir = log_dir)

