from Kernel import Kernel
from agent.ExchangeAgent import ExchangeAgent
from agent.BackgroundAgent import BackgroundAgent
from agent.MomentumAgent import MomentumAgent
from util.order import LimitOrder
from util.oracle.DataOracle import DataOracle
from util import util

import datetime as dt
import numpy as np
import pandas as pd
import sys


# Some config files require additional command line parameters to easily
# control agent or simulation hyperparameters during coarse parallelization.
import argparse

parser = argparse.ArgumentParser(description='Detailed options for momentum config.')
parser.add_argument('-b', '--bg_noise', type=float, default=0.01,
                    help='Observation noise std for background agents')
parser.add_argument('-c', '--config', required=True,
                    help='Name of config file to execute')
parser.add_argument('-d', '--date', default='2014-01-28',
                    help='Historical date to simulate')
parser.add_argument('-f', '--frequency', default='5m',
                    help='Base Timestamp frequency for BackgroundTrader actions')
parser.add_argument('-l', '--log_dir', default=None,
                    help='Log directory name (default: unix timestamp at program start)')
parser.add_argument('-o', '--offset_unit', default='s',
                    help='Wakeup offset (jitter) unit for BackgroundTrader actions')
parser.add_argument('-s', '--seed', type=int, default=None,
                    help='numpy.random.seed() for simulation')
parser.add_argument('-t', '--arb_last_trade', action='store_true',
                    help='Arbitrage last trade instead of spread midpoint')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='Maximum verbosity!')
parser.add_argument('--config_help', action='store_true',
                    help='Print argument options for this config file')

args, remaining_args = parser.parse_known_args()

if args.config_help:
  parser.print_help()
  sys.exit()

# Historical date to simulate.  Log file directory if not default.
historical_date = pd.to_datetime(args.date)
log_dir = args.log_dir

# Observation noise for background agents.  BG agent wake frequency + noise.
noise_std = args.bg_noise
freq = args.frequency
offset_unit = args.offset_unit

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

# Config parameter to arbitrage last trade vs spread (for BackgroundTrader).
arb_last_trade = args.arb_last_trade


print ("Silent mode: {}".format(util.silent_mode))
print ("BackgroundAgent freq: {}".format(freq))
print ("BackgroundAgent noise: {:0.4f}".format(noise_std))
print ("BackgroundAgent arbs last trade: {}".format(arb_last_trade))
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
defaultComputationDelay = 1000000    # one millisecond


# IMPORTANT NOTE CONCERNING AGENT IDS: the id passed to each agent must:
#    1. be unique
#    2. equal its index in the agents list
# This is to avoid having to call an extra getAgentListIndexByID()
# in the kernel every single time an agent must be referenced.


### Configure some background agents.
num_agents = 100

# Cash in this simulator is always in CENTS.
starting_cash = 1000000 * 100           # a million dollars

# Set a mean order volume around which BG agents should select somewhat random sizes.
# Eventually we might like to pull in historical volume from the oracle.
bg_trade_vol = 300

# Here are those background agents.
agents = [ BackgroundAgent(i, "Background Agent {}".format(i), "IBM", starting_cash, noise_std, arb_last_trade, freq, bg_trade_vol, offset_unit) for i in range(0,num_agents) ]
agent_types = ["BackgroundAgent" for i in range(num_agents)]


### Configure some momentum agents.
num_momentum_agents = 10
lookback = 5
agents.extend([ MomentumAgent(i, "Momentum Agent {}".format(i), "IBM", starting_cash, lookback)
                for i in range(num_agents, num_agents + num_momentum_agents) ])
num_agents += num_momentum_agents
agent_types.extend(["MomentumAgent" for i in range(num_momentum_agents)])


### Configure an exchange agent.

# Let's open the exchange at 9:30 AM.
mkt_open = midnight + pd.to_timedelta('09:30:00')

# And close it at 4:00 PM.
mkt_close = midnight + pd.to_timedelta('16:00:00')

# This is a list of symbols the exchange should trade.  It can handle any number.
# It keeps a separate order book for each symbol.  The example data includes
# only IBM.
symbols = ['IBM']

num_exchanges = 1
agents.extend([ ExchangeAgent(i, "Exchange Agent {}".format(i), mkt_open, mkt_close, symbols, book_freq='S')
                for i in range(num_agents, num_agents + num_exchanges) ])
agent_types.extend(["ExchangeAgent" for i in range(num_exchanges)])



### Configure a simple message latency matrix for the agents.  Each entry is the minimum
# nanosecond delay on communication [from][to] agent ID.

# Square numpy array with dimensions equal to total agent count.  Background Agents,
# by far the largest population, are handled at init, drawn from a uniform distribution from:
# Times Square (3.9 miles from NYSE, approx. 21 microseconds at the speed of light) to:
# Pike Place Starbucks in Seattle, WA (2402 miles, approx. 13 ms at the speed of light).
# Other agents are set afterward (and the mirror half of the matrix is also).

latency = np.random.uniform(low = 21000, high = 13000000, size=(len(agent_types),len(agent_types)))

for i, t1 in zip(range(latency.shape[0]), agent_types):
  for j, t2 in zip(range(latency.shape[1]), agent_types):
    # Three cases for symmetric array.  Set latency when j > i, copy it when i > j, same agent when i == j.
    if j > i:
      # A hypothetical order book exploiting agent might require a special very low latency to represent
      # colocation with the exchange hardware.
      if (t1 == "ExploitAgent" and t2 == "ExchangeAgent") or (t2 == "ExploitAgent" and t1 == "ExchangeAgent"):
        # We don't have any exploiting agents in this configuration, so any arbitrary number is fine.
        # Let's use about 1/3 usec or approx 100m of fiber-optic cable.
        latency[i,j] = 333
    elif i > j:
      # This "bottom" half of the matrix simply mirrors the top.
      latency[i,j] = latency[j,i]
    else:
      # This is the same agent.  How long does it take to reach localhost?  In our data center, it actually
      # takes about 20 microseconds.
      latency[i,j] = 20000


# Configure a simple latency noise model for the agents.
# Index is ns extra delay, value is probability of this delay being applied.
# We may later want to substitute some realistic noise model or sample from a geographic database.
noise = [ 0.4, 0.25, 0.15, 0.1, 0.05, 0.025, 0.025 ]


# Create the data oracle for this experiment.  All agents will use the same one.
oracle = DataOracle(historical_date.date(), symbols)


# Start a basic kernel.
kernel = Kernel("Base Kernel")
kernel.runner(agents = agents, startTime = kernelStartTime,
              stopTime = kernelStopTime, agentLatency = latency,
              latencyNoise = noise,
              defaultComputationDelay = defaultComputationDelay,
              oracle = oracle, log_dir = log_dir)

