from Kernel import Kernel
from agent.ExchangeAgent import ExchangeAgent
from agent.NoiseAgent import NoiseAgent
from agent.ValueAgent import ValueAgent
from agent.market_makers.MarketMakerAgent import MarketMakerAgent
from util.order import LimitOrder
from util.oracle.SparseMeanRevertingOracle import SparseMeanRevertingOracle
from util import util

import numpy as np
import pandas as pd
import sys

# Some config files require additional command line parameters to easily
# control agent or simulation hyperparameters during coarse parallelization.
import argparse

parser = argparse.ArgumentParser(description='Detailed options for sparse_zi config.')
parser.add_argument('-b', '--book_freq', default=None,
                    help='Frequency at which to archive order book for visualization')
parser.add_argument('-c', '--config', required=True,
                    help='Name of config file to execute')
parser.add_argument('-l', '--log_dir', default=None,
                    help='Log directory name (default: unix timestamp at program start)')
parser.add_argument('-n', '--obs_noise', type=float, default=1000000,
                    help='Observation noise variance for zero intelligence agents (sigma^2_n)')
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
historical_date = pd.to_datetime('2019-06-28')

# Requested log directory.
log_dir = args.log_dir

# Requested order book snapshot archive frequency.
book_freq = args.book_freq

# Observation noise variance for zero intelligence agents.
# This is a property of the agents, not the stock.
# Later, it could be a matrix across both.
sigma_n = args.obs_noise

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
if not seed: seed = int(pd.Timestamp.now().timestamp() * 1000000) % (2 ** 32 - 1)
np.random.seed(seed)

# Config parameter that causes util.util.print to suppress most output.
# Also suppresses formatting of limit orders (which is time consuming).
util.silent_mode = not args.verbose
LimitOrder.silent_mode = not args.verbose

# Config parameter that causes every order-related action to be logged by
# every agent.  Activate only when really needed as there is a significant
# time penalty to all that object serialization!
log_orders = args.log_orders

print("Silent mode: {}".format(util.silent_mode))
print("Logging orders: {}".format(log_orders))
print("Book freq: {}".format(book_freq))
print("ZeroIntelligenceAgent noise: {:0.4f}".format(sigma_n))
print("Configuration seed: {}\n".format(seed))

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
defaultComputationDelay = 1000000000  # one second

# IMPORTANT NOTE CONCERNING AGENT IDS: the id passed to each agent must:
#    1. be unique
#    2. equal its index in the agents list
# This is to avoid having to call an extra getAgentListIndexByID()
# in the kernel every single time an agent must be referenced.


# This is a list of symbols the exchange should trade.  It can handle any number.
# It keeps a separate order book for each symbol.  The example data includes
# only JPM.  This config uses generated data, so the symbol doesn't really matter.

# megashock_lambda_a is used to select spacing for megashocks (using an exponential
# distribution equivalent to a centralized Poisson process).  Megashock mean
# and variance control the size (bimodal distribution) of the individual shocks.

# Note: sigma_s is no longer used by the agents or the fundamental (for sparse discrete simulation).

symbols = {'JPM': {'r_bar': 1e5, 'kappa': 1.67e-12, 'agent_kappa': 1.67e-15, 'sigma_s': 0, 'fund_vol': 1e-8,
                   'megashock_lambda_a': 2.77778e-13, 'megashock_mean': 1e3, 'megashock_var': 5e4,
                   'random_state': np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64'))}}

### Configure the Kernel.
kernel = Kernel("Base Kernel",
                random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')))

### Configure the agents.  When conducting "agent of change" experiments, the
### new agents should be added at the END only.
agent_count = 0
agents = []
agent_types = []

### Configure an exchange agent.

# Let's open the exchange at 9:30 AM.
mkt_open = midnight + pd.to_timedelta('09:30:00')

# And close it at 4:00 PM.
mkt_close = midnight + pd.to_timedelta('10:30:00')

# Configure an appropriate oracle for all traded stocks.
# All agents requiring the same type of Oracle will use the same oracle instance.
oracle = SparseMeanRevertingOracle(mkt_open, mkt_close, symbols)

# Create the exchange.
num_exchanges = 1
agents.extend([ExchangeAgent(j, "Exchange Agent {}".format(j), "ExchangeAgent", mkt_open, mkt_close,
                             [s for s in symbols], log_orders=log_orders, book_freq=book_freq, pipeline_delay=0,
                             computation_delay=0, stream_history=10, random_state=np.random.RandomState(
        seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')))
               for j in range(agent_count, agent_count + num_exchanges)])
agent_types.extend(["ExchangeAgent" for j in range(num_exchanges)])
agent_count += num_exchanges

### Configure some zero intelligence agents.

# Cash in this simulator is always in CENTS.
starting_cash = 10000000

# Here are the zero intelligence agents.
symbol = 'JPM'
s = symbols[symbol]

# Tuples are: (# agents, R_min, R_max, eta).

# Some configs for ZI agents only (among seven parameter settings).

#number of noise agents
num_noise = 100

# ZI strategy split.  Note that agent arrival rates are quite small, because our minimum
# time step is a nanosecond, and we want the agents to arrive more on the order of
# minutes.

agents.extend( [NoiseAgent(j, "NoiseAgent {}".format(j),
                                         "NoiseAgent",
                                         random_state=np.random.RandomState(
                                             seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')),
                                         log_orders=log_orders, symbol=symbol, starting_cash=starting_cash,
                                         wakeup_time = mkt_open + np.random.rand() * (mkt_close - mkt_open) ) for j in range(agent_count, agent_count + num_noise )])
agent_count += num_noise
agent_types.extend(['NoiseAgent' for j in range(agent_count, agent_count + num_noise) ])

# 100 agents
num_value = 50

# ZI strategy split.  Note that agent arrival rates are quite small, because our minimum
# time step is a nanosecond, and we want the agents to arrive more on the order of
# minutes.
agents.extend([ValueAgent(j, "Value Agent {}".format(j),
                                         "ValueAgent {}".format(j),
                                         random_state=np.random.RandomState(
                                             seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')),
                                         log_orders=log_orders, symbol=symbol, #starting_cash=starting_cash,
                                         sigma_n=sigma_n, r_bar=s['r_bar'], kappa=s['agent_kappa'],
                                         sigma_s=s['fund_vol'],
                                         lambda_a=1e-12) for j in range(agent_count, agent_count + num_value )])
agent_types.extend(["ValueAgent {}".format(j) for j in range(num_value)])
agent_count += num_value

### Configure a simple message latency matrix for the agents.  Each entry is the minimum
### nanosecond delay on communication [from][to] agent ID.

# Square numpy array with dimensions equal to total agent count.  Most agents are handled
# at init, drawn from a uniform distribution from:
# Times Square (3.9 miles from NYSE, approx. 21 microseconds at the speed of light) to:
# Pike Place Starbucks in Seattle, WA (2402 miles, approx. 13 ms at the speed of light).
# Other agents can be explicitly set afterward (and the mirror half of the matrix is also).

# This configures all agents to a starting latency as described above.
latency = np.random.uniform(low=21000, high=13000000, size=(len(agent_types), len(agent_types)))

# Overriding the latency for certain agent pairs happens below, as does forcing mirroring
# of the matrix to be symmetric.
for i, t1 in zip(range(latency.shape[0]), agent_types):
    for j, t2 in zip(range(latency.shape[1]), agent_types):
        # Three cases for symmetric array.  Set latency when j > i, copy it when i > j, same agent when i == j.
        if j > i:
            # Presently, strategy agents shouldn't be talking to each other, so we set them to extremely high latency.
            if (t1 == "ZeroIntelligenceAgent" and t2 == "ZeroIntelligenceAgent"):
                latency[i, j] = 1000000000 * 60 * 60 * 24  # Twenty-four hours.
        elif i > j:
            # This "bottom" half of the matrix simply mirrors the top.
            latency[i, j] = latency[j, i]
        else:
            # This is the same agent.  How long does it take to reach localhost?  In our data center, it actually
            # takes about 20 microseconds.
            latency[i, j] = 20000

# Configure a simple latency noise model for the agents.
# Index is ns extra delay, value is probability of this delay being applied.
noise = [0.25, 0.25, 0.20, 0.15, 0.10, 0.05]

# Start the kernel running.
kernel.runner(agents=agents, startTime=kernelStartTime,
              stopTime=kernelStopTime, agentLatency=latency,
              latencyNoise=noise,
              defaultComputationDelay=defaultComputationDelay,
              oracle=oracle, log_dir=log_dir)

