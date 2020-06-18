from Kernel import Kernel
from agent.ExchangeAgent import ExchangeAgent
from agent.OrderBookImbalanceAgent import OrderBookImbalanceAgent
from agent.ValueAgent import ValueAgent
from agent.ZeroIntelligenceAgent import ZeroIntelligenceAgent
from agent.examples.MarketMakerAgent import MarketMakerAgent
from agent.examples.MomentumAgent import MomentumAgent
from statistics import median, mean, stdev
from util.order import LimitOrder
from util.oracle.SparseMeanRevertingOracle import SparseMeanRevertingOracle
from util import util
from util.model.QTable import QTable

import numpy as np
import pandas as pd
import sys

from math import ceil, floor

###### Helper functions for this configuration file.  Just commonly-used code ######
###### that would otherwise have been repeated many times.                    ######

def get_rand_obj(seed_obj):
  return np.random.RandomState(seed = seed_obj.randint(low = 0, high = 2**32))


###### Wallclock tracking for overall experimental scheduling to CPUs.
wallclock_start = pd.Timestamp('now')

print ("\n====== Experimental wallclock elapsed: {} ======\n".format(
                                pd.Timestamp('now') - wallclock_start))


###### One-time configuration section.  This section sets up definitions that ######
###### will apply to the entire experiment.  They will not be repeated or     ######
###### reinitialized for each instance of the simulation contained within     ######
###### this experiment.                                                       ######


### EXPERIMENT CONFIGURATION.
obi_perf = []


### DATE CONFIGURATION.

# Since the simulator often pulls historical data, we use nanosecond
# timestamps (pandas.Timestamp) for all simulation time tracking.
# Thus our discrete time stamps are effectively nanoseconds, although
# they can be interepreted otherwise for ahistorical (e.g. generated)
# simulations.  These timestamps do require a valid date component.
midnight = pd.to_datetime('2014-01-28')


### STOCK SYMBOL CONFIGURATION.
symbols = { 'IBM' : { 'r_bar' : 1e5, 'kappa' : 1.67e-12, 'agent_kappa' : 1.67e-15,
                      'sigma_s' : 0, 'fund_vol' : 1e-8, 'megashock_lambda_a' : 2.77778e-13,
                      'megashock_mean' : 1e3, 'megashock_var' : 5e4 }
          }


### INITIAL AGENT DISTRIBUTION.
### You must instantiate the agents in the same order you record them
### in the agent_types and agent_strats lists.  (Currently they are
### parallel arrays.)
###
### When conducting "agent of change" experiments, the new agents should
### be added at the END only.

agent_types = []
agent_strats = []

### Count by agent type.
num_exch = 1
num_zi = 50
num_val = 50
num_obi = 1
num_mm = 0
num_mom = 0


### EXCHANGE AGENTS
mkt_open = midnight + pd.to_timedelta('09:30:00')
mkt_close = midnight + pd.to_timedelta('16:00:00')

### Record the type and strategy of the agents for reporting purposes.
for i in range(num_exch):
  agent_types.append("ExchangeAgent")
  agent_strats.append("ExchangeAgent")


### ZERO INTELLIGENCE AGENTS

### ZeroIntelligence fixed parameters (i.e. not strategic).
zi_obs_noise = 1000000    # a property of the agent, not an individual stock

### Lay out the ZI strategies (parameter settings) that will be used in this
### experiment, so we can assign particular numbers of agents to each strategy.
### Tuples are: (R_min, R_max, eta).

zi_strategy = [ (0, 250, 1), (0, 500, 1), (0, 1000, 0.8), (0, 1000, 1),
                (0, 2000, 0.8), (250, 500, 0.8), (250, 500, 1) ]

### Record the initial distribution of agents to ZI strategies.
### Split the agents as evenly as possible among the strategy settings.
zi = [ floor(num_zi / len(zi_strategy)) ] * len(zi_strategy)

i = 0
while sum(zi) < num_zi:
  zi[i] += 1
  i += 1

### Record the type and strategy of the agents for reporting purposes.
for i in range(len(zi_strategy)):
  x = zi_strategy[i]
  strat_name = "Type {} [{} <= R <= {}, eta={}]".format(i+1, x[0], x[1], x[2])
  agent_types.extend([ 'ZeroIntelligenceAgent' ] * zi[i])
  agent_strats.extend([ 'ZeroIntelligenceAgent ({})'.format(strat_name) ] * zi[i])


### VALUE AGENTS

### Value agent fixed parameters (i.e. not strategic).
zi_obs_noise = 1000000    # a property of the agent, not an individual stock

for i in range(num_val):
  agent_types.extend([ 'ValueAgent' ])
  agent_strats.extend([ 'ValueAgent' ])


### OBI AGENTS

### OBI fixed parameters (i.e. not strategic).

### Record the type and strategy of the agents for reporting purposes.
for i in range(num_obi):
  agent_types.append("OBIAgent")
  agent_strats.append("OBIAgent")


### MARKET MAKER AGENTS

### Market Maker fixed parameters (i.e. not strategic).

### Record the type and strategy of the agents for reporting purposes.
for i in range(num_mm):
  agent_types.append("MarketMakerAgent")
  agent_strats.append("MarketMakerAgent")


### MOMENTUM AGENTS

### Momentum Agent fixed parameters.

### Record the type and strategy of the agents for reporting purposes.
for i in range(num_mom):
  agent_types.append("MomentumAgent")
  agent_strats.append("MomentumAgent")


### FINAL AGENT PREPARATION

### Record the total number of agents here, so we can create a list of lists
### of random seeds to use for the agents across the iterated simulations.
### Also create an empty list of appropriate size to store agent state
### across simulations (for those agents which require it).

num_agents = num_exch + num_zi + num_obi + num_val + num_mm + num_mom


### SIMULATION CONTROL SETTINGS.

# We allow some high-level parameters to be specified on the command line at
# runtime, rather than being explicitly coded in the config file.  This really
# only makes sense for parameters that affect the entire series of simulations
# (i.e. the entire "experiment"), rather than a single instance of the simulation.
import argparse

parser = argparse.ArgumentParser(description='Detailed options for sparse_zi config.')
parser.add_argument('-b', '--book_freq', default=None,
                    help='Frequency at which to archive order book for visualization')
parser.add_argument('-c', '--config', required=True,
                    help='Name of config file to execute')
parser.add_argument('-l', '--log_dir', default=None,
                    help='Log directory name (default: unix timestamp at program start)')
parser.add_argument('-o', '--log_orders', action='store_true',
                    help='Log every order-related action by every agent.')
parser.add_argument('-s', '--seed', type=int, default=None,
                    help='numpy.random.seed() for simulation')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='Maximum verbosity!')
parser.add_argument('--config_help', action='store_true',
                    help='Print argument options for this config file')

# Specialized for loop_obi.
parser.add_argument('-f', '--obi_freq', type=int, default=1e9 * 60,
                    help='OBI subscription frequency')
parser.add_argument('-r', '--flat_range', type=float, default=0.1,
                    help='OBI dead zone for staying flat')
parser.add_argument('-y', '--entry_threshold', type=float, default=0.1,
                    help='OBI imbalance to enter a position')
parser.add_argument('-t', '--trail_dist', type=float, default=0.05,
                    help='OBI trailing stop distance to exit positions')
parser.add_argument('-e', '--levels', type=int, default=10,
                    help='OBI order book levels to consider')
parser.add_argument('-n', '--num_simulations', type=int, default=5,
                    help='Number of consecutive simulations in one episode.')


args, remaining_args = parser.parse_known_args()

if args.config_help:
  parser.print_help()
  sys.exit()


# If nothing specifically requested, use starting timestamp.  In either case, successive
# simulations will have simulation number appended.
log_dir = args.log_dir

if log_dir is None: log_dir = str(int(pd.Timestamp('now').timestamp()))


# Requested order book snapshot archive frequency.
book_freq = args.book_freq

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

# Specialized for loop_obi.
obi_freq = args.obi_freq
flat_range = args.flat_range
entry_threshold = args.entry_threshold
trail_dist = args.trail_dist
levels = args.levels
num_consecutive_simulations = args.num_simulations


print ("Silent mode: {}".format(util.silent_mode))
print ("Logging orders: {}".format(log_orders))
print ("Book freq: {}".format(book_freq))
print ("OBI Freq: {}, OB Levels: {}, Entry Thresh: {}, Trail Dist: {}".format(
       obi_freq, levels, entry_threshold, trail_dist))
print ("Configuration seed: {}\n".format(seed))



### STOCHASTIC CONTROL

### For every entity that requires a source of randomness, create (from the global seed)
### a RandomState object, which can be used to generate SEEDS for that entity at the
### start of each simulation.  This will premit each entity to receive a different
### seed for each simulation, but the entire experiment will still be deterministic
### given the same initial (global) seed.

kernel_seeds = np.random.RandomState(seed=np.random.randint(low=0,high=2**32))

symbol_seeds = {}
for sym in symbols:  symbol_seeds[sym] = np.random.RandomState(seed=np.random.randint(low=0,high=2**32))

agent_seeds = [ np.random.RandomState(seed=np.random.randint(low=0,high=2**32)) ] * num_agents



### LATENCY CONFIGURATION

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
    if j > i:
      # Presently, strategy agents shouldn't be talking to each other, so we set them to extremely high latency.
      if (t1 == "ZeroIntelligenceAgent" and t2 == "ZeroIntelligenceAgent"):
        latency[i,j] = 1000000000 * 60 * 60 * 24    # Twenty-four hours.
    elif i > j:
      # This "bottom" half of the matrix simply mirrors the top.
      latency[i,j] = latency[j,i]
    else:
      # This is the same agent.  How long does it take to reach localhost?  In our data center, it actually
      # takes about 20 microseconds.
      latency[i,j] = 20000

# OBI to Exchange and back.
latency[0,101] = 1
latency[101,0] = 1

# Configure a simple latency noise model for the agents.
# Index is ns extra delay, value is probability of this delay being applied.
noise = [ 0.25, 0.25, 0.20, 0.15, 0.10, 0.05 ]



### FINAL GLOBAL CONFIGURATION FOR ALL SIMULATIONS

# The kernel's start and stop times must be pandas.Timestamp
# objects, including a date.  (For ahistorical simulations, the date
# selected is irrelevant.)  This range represents the maximum extents
# of simulated time, and should generally be a superset of "market hours".
# There is no requirement these times be on the same date, although
# none of the current agents handle markets closing and reopening.
kernelStartTime = midnight
kernelStopTime = midnight + pd.to_timedelta('17:00:00')

# This will configure the kernel with a default computation delay
# (time penalty) for each agent's wakeup and recvMsg.  An agent
# can change this at any time for itself.  (nanoseconds)
defaultComputationDelay = 1000000000        # one second



###### Per-simulation configuration section.  This section initializes        ######
###### from scratch those objects and settings that should be reset withr     ######
###### each "run" of the simulation within an overall experiment.             ######

for sim in range(num_consecutive_simulations):   # eventually make this a stopping criteria

  # Flush the agent population and start over for each simulation.
  agents = []

  # The random state of each symbol needs to be set for each simulation, so the
  # stocks won't always do the same thing.  Note that the entire experiment
  # should still be fully repeatable with the same initial seed, because the
  # list of random seeds for a symbol is fixed at the start, based on the initial
  # seed.
  for symbol in symbols: symbols[symbol]['random_state'] = get_rand_obj(symbol_seeds[symbol])

  # Obtain a fresh simulation Kernel with the next appropriate random_state, seeded
  # from the list obtained before the first simulation.
  kernel = Kernel("Base Kernel", random_state = get_rand_obj(kernel_seeds))

  # Configure an appropriate oracle for all traded stocks.
  # All agents requiring the same type of Oracle will use the same oracle instance.
  # The oracle does not require its own source of randomness, because each symbol
  # and agent has those, and the oracle will always use on of those sources, as appropriate.
  oracle = SparseMeanRevertingOracle(mkt_open, mkt_close, symbols)


  # Create the agents in the same order they were specified in the first configuration
  # section (outside the simulation loop).  It is very important they be in the same
  # order.

  agent_id = 0

  # Create the exchange.
  for i in range(num_exch):
    agents.append( ExchangeAgent(agent_id, "{} {}".format(agent_types[agent_id], agent_id),
                                 agent_strats[agent_id], mkt_open, mkt_close,
                                 [s for s in symbols], log_orders = log_orders,
                                 book_freq = book_freq, pipeline_delay = 0,
                                 computation_delay = 0, stream_history = 10,
                                 random_state = get_rand_obj(agent_seeds[agent_id])) )
    agent_id += 1


  # Configure some zero intelligence agents.
  starting_cash = 10000000       # Cash in this simulator is always in CENTS.
  symbol = 'IBM'
  s = symbols[symbol]

  # ZI strategy split.  Note that agent arrival rates are quite small, because our minimum
  # time step is a nanosecond, and we want the agents to arrive more on the order of
  # minutes.
  for n, x in zip(zi, zi_strategy):
    strat_name = agent_strats[agent_id]
    while n > 0:
      agents.append(ZeroIntelligenceAgent(agent_id, "ZI Agent {}".format(agent_id), strat_name, random_state = get_rand_obj(agent_seeds[agent_id]), log_orders=log_orders, symbol=symbol, starting_cash=starting_cash, sigma_n=zi_obs_noise, r_bar=s['r_bar'], kappa=s['agent_kappa'], sigma_s=s['fund_vol'], q_max=10, sigma_pv=5e6, R_min=x[0], R_max=x[1], eta=x[2], lambda_a=1e-12))
      agent_id += 1
      n -= 1

  # Add value agents.
  for i in range(num_val):
    agents.extend([ ValueAgent(agent_id, "Value Agent {}".format(agent_id), "ValueAgent", symbol = symbol, random_state = get_rand_obj(agent_seeds[agent_id]), log_orders=log_orders, starting_cash=starting_cash, sigma_n=zi_obs_noise, r_bar=s['r_bar'], kappa=s['agent_kappa'], sigma_s=s['fund_vol'], lambda_a=1e-12) ])
    agent_id += 1


  # Add an OBI agent to try to beat this market.
  for i in range(num_obi):
    random_state = get_rand_obj(agent_seeds[agent_id])
    agents.extend([ OrderBookImbalanceAgent(agent_id, "OBI Agent {}".format(agent_id), "OrderBookImbalanceAgent", symbol = symbol, starting_cash = starting_cash, levels = levels, entry_threshold = entry_threshold, trail_dist = trail_dist, freq = obi_freq, random_state = random_state) ])
    agent_id += 1

  # Add market maker agents.
  for i in range(num_mm):
    random_state = get_rand_obj(agent_seeds[agent_id])
    agents.extend([ MarketMakerAgent(agent_id, "Market Maker Agent {}".format(agent_id), "MarketMakerAgent", symbol=symbol, starting_cash=starting_cash, min_size=500, max_size=1000, subscribe=True, log_orders=False, random_state = random_state) ])
    agent_id += 1

  # Add momentum agents.
  for i in range(num_mom):
    random_state = get_rand_obj(agent_seeds[agent_id])
    agents.extend([ MomentumAgent(agent_id, "Momentum Agent {}".format(agent_id), "MomentumAgent", symbol=symbol, starting_cash=starting_cash, min_size=1, max_size=10, subscribe=True, log_orders=False, random_state = random_state) ])
    agent_id += 1


  # Start the kernel running.  This call will not return until the
  # simulation is complete.  (Eventually this should be made
  # parallel for learning.)
  agent_saved_states = kernel.runner(
                agents = agents, startTime = kernelStartTime,
                stopTime = kernelStopTime, agentLatency = latency,
                latencyNoise = noise,
                defaultComputationDelay = defaultComputationDelay,
                oracle = oracle, log_dir = "{}_{}".format(log_dir,sim))

  obi_perf.append(agent_saved_states['agent_state'][0])

  print ("\n====== Experimental wallclock elapsed: {} ======\n".format(
                                  pd.Timestamp('now') - wallclock_start))

