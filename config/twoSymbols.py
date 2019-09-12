from Kernel import Kernel
from agent.ExchangeAgent import ExchangeAgent
from agent.etf.EtfPrimaryAgent import EtfPrimaryAgent
from agent.HeuristicBeliefLearningAgent import HeuristicBeliefLearningAgent
from agent.examples.ImpactAgent import ImpactAgent
from agent.ZeroIntelligenceAgent import ZeroIntelligenceAgent
from agent.examples.MomentumAgent import MomentumAgent
from agent.etf.EtfArbAgent import EtfArbAgent
from agent.etf.EtfMarketMakerAgent import EtfMarketMakerAgent
from util.order import LimitOrder
from util.oracle.MeanRevertingOracle import MeanRevertingOracle
from util import util

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
# Here we go for 8:00 PM the same day to reflect the ETF primary market
kernelStopTime = midnight + pd.to_timedelta('20:00:00')

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
symbols = { 'SYM1' : { 'r_bar' : 100000, 'kappa' : 0.05, 'sigma_s' : sigma_s }, 'SYM2': { 'r_bar' : 150000, 'kappa' : 0.05, 'sigma_s' : sigma_s } , 'ETF' : { 'r_bar' : 250000, 'kappa' : 0.10, 'sigma_s' : np.sqrt(2) * sigma_s, 'portfolio': ['SYM1', 'SYM2']}}
#symbols = { 'IBM' : { 'r_bar' : 100000, 'kappa' : 0.05, 'sigma_s' : sigma_s }, 'GOOG' : { 'r_bar' : 150000, 'kappa' : 0.05, 'sigma_s' : sigma_s } }
symbols_full = symbols.copy()

#seed=np.random.randint(low=0,high=2**32)
#seed = 2000

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
#mkt_close = midnight + pd.to_timedelta('09:30:00.001')
mkt_close = midnight + pd.to_timedelta('9:30:00.000001')


# Configure an appropriate oracle for all traded stocks.
# All agents requiring the same type of Oracle will use the same oracle instance.
oracle = MeanRevertingOracle(mkt_open, mkt_close, symbols)


# Create the exchange.
num_exchanges = 1
agents.extend([ ExchangeAgent(j, "Exchange Agent {}".format(j), "ExchangeAgent", mkt_open, mkt_close, [s for s in symbols_full], log_orders=log_orders, book_freq=book_freq, pipeline_delay = 0, computation_delay = 0, stream_history = 10, random_state = np.random.RandomState(seed=np.random.randint(low=0,high=2**32)))
                for j in range(agent_count, agent_count + num_exchanges) ])
agent_types.extend(["ExchangeAgent" for j in range(num_exchanges)])
agent_count += num_exchanges


# Let's open the exchange at 5:00 PM.
prime_open = midnight + pd.to_timedelta('17:00:00')

# And close it at 5:00:01 PM
prime_close = midnight + pd.to_timedelta('17:00:01')

# Create the primary.
num_primes = 1
agents.extend([ EtfPrimaryAgent(j, "ETF Primary Agent {}".format(j), "EtfPrimaryAgent", prime_open, prime_close, 'ETF', pipeline_delay = 0, computation_delay = 0, random_state = np.random.RandomState(seed=np.random.randint(low=0,high=2**32)))
                for j in range(agent_count, agent_count + num_primes) ])
agent_types.extend(["EtfPrimeAgent" for j in range(num_primes)])
agent_count += num_primes


### Configure some zero intelligence agents.

# Cash in this simulator is always in CENTS.
starting_cash = 10000000

# Here are the zero intelligence agents.
symbol1 = 'SYM1'
symbol2 = 'SYM2'
symbol3 = 'ETF'
print(symbols_full)
s1 = symbols_full[symbol1]
s2 = symbols_full[symbol2]
s3 = symbols_full[symbol3]

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
zi = [ (100, 0, 250, 1), (100, 0, 500, 1), (100, 0, 10000, 0.8), (100, 0, 10000, 1), (100, 0, 2000, 0.8), (100, 250, 500, 0.8), (100, 250, 500, 1) ]
hbl = [ (75, 250, 500, 1, 2), (75, 250, 500, 1, 3), (75, 250, 500, 1, 5), (75, 250, 500, 1, 8) ] 



# ZI strategy split.
for i,x in enumerate(zi):
  strat_name = "Type {} [{} <= R <= {}, eta={}]".format(i+1, x[1], x[2], x[3])
  agents.extend([ ZeroIntelligenceAgent(j, "ZI Agent {} {}".format(j, strat_name), "ZeroIntelligenceAgent {}".format(strat_name), random_state = np.random.RandomState(seed=np.random.randint(low=0,high=2**32)),log_orders=log_orders, symbol=symbol1, starting_cash=starting_cash, sigma_n=sigma_n, r_bar=s1['r_bar'], q_max=10, sigma_pv=5000000, R_min=x[1], R_max=x[2], eta=x[3], lambda_a=0.005) for j in range(agent_count,agent_count+x[0]) ])
  agent_types.extend([ "ZeroIntelligenceAgent {}".format(strat_name) for j in range(x[0]) ])
  agent_count += x[0]

for i,x in enumerate(zi):
  strat_name = "Type {} [{} <= R <= {}, eta={}]".format(i+1, x[1], x[2], x[3])
  agents.extend([ ZeroIntelligenceAgent(j, "ZI Agent {} {}".format(j, strat_name), "ZeroIntelligenceAgent {}".format(strat_name), random_state = np.random.RandomState(seed=np.random.randint(low=0,high=2**32)),log_orders=log_orders, symbol=symbol2, starting_cash=starting_cash, sigma_n=sigma_n, r_bar=s2['r_bar'], q_max=10, sigma_pv=5000000, R_min=x[1], R_max=x[2], eta=x[3], lambda_a=0.005) for j in range(agent_count,agent_count+x[0]) ])
  agent_types.extend([ "ZeroIntelligenceAgent {}".format(strat_name) for j in range(x[0]) ])
  agent_count += x[0]
    
for i,x in enumerate(zi):
  strat_name = "Type {} [{} <= R <= {}, eta={}]".format(i+1, x[1], x[2], x[3])
  agents.extend([ ZeroIntelligenceAgent(j, "ZI Agent {} {}".format(j, strat_name), "ZeroIntelligenceAgent {}".format(strat_name), random_state = np.random.RandomState(seed=np.random.randint(low=0,high=2**32)),log_orders=log_orders, symbol=symbol3, starting_cash=starting_cash, sigma_n=sigma_n, portfolio = {'SYM1':s1['r_bar'], 'SYM2': s2['r_bar']}, q_max=10, sigma_pv=5000000, R_min=x[1], R_max=x[2], eta=x[3], lambda_a=0.005) for j in range(agent_count,agent_count+x[0]) ])
  agent_types.extend([ "ZeroIntelligenceAgent {}".format(strat_name) for j in range(x[0]) ])
  agent_count += x[0]

    
# HBL strategy split.
for i,x in enumerate(hbl):
  strat_name = "Type {} [{} <= R <= {}, eta={}, L={}]".format(i+1, x[1], x[2], x[3], x[4])
  agents.extend([ HeuristicBeliefLearningAgent(j, "HBL Agent {} {}".format(j, strat_name), "HeuristicBeliefLearningAgent {}".format(strat_name), random_state = np.random.RandomState(seed=np.random.randint(low=0,high=2**32)), log_orders=log_orders, symbol=symbol1, starting_cash=starting_cash, sigma_n=sigma_n, r_bar=s1['r_bar'], q_max=10, sigma_pv=5000000, R_min=x[1], R_max=x[2], eta=x[3], lambda_a=0.005, L=x[4]) for j in range(agent_count,agent_count+x[0]) ])
  agent_types.extend([ "HeuristicBeliefLearningAgent {}".format(strat_name) for j in range(x[0]) ])
  agent_count += x[0]

for i,x in enumerate(hbl):
  strat_name = "Type {} [{} <= R <= {}, eta={}, L={}]".format(i+1, x[1], x[2], x[3], x[4])
  agents.extend([ HeuristicBeliefLearningAgent(j, "HBL Agent {} {}".format(j, strat_name), "HeuristicBeliefLearningAgent {}".format(strat_name), random_state = np.random.RandomState(seed=np.random.randint(low=0,high=2**32)), log_orders=log_orders, symbol=symbol2, starting_cash=starting_cash, sigma_n=sigma_n, r_bar=s2['r_bar'], q_max=10, sigma_pv=5000000, R_min=x[1], R_max=x[2], eta=x[3], lambda_a=0.005, L=x[4]) for j in range(agent_count,agent_count+x[0]) ])
  agent_types.extend([ "HeuristicBeliefLearningAgent {}".format(strat_name) for j in range(x[0]) ])
  agent_count += x[0]

for i,x in enumerate(hbl):
  strat_name = "Type {} [{} <= R <= {}, eta={}, L={}]".format(i+1, x[1], x[2], x[3], x[4])
  agents.extend([ HeuristicBeliefLearningAgent(j, "HBL Agent {} {}".format(j, strat_name), "HeuristicBeliefLearningAgent {}".format(strat_name), random_state = np.random.RandomState(seed=np.random.randint(low=0,high=2**32)), log_orders=log_orders, symbol=symbol3, starting_cash=starting_cash, sigma_n=sigma_n, portfolio = {'SYM1':s1['r_bar'], 'SYM2': s2['r_bar']}, q_max=10, sigma_pv=5000000, R_min=x[1], R_max=x[2], eta=x[3], lambda_a=0.005, L=x[4]) for j in range(agent_count,agent_count+x[0]) ])
  agent_types.extend([ "HeuristicBeliefLearningAgent {}".format(strat_name) for j in range(x[0]) ])
  agent_count += x[0]
    
# Trend followers agent    
i = agent_count
lookback = 10
num_tf = 20
for j in range(num_tf):
  agents.append(MomentumAgent(i, "Momentum Agent {}".format(i), symbol=symbol1, starting_cash = starting_cash, lookback=lookback, random_state = np.random.RandomState(seed=np.random.randint(low=0,high=2**32)), log_orders = log_orders))
  agent_types.append("MomentumAgent {}".format(i))
  i+=1
agent_count += num_tf

#for j in range(num_tf):
  #agents.append(MomentumAgent(i, "Momentum Agent {}".format(i), symbol=symbol2, startingCash = starting_cash, lookback=lookback))
  #agent_types.append("MomentumAgent {}".format(i))
  #i+=1
#agent_count += num_tf

#for j in range(num_tf):
  #agents.append(MomentumAgent(i, "Momentum Agent {}".format(i), symbol=symbol3, startingCash = starting_cash, lookback=lookback))
  #agent_types.append("MomentumAgent {}".format(i))
  #i+=1
#agent_count += num_tf
    
# ETF arbitrage agent    
i = agent_count
gamma = 0
num_arb = 25
for j in range(num_arb):
  agents.append(EtfArbAgent(i, "Etf Arb Agent {}".format(i), "EtfArbAgent", portfolio = ['SYM1','SYM2'], gamma = gamma, starting_cash = starting_cash, lambda_a=0.005, log_orders=log_orders, random_state = np.random.RandomState(seed=np.random.randint(low=0,high=2**32))))
  agent_types.append("EtfArbAgent {}".format(i))
  i+=1
agent_count += num_arb

# ETF market maker agent    
#i = agent_count
#gamma = 100
#num_mm = 10
mm = [(5,0),(5,50),(5,100),(5,200),(5,300)]
#for j in range(num_mm):
  #agents.append(EtfMarketMakerAgent(i, "Etf MM Agent {}".format(i), "EtfMarketMakerAgent", portfolio = ['IBM','GOOG'], gamma = gamma, starting_cash = starting_cash, lambda_a=0.005, log_orders=log_orders, random_state = np.random.RandomState(seed=np.random.randint(low=0,high=2**32))))
  #agent_types.append("EtfMarketMakerAgent {}".format(i))
  #i+=1
#agent_count += num_mm

for i,x in enumerate(mm):
  strat_name = "Type {} [gamma = {}]".format(i+1, x[1])
  print(strat_name)
  agents.extend([ EtfMarketMakerAgent(j, "Etf MM Agent {} {}".format(j, strat_name), "EtfMarketMakerAgent {}".format(strat_name), portfolio = ['SYM1','SYM2'], gamma = x[1], starting_cash = starting_cash, lambda_a=0.005, log_orders=log_orders, random_state = np.random.RandomState(seed=np.random.randint(low=0,high=2**32))) for j in range(agent_count,agent_count+x[0]) ])
  agent_types.extend([ "EtfMarketMakerAgent {}".format(strat_name) for j in range(x[0]) ])
  agent_count += x[0]

# Impact agent.

# 200 time steps in...
impact_time = midnight + pd.to_timedelta('09:30:00.0000002')

i = agent_count
agents.append(ImpactAgent(i, "Impact Agent1 {}".format(i), "ImpactAgent1", symbol = "SYM1", starting_cash = starting_cash, impact = impact, impact_time = impact_time, random_state = np.random.RandomState(seed=np.random.randint(low=0,high=2**32))))
agent_types.append("ImpactAgent 1 {}".format(i))
agent_count += 1

impact_time = midnight + pd.to_timedelta('09:30:00.0000005')
i = agent_count
agents.append(ImpactAgent(i, "Impact Agent2 {}".format(i), "ImpactAgent2", symbol = "SYM1", starting_cash = starting_cash, impact = impact, impact_time = impact_time, random_state = np.random.RandomState(seed=np.random.randint(low=0,high=2**32))))
agent_types.append("ImpactAgent 2 {}".format(i))
agent_count += 1

#i = agent_count
#agents.append(ImpactAgent(i, "Impact Agent3 {}".format(i), "ImpactAgent3", symbol = "ETF", starting_cash = starting_cash, greed = greed, impact = impact, impact_time = impact_time, random_state = np.random.RandomState(seed=np.random.randint(low=0,high=2**32))))
#agent_types.append("ImpactAgent 3 {}".format(i))
#agent_count += 1

### Configure a simple message latency matrix for the agents.  Each entry is the minimum
### nanosecond delay on communication [from][to] agent ID.

# Square numpy array with dimensions equal to total agent count.  Most agents are handled
# at init, drawn from a uniform distribution from:
# Times Square (3.9 miles from NYSE, approx. 21 microseconds at the speed of light) to:
# Pike Place Starbucks in Seattle, WA (2402 miles, approx. 13 ms at the speed of light).
# Other agents can be explicitly set afterward (and the mirror half of the matrix is also).

# This configures all agents to a starting latency as described above.
#latency = np.random.uniform(low = 21000, high = 13000000, size=(len(agent_types),len(agent_types)))
latency = np.random.uniform(low = 10, high = 100, size=(len(agent_types),len(agent_types)))

# Overriding the latency for certain agent pairs happens below, as does forcing mirroring
# of the matrix to be symmetric.
for i, t1 in zip(range(latency.shape[0]), agent_types):
  for j, t2 in zip(range(latency.shape[1]), agent_types):
    # Three cases for symmetric array.  Set latency when j > i, copy it when i > j, same agent when i == j.
    if j > i:
      # Arb agents should be the fastest in the market.
      if (("ExchangeAgent" in t1 and "EtfArbAgent" in t2) 
          or ("ExchangeAgent" in t2 and "EtfArbAgent" in t1)):
        #latency[i,j] = 20000
        latency[i,j] = 5
      elif (("ExchangeAgent" in t1 and "EtfMarketMakerAgent" in t2) 
          or ("ExchangeAgent" in t2 and "EtfMarketMakerAgent" in t1)):
        #latency[i,j] = 20000
        latency[i,j] = 1
      elif (("ExchangeAgent" in t1 and "ImpactAgent" in t2) 
          or ("ExchangeAgent" in t2 and "ImpactAgent" in t1)):
        #latency[i,j] = 20000
        latency[i,j] = 1
        
    elif i > j:
      # This "bottom" half of the matrix simply mirrors the top.
      if (("ExchangeAgent" in t1 and "EtfArbAgent" in t2) 
          or ("ExchangeAgent" in t2 and "EtfArbAgent" in t1)):
        #latency[i,j] = 20000
        latency[i,j] = 5
      elif (("ExchangeAgent" in t1 and "EtfMarketMakerAgent" in t2) 
          or ("ExchangeAgent" in t2 and "EtfMarketMakerAgent" in t1)):
        #latency[i,j] = 20000
        latency[i,j] = 1
      elif (("ExchangeAgent" in t1 and "ImpactAgent" in t2) 
          or ("ExchangeAgent" in t2 and "ImpactAgent" in t1)):
        #latency[i,j] = 20000
        latency[i,j] = 1
      else: latency[i,j] = latency[j,i]
    else:
      # This is the same agent.  How long does it take to reach localhost?  In our data center, it actually
      # takes about 20 microseconds.
      #latency[i,j] = 10000
      latency[i,j] = 1

# Configure a simple latency noise model for the agents.
# Index is ns extra delay, value is probability of this delay being applied.
# In this config, there is no latency (noisy or otherwise).
noise = [ 0.0 ]



# Start the kernel running.
kernel.runner(agents = agents, startTime = kernelStartTime,
              stopTime = kernelStopTime, agentLatency = latency,
              latencyNoise = noise,
              defaultComputationDelay = defaultComputationDelay,
              oracle = oracle, log_dir = log_dir)

