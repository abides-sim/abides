from Kernel import Kernel

from agent.MarketReplayAgent import MarketReplayAgent
from agent.ExchangeAgent import ExchangeAgent
from agent.ExperimentalAgent import ExperimentalAgent

from util import util
from util.oracle.RandomOrderBookOracle import RandomOrderBookOracle
from util.order import LimitOrder

import datetime as dt
import numpy as np
import pandas as pd
import sys
import argparse

parser = argparse.ArgumentParser(description='Options for Market Replay Agent Config.')

# General Config for all agents
parser.add_argument('-c', '--config', required=True,
                    help='Name of config file to execute')
parser.add_argument('-s', '--seed', type=int, default=None,
                    help='numpy.random.seed() for simulation')
parser.add_argument('-l', '--log_dir', default=None,
                    help='Log directory name (default: unix timestamp at program start)')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='Maximum verbosity!')
parser.add_argument('-o', '--log_orders', action='store_true',
                    help='Log every order-related action by every agent.')
parser.add_argument('--config_help', action='store_true',
                    help='Print argument options for this config file')

args, remaining_args = parser.parse_known_args()

log_orders = args.log_orders

if args.config_help:
  parser.print_help()
  sys.exit()

# Simulation Start Time
simulation_start_time = dt.datetime.now()
print ("Simulation Start Time: {}".format(simulation_start_time))

# Random Seed Config
seed = args.seed
if not seed: seed = int(pd.Timestamp.now().timestamp() * 1000000) % (2**32 - 1)
np.random.seed(seed)
print ("Configuration seed: {}".format(seed))

random_state = np.random.RandomState(seed=np.random.randint(low=1))

util.silent_mode = not args.verbose
LimitOrder.silent_mode = not args.verbose
print ("Silent mode: {}".format(util.silent_mode))

######################## Agents Config #########################################################################

# 1) Symbols
symbols = ['AAPL']
print("Symbols traded: {}".format(symbols))

# 2) Historical Date to simulate
date = '2019-06-19'
date_pd = pd.to_datetime(date)
print("Historical Simulation Date: {}".format(date))

agents = []

# 3) ExchangeAgent Config
num_exchanges = 1
mkt_open  = date_pd + pd.to_timedelta('09:30:00')
mkt_close = date_pd + pd.to_timedelta('09:35:00')
print("ExchangeAgent num_exchanges: {}".format(num_exchanges))
print("ExchangeAgent mkt_open: {}".format(mkt_open))
print("ExchangeAgent mkt_close: {}".format(mkt_close))

ea = ExchangeAgent(id         = 0,
                   name       = 'Exchange_Agent',
                   type       = 'ExchangeAgent',
                   mkt_open   = mkt_open,
                   mkt_close  = mkt_close,
                   symbols    = symbols,
                   log_orders = log_orders,
                   book_freq  = None,
                   pipeline_delay = 0,
                   computation_delay = 0,
                   stream_history = 10,
                   random_state = random_state)

agents.extend([ea])

# 4) MarketReplayAgent Config
market_replay_agents = [MarketReplayAgent(id     = 1,
                                          name    = "Market_Replay_Agent",
                                          type    = 'MarketReplayAgent',
                                          symbol  = symbols[0],
                                          log_orders = log_orders,
                                          date    = date,
                                          starting_cash = 0,
                                          random_state = random_state)]
agents.extend(market_replay_agents)

# 5) ExperimentalAgent Config
experimental_agents = [ExperimentalAgent(id      = 2,
                               name    = "Experimental_Agent",
                               symbol  = symbols[0],
                               starting_cash = 10000000,
                               log_orders = log_orders,
                               execution_timestamp = pd.Timestamp("2019-06-19 09:32:00"),
                               quantity = 1000,
                               is_buy_order = True,
                               limit_price = 500,
                               random_state = random_state)]
agents.extend(experimental_agents)
#######################################################################################################################

# 6) Kernel Parameters
kernel = Kernel("Market Replay Kernel", random_state = random_state)

kernelStartTime = date_pd + pd.to_timedelta('09:30:00')
kernelStopTime = date_pd + pd.to_timedelta('09:35:00')
defaultComputationDelay = 0
latency = np.zeros((3, 3))
noise = [ 0.0 ]

oracle = RandomOrderBookOracle(symbol = 'AAPL',
                               market_open_ts =  mkt_open,
                               market_close_ts = mkt_close,
                               buy_price_range = [90, 105],
                               sell_price_range = [95, 110],
                               quantity_range = [50, 500],
                               seed=seed)

kernel.runner(agents = agents, startTime = kernelStartTime,
              stopTime = kernelStopTime, agentLatency = latency,
              latencyNoise = noise,
              defaultComputationDelay = defaultComputationDelay,
              defaultLatency=0,
              oracle = oracle, log_dir = args.log_dir)

simulation_end_time = dt.datetime.now()
print ("Simulation End Time: {}".format(simulation_end_time))
print ("Time taken to run simulation: {}".format(simulation_end_time - simulation_start_time))