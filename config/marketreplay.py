from Kernel import Kernel

from agent.MarketReplayAgent import MarketReplayAgent
from agent.ExchangeAgent import ExchangeAgent
from agent.ExperimentalAgent import ExperimentalAgent
from util.oracle.OrderBookOracle import OrderBookOracle

from util import util
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
date = '2012-06-21'
date_pd = pd.to_datetime(date)
print("Historical Simulation Date: {}".format(date))

agents = []

# 3) ExchangeAgent Config
num_exchanges = 1
mkt_open  = date_pd + pd.to_timedelta('09:30:00')
mkt_close = date_pd + pd.to_timedelta('09:30:05')
print("ExchangeAgent num_exchanges: {}".format(num_exchanges))
print("ExchangeAgent mkt_open: {}".format(mkt_open))
print("ExchangeAgent mkt_close: {}".format(mkt_close))

ea = ExchangeAgent(id = 0,
                   name = 'Exchange_Agent',
                   type = 'ExchangeAgent',
                   mkt_open = mkt_open,
                   mkt_close = mkt_close,
                   symbols = symbols,
                   log_orders=log_orders,
                   book_freq = '1s',
                   pipeline_delay = 0,
                   computation_delay = 0,
                   stream_history = 10,
                   random_state = random_state)

agents.extend([ea])

# 4) MarketReplayAgent Config
num_mr_agents  = 1
cash_mr_agents = 10000000

mr_agents = [MarketReplayAgent(id      = 1,
                              name    = "Market_Replay_Agent",
                              symbol  = symbols[0],
                              date    = date,
                              startingCash = cash_mr_agents,
                              random_state = random_state)]
agents.extend(mr_agents)

# 5) ExperimentalAgent Config
num_exp_agents  = 1
cash_exp_agents = 10000000

exp_agents = [ExperimentalAgent(id      = 2,
                               name    = "Experimental_Agent",
                               symbol  = symbols[0],
                               startingCash = cash_exp_agents,
                               execution_timestamp = pd.Timestamp("2012-06-21 09:30:02"),
                               quantity = 1000,
                               is_buy_order = True,
                               limit_price = 500,
                               random_state = random_state)]
agents.extend(exp_agents)
#######################################################################################################################

# 6) Kernel Parameters
kernel = Kernel("Market Replay Kernel", random_state = random_state)

kernelStartTime = date_pd + pd.to_timedelta('09:30:00')
kernelStopTime = date_pd + pd.to_timedelta('09:30:05')
defaultComputationDelay = 0
latency = np.zeros((3, 3))
noise = [ 0.0 ]

# 7) Data Oracle
oracle = OrderBookOracle(symbol='AAPL',
                         date='2012-06-21',
                         orderbook_file_path='C:/_code/py/air/abides_open_source/abides/data/LOBSTER/AAPL_2012-06-21_34200000_57600000_orderbook_10.csv',
                         message_file_path='C:/_code/py/air/abides_open_source/abides/data/LOBSTER/AAPL_2012-06-21_34200000_57600000_message_10.csv',
                         num_price_levels=10)

kernel.runner(agents = agents, startTime = kernelStartTime,
              stopTime = kernelStopTime, agentLatency = latency,
              latencyNoise = noise,
              defaultComputationDelay = defaultComputationDelay,
              defaultLatency=0,
              oracle = oracle, log_dir = args.log_dir)

simulation_end_time = dt.datetime.now()
print ("Simulation End Time: {}".format(simulation_end_time))
print ("Time taken to run simulation: {}".format(simulation_end_time - simulation_start_time))