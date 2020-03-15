import argparse
import numpy as np
import pandas as pd
import sys
import datetime as dt

from agent.ExchangeAgent import ExchangeAgent
from agent.examples.MarketReplayAgent import MarketReplayAgent

from agent.execution.TWAPExecutionAgent import TWAPExecutionAgent
from agent.execution.VWAPExecutionAgent import VWAPExecutionAgent

from Kernel import Kernel
from util import util
from util.order import LimitOrder

########################################################################################################################
############################################### GENERAL CONFIG #########################################################

parser = argparse.ArgumentParser(description='Detailed options for market replay config.')

parser.add_argument('-c',
                    '--config',
                    required=True,
                    help='Name of config file to execute')
parser.add_argument('-t',
                    '--ticker',
                    required=True,
                    help='Name of the stock/symbol')
parser.add_argument('-d',
                    '--date',
                    required=True,
                    help='Historical date')
parser.add_argument('-e',
                    '--execution_agents',
                    action='store_true',
                    help='Flag to add the execution agents')
parser.add_argument('-s',
                    '--seed',
                    type=int,
                    default=None,
                    help='numpy.random.seed() for simulation')
parser.add_argument('-l',
                    '--log_dir',
                    default=None,
                    help='Log directory name (default: unix timestamp at program start)')
parser.add_argument('-v',
                    '--verbose',
                    action='store_true',
                    help='Maximum verbosity!')
parser.add_argument('--config_help',
                    action='store_true',
                    help='Print argument options for this config file')

args, remaining_args = parser.parse_known_args()

if args.config_help:
    parser.print_help()
    sys.exit()

seed = args.seed  # Random seed specification on the command line.
if not seed: seed = int(pd.Timestamp.now().timestamp() * 1000000) % (2 ** 32 - 1)
np.random.seed(seed)

util.silent_mode = not args.verbose
LimitOrder.silent_mode = not args.verbose

simulation_start_time = dt.datetime.now()
print("Simulation Start Time: {}".format(simulation_start_time))
print("Configuration seed: {}".format(seed))
print("Symbol: {}".format(args.ticker))
print("Date: {}".format(args.date))

######################## Agents Config #########################################################################

# Historical date to simulate.
historical_date = args.date
historical_date_pd = pd.to_datetime(historical_date)
symbol = args.ticker

agent_count, agents, agent_types = 0, [], []

# 1) Exchange Agent
mkt_open = historical_date_pd + pd.to_timedelta('09:00:00')
mkt_close = historical_date_pd + pd.to_timedelta('16:00:00')

print("Market Open : {}".format(mkt_open))
print("Market Close: {}".format(mkt_close))

agents.extend([ExchangeAgent(id=0,
                             name="EXCHANGE_AGENT",
                             type="ExchangeAgent",
                             mkt_open=mkt_open,
                             mkt_close=mkt_close,
                             symbols=[symbol],
                             log_orders=True,
                             pipeline_delay=0,
                             computation_delay=0,
                             stream_history=10,
                             book_freq=0,
                             random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                       dtype='uint64')))])
agent_types.extend("ExchangeAgent")
agent_count += 1

# 2) Market Replay Agent
file_name = f'DOW30/{symbol}/{symbol}.{historical_date}'
orders_file_path = f'/efs/data/{file_name}'

agents.extend([MarketReplayAgent(id=1,
                                 name="MARKET_REPLAY_AGENT",
                                 type='MarketReplayAgent',
                                 symbol=symbol,
                                 log_orders=False,
                                 date=historical_date_pd,
                                 start_time=mkt_open,
                                 end_time=mkt_close,
                                 orders_file_path=orders_file_path,
                                 processed_orders_folder_path='/efs/data/marketreplay/',
                                 starting_cash=0,
                                 random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                           dtype='uint64')))])
agent_types.extend("MarketReplayAgent")
agent_count += 1

# 3) Execution Agent Config
trade = True if args.execution_agents else False

twap_agent = TWAPExecutionAgent(id=agent_count,
                                name='TWAP_EXECUTION_AGENT',
                                type='ExecutionAgent',
                                symbol=symbol,
                                starting_cash=0,
                                start_time=historical_date_pd + pd.to_timedelta('10:00:00'),
                                end_time=historical_date_pd + pd.to_timedelta('12:00:00'),
                                freq=60,
                                direction='BUY',
                                quantity=12e3,
                                trade=trade,
                                log_orders=True,
                                random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                          dtype='uint64')))
execution_agents = [twap_agent]
"""
vwap_agent = VWAPExecutionAgent(id=agent_count,
                                name='VWAP_EXECUTION_AGENT',
                                type='ExecutionAgent',
                                symbol=symbol,
                                starting_cash=0,
                                start_time=historical_date_pd + pd.to_timedelta('10:00:00'),
                                end_time=historical_date_pd + pd.to_timedelta('12:00:00'),
                                freq=60,
                                direction='BUY',
                                quantity=12e3,
                                volume_profile_path=None,
                                trade=trade,
                                log_orders=True,
                                random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                          dtype='uint64')))
execution_agents = [vwap_agent]
"""
agents.extend(execution_agents)
agent_types.extend("ExecutionAgent")
agent_count += 1

print("Number of Agents: {}".format(agent_count))

########################################################################################################################
########################################### KERNEL AND OTHER CONFIG ####################################################

kernel = Kernel("Market Replay Kernel",
                random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')))

kernelStartTime = historical_date_pd
kernelStopTime = historical_date_pd + pd.to_timedelta('16:01:00')

defaultComputationDelay = 0
latency = np.zeros((agent_count, agent_count))
noise = [0.0]

kernel.runner(agents=agents,
              startTime=kernelStartTime,
              stopTime=kernelStopTime,
              agentLatency=latency,
              latencyNoise=noise,
              defaultComputationDelay=defaultComputationDelay,
              defaultLatency=0,
              oracle=None,
              log_dir=args.log_dir)

simulation_end_time = dt.datetime.now()
print("Simulation End Time: {}".format(simulation_end_time))
print("Time taken to run simulation: {}".format(simulation_end_time - simulation_start_time))