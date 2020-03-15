import argparse
import numpy as np
import pandas as pd
import sys
import datetime as dt
from dateutil.parser import parse

from agent.ExchangeAgent import ExchangeAgent
from agent.NoiseAgent import NoiseAgent
from agent.ValueAgent import ValueAgent
from agent.market_makers.MarketMakerAgent import MarketMakerAgent
from agent.examples.MomentumAgent import MomentumAgent

from agent.execution.TWAPExecutionAgent import TWAPExecutionAgent
from agent.execution.VWAPExecutionAgent import VWAPExecutionAgent
from agent.execution.POVExecutionAgent import POVExecutionAgent

from Kernel import Kernel
from util import util
from util.order import LimitOrder
from util.oracle.ExternalFileOracle import ExternalFileOracle

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
                    help='Ticker (symbol) to use for simulation')
parser.add_argument('-d', '--historical-date',
                    required=True,
                    type=parse,
                    help='historical date being simulated in format YYYYMMDD.')
parser.add_argument('-f',
                    '--fundamental-file-path',
                    required=True,
                    help="Path to external fundamental file.")
parser.add_argument('-e',
                    '--execution_agents',
                    action='store_true',
                    help='Flag to add the execution agents')
parser.add_argument('-p',
                    '--pov',
                    type=float,
                    default=0.1,
                    help='Participation of Volume level')
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
parser.add_argument('--wide-book', action='store_true',
                    help='Store orderbook in `wide` format')

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

######################## Agents Config #########################################################################

# Historical date to simulate.
historical_date_pd = pd.to_datetime(args.historical_date)
mkt_open = historical_date_pd + pd.to_timedelta('10:45:00')
mkt_close = historical_date_pd + pd.to_timedelta('11:45:00')

agent_count, agents, agent_types = 0, [], []

# Hyperparameters
symbol = args.ticker
starting_cash = 10000000  # Cash in this simulator is always in CENTS.

# Oracle
symbols = {
    symbol : {
        'fundamental_file_path': args.fundamental_file_path,
        'random_state': np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64'))
    }
}
oracle = ExternalFileOracle(symbols)

r_bar = util.get_value_from_timestamp(oracle.fundamentals[symbol], mkt_open)

sigma_n = r_bar / 10
kappa = 1.67e-15
lambda_a = 7e-13

# 1) Exchange Agent
agents.extend([ExchangeAgent(id=0,
                             name="EXCHANGE_AGENT",
                             type="ExchangeAgent",
                             mkt_open=mkt_open,
                             mkt_close=mkt_close,
                             symbols=[symbol],
                             log_orders=True,
                             pipeline_delay=0,
                             computation_delay=0,
                             stream_history=sys.maxsize,
                             book_freq=0,
                             wide_book=args.wide_book,
                             random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                       dtype='uint64')))])
agent_types.extend("ExchangeAgent")
agent_count += 1

# 2) Noise Agents
num_noise = 5000
agents.extend([NoiseAgent(id=j,
                          name="NOISE_AGENT_{}".format(j),
                          type="NoiseAgent",
                          symbol=symbol,
                          starting_cash=starting_cash,
                          wakeup_time=util.get_wake_time(mkt_open, mkt_close),
                          random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')))
               for j in range(agent_count, agent_count + num_noise)])
agent_count += num_noise
agent_types.extend(['NoiseAgent'])

# 3) Value Agents
num_value = 100
agents.extend([ValueAgent(id=j,
                          name="VALUE_AGENT_{}".format(j),
                          type="ValueAgent",
                          symbol=symbol,
                          starting_cash=starting_cash,
                          sigma_n=sigma_n,
                          r_bar=r_bar,
                          kappa=kappa,
                          lambda_a=lambda_a,
                          random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')))
               for j in range(agent_count, agent_count + num_value)])
agent_count += num_value
agent_types.extend(['ValueAgent'])

# 4) Market Maker Agent
num_mm_agents = 1
agents.extend([MarketMakerAgent(id=j,
                                name="MARKET_MAKER_AGENT_{}".format(j),
                                type='MarketMakerAgent',
                                symbol=symbol,
                                starting_cash=starting_cash,
                                min_size=200,
                                max_size=201,
                                wake_up_freq="1S",
                                log_orders=False,
                                random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                          dtype='uint64')))
               for j in range(agent_count, agent_count + num_mm_agents)])
agent_count += num_mm_agents
agent_types.extend('MarketMakerAgent')


# 5) Momentum Agents
num_momentum_agents = 25
agents.extend([MomentumAgent(id=j,
                             name="MOMENTUM_AGENT_{}".format(j),
                             type="MomentumAgent",
                             symbol=symbol,
                             starting_cash=starting_cash,
                             min_size=1,
                             max_size=10,
                             log_orders=False,
                             random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                       dtype='uint64')))
               for j in range(agent_count, agent_count + num_momentum_agents)])
agent_count += num_momentum_agents
agent_types.extend("MomentumAgent")

# 6) Execution Agent Config
trade = True if args.execution_agents else False

#### Participation of Volume Agent parameters
pov_agent_start_time = '11:00:00'
pov_agent_end_time = '11:30:00'
pov_proportion_of_volume = args.pov
pov_quantity = 12e5
pov_frequency = '1min'
pov_direction = "BUY"

pov_agent = POVExecutionAgent(id=agent_count,
                              name='POV_EXECUTION_AGENT',
                              type='ExecutionAgent',
                              symbol=symbol,
                              starting_cash=starting_cash,
                              start_time=historical_date_pd+pd.to_timedelta(pov_agent_start_time),
                              end_time=historical_date_pd+pd.to_timedelta(pov_agent_end_time),
                              freq=pov_frequency,
                              lookback_period=pov_frequency,
                              pov=pov_proportion_of_volume,
                              direction=pov_direction,
                              quantity=pov_quantity,
                              trade=trade,
                              log_orders=True,
                              random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                          dtype='uint64')))

execution_agents = [pov_agent]

"""
twap_agent = TWAPExecutionAgent(id=agent_count,
                                name='TWAP_EXECUTION_AGENT',
                                type='ExecutionAgent',
                                symbol=symbol,
                                starting_cash=0,
                                start_time=historical_date_pd + pd.to_timedelta('11:00:00'),
                                end_time=historical_date_pd + pd.to_timedelta('13:00:00'),
                                freq=60,
                                direction='BUY',
                                quantity=12e3,
                                trade=trade,
                                log_orders=True,
                                random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                          dtype='uint64')))
execution_agents = [twap_agent]
"""
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

kernelStartTime = historical_date_pd + pd.to_timedelta('10:44:00')
kernelStopTime = historical_date_pd + pd.to_timedelta('11:46:00')

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
              oracle=oracle,
              log_dir=args.log_dir)

simulation_end_time = dt.datetime.now()
print("Simulation End Time: {}".format(simulation_end_time))
print("Time taken to run simulation: {}".format(simulation_end_time - simulation_start_time))