import argparse
import numpy as np
import pandas as pd
import sys
import datetime as dt
from dateutil.parser import parse

from Kernel import Kernel
from util import util
from util.order import LimitOrder
from util.oracle.SparseMeanRevertingOracle import SparseMeanRevertingOracle
from util.oracle.ExternalFileOracle import ExternalFileOracle
from model.LatencyModel import LatencyModel

from agent.ExchangeAgent import ExchangeAgent
from agent.NoiseAgent import NoiseAgent
from agent.ValueAgent import ValueAgent
from agent.market_makers.AdaptiveMarketMakerAgent import AdaptiveMarketMakerAgent
from agent.examples.MomentumAgent import MomentumAgent

from agent.execution.TWAPExecutionAgent import TWAPExecutionAgent
from agent.execution.VWAPExecutionAgent import VWAPExecutionAgent
from agent.execution.POVExecutionAgent import POVExecutionAgent

########################################################################################################################
############################################### GENERAL CONFIG #########################################################

parser = argparse.ArgumentParser(description='Detailed options for the config.')

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
                    required=False,
                    help="Path to external fundamental file.")
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

######################## Agents Config #########################################################################

# Historical date to simulate.
historical_date = pd.to_datetime(args.historical_date)
mkt_open = historical_date + pd.to_timedelta('09:30:00')
mkt_close = historical_date + pd.to_timedelta('11:30:00')

agent_count, agents, agent_types = 0, [], []

# Hyperparameters
symbol = args.ticker
starting_cash = 10000000  # Cash in this simulator is always in CENTS.

# Choice between two oracles for generating the fundamental time series
# (1) Sparse Mean Reverting Oracle

r_bar = 1e5
symbols = {symbol: {'r_bar': r_bar,
                    'kappa': 1.67e-16,
                    'sigma_s': 0,
                    'fund_vol': 1e-8,   # volatility of fundamental time series.
                    'megashock_lambda_a': 2.77778e-18,
                    'megashock_mean': 1e3,
                    'megashock_var': 5e4,
                    'random_state': np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32))}}

oracle = SparseMeanRevertingOracle(mkt_open, mkt_close, symbols)


# (2) External File Oracle
"""
symbols = {
    symbol: {
        'fundamental_file_path': args.fundamental_file_path,
        'random_state': np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32))
    }
}
oracle = ExternalFileOracle(symbols)
r_bar = oracle.fundamentals[symbol].values[0]
"""

# Agents:

# 1) Exchange Agent
# How many orders in the past to store for transacted volume computation
# stream_history_length = int(pd.to_timedelta(args.mm_wake_up_freq).total_seconds() * 100)
stream_history_length = 25000
agents.extend([ExchangeAgent(id=0,
                             name="ExchangeAgent",
                             type="ExchangeAgent",
                             mkt_open=mkt_open,
                             mkt_close=mkt_close,
                             symbols=[symbol],
                             log_orders=True,
                             pipeline_delay=0,
                             computation_delay=0,
                             stream_history=stream_history_length,
                             book_freq=0,
                             wide_book=True,
                             random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32)))])
agent_types.extend("ExchangeAgent")
agent_count += 1

# 2) Noise Agents
num_noise = 5000
noise_mkt_open = historical_date + pd.to_timedelta("09:00:00")
noise_mkt_close = historical_date + pd.to_timedelta("16:00:00")
agents.extend([NoiseAgent(id=j,
                          name="NoiseAgent_{}".format(j),
                          type="NoiseAgent",
                          symbol=symbol,
                          starting_cash=starting_cash,
                          wakeup_time=util.get_wake_time(noise_mkt_open, noise_mkt_close),
                          log_orders=False,
                          log_to_file=False,
                          random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32)))
               for j in range(agent_count, agent_count + num_noise)])
agent_count += num_noise
agent_types.extend(['NoiseAgent'])

# 3) Value Agents
num_value = 100
agents.extend([ValueAgent(id=j,
                          name="ValueAgent_{}".format(j),
                          type="ValueAgent",
                          symbol=symbol,
                          starting_cash=starting_cash,
                          sigma_n=r_bar / 10,
                          r_bar=r_bar,
                          kappa=1.67e-15,
                          lambda_a=7e-11,
                          log_orders=False,
                          log_to_file=False,
                          random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32)))
               for j in range(agent_count, agent_count + num_value)])
agent_count += num_value
agent_types.extend(['ValueAgent'])

# 4) Market Maker Agents

"""
window_size ==  Spread of market maker (in ticks) around the mid price
pov == Percentage of transacted volume seen in previous `mm_wake_up_freq` that
       the market maker places at each level
num_ticks == Number of levels to place orders in around the spread
wake_up_freq == How often the market maker wakes up

"""

# each elem of mm_params is tuple (window_size, pov, num_ticks, wake_up_freq, min_order_size)
mm_params = [('adaptive', 0.025, 10, '10S', 1),
             ('adaptive', 0.025, 10, '10S', 1)]

num_mm_agents = len(mm_params)

agents.extend([AdaptiveMarketMakerAgent(id=j,
                                        name="AdaptiveMarketMakerAgent_{}".format(j),
                                        type='AdaptiveMarketMakerAgent',
                                        symbol=symbol,
                                        starting_cash=starting_cash,
                                        pov=mm_params[idx][1],
                                        min_order_size=mm_params[idx][4],
                                        window_size=mm_params[idx][0],
                                        num_ticks=mm_params[idx][2],
                                        wake_up_freq=mm_params[idx][3],
                                        cancel_limit_delay=50,
                                        skew_beta=0,
                                        level_spacing=5,
                                        spread_alpha=0.75,
                                        backstop_quantity=50000,
                                        log_orders=True,
                                        random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32)))
               for idx, j in enumerate(range(agent_count, agent_count + num_mm_agents))])
agent_count += num_mm_agents
agent_types.extend('AdaptiveMarketMakerAgent')

# 5) Momentum Agents
num_momentum_agents = 25
agents.extend([MomentumAgent(id=j,
                             name="MOMENTUM_AGENT_{}".format(j),
                             type="MomentumAgent",
                             symbol=symbol,
                             starting_cash=starting_cash,
                             min_size=1,
                             max_size=10,
                             wake_up_freq='20s',
                             log_orders=True,
                             random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32)))
               for j in range(agent_count, agent_count + num_momentum_agents)])
agent_count += num_momentum_agents
agent_types.extend("MomentumAgent")

# 6) Execution Agent Config
trade = True if args.execution_agents else False

execution_agent_start_time = historical_date + pd.to_timedelta("10:00:00")
execution_agent_end_time = historical_date + pd.to_timedelta("11:00:00")
execution_quantity = 12e5
execution_frequency = '1min'
execution_direction = "BUY"
execution_time_horizon = pd.date_range(start=execution_agent_start_time, end=execution_agent_end_time,
                                       freq=execution_frequency)

twap_agent = TWAPExecutionAgent(id=agent_count,
                                name='TWAPExecutionAgent',
                                type='ExecutionAgent',
                                symbol=symbol,
                                starting_cash=0,
                                direction=execution_direction,
                                quantity=execution_quantity,
                                execution_time_horizon=execution_time_horizon,
                                freq=execution_frequency,
                                trade=trade,
                                log_orders=True,
                                random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32)))
execution_agents = [twap_agent]

"""
vwap_agent = VWAPExecutionAgent(id=agent_count,
                                name='VWAPExecutionAgent',
                                type='ExecutionAgent',
                                symbol=symbol,
                                starting_cash=0,
                                direction=execution_direction,
                                quantity=execution_quantity,
                                execution_time_horizon=execution_time_horizon,
                                freq=execution_frequency,
                                volume_profile_path=None,
                                trade=trade,
                                log_orders=True,
                                random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32)))
execution_agents = [vwap_agent]

pov_agent = POVExecutionAgent(id=agent_count,
                              name='POVExecutionAgent',
                              type='ExecutionAgent',
                              symbol=symbol,
                              starting_cash=starting_cash,
                              start_time=execution_agent_start_time,
                              end_time=execution_agent_end_time,
                              freq=execution_frequency,
                              lookback_period=execution_frequency,
                              pov=0.1,
                              direction=execution_direction,
                              quantity=execution_quantity,
                              trade=trade,
                              log_orders=True,
                              random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32)))
execution_agents = [pov_agent]
"""

agents.extend(execution_agents)
agent_types.extend("ExecutionAgent")
agent_count += 1

########################################################################################################################
########################################### KERNEL AND OTHER CONFIG ####################################################

kernel = Kernel("Kernel", random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32)))

kernelStartTime = historical_date
kernelStopTime = mkt_close + pd.to_timedelta('00:01:00')

defaultComputationDelay = 50  # 50 nanoseconds

# LATENCY
latency_rstate = np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32))
pairwise = (agent_count, agent_count)

# All agents sit on line from Seattle to NYC
nyc_to_seattle_meters = 3866660
pairwise_distances = util.generate_uniform_random_pairwise_dist_on_line(0.0, nyc_to_seattle_meters, agent_count,
                                                                        random_state=latency_rstate)
pairwise_latencies = util.meters_to_light_ns(pairwise_distances)

model_args = {
    'connected': True,
    'min_latency': pairwise_latencies
}

latency_model = LatencyModel(latency_model='deterministic',
                             random_state=latency_rstate,
                             kwargs=model_args)
# KERNEL
kernel.runner(agents=agents,
              startTime=kernelStartTime,
              stopTime=kernelStopTime,
              agentLatencyModel=latency_model,
              defaultComputationDelay=defaultComputationDelay,
              oracle=oracle,
              log_dir=args.log_dir)

simulation_end_time = dt.datetime.now()
print("Simulation End Time: {}".format(simulation_end_time))
print("Time taken to run simulation: {}".format(simulation_end_time - simulation_start_time))