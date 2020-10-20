# RMSC-2 (Reference Market Simulation Configuration) with data subscription mechanism:
# - 1     Exchange Agent
# - 1     Market Maker Agent
# - 50    ZI Agent
# - 25    HBL Agent
# - 24    Momentum Agent

import argparse
import numpy as np
import pandas as pd
import sys
import datetime as dt
import importlib

from Kernel import Kernel
from util import util
from util.order import LimitOrder
from util.oracle.SparseMeanRevertingOracle import SparseMeanRevertingOracle
from model.LatencyModel import LatencyModel

from agent.ExchangeAgent import ExchangeAgent
from agent.market_makers.MarketMakerAgent import MarketMakerAgent
from agent.examples.MomentumAgent import MomentumAgent
from agent.examples.SubscriptionAgent import SubscriptionAgent

from agent.ZeroIntelligenceAgent import ZeroIntelligenceAgent
from agent.HeuristicBeliefLearningAgent import HeuristicBeliefLearningAgent

########################################################################################################################
############################################### GENERAL CONFIG #########################################################

parser = argparse.ArgumentParser(
    description='Detailed options for RMSC-1 (Reference Market Simulation Configuration) config.')

parser.add_argument('-c',
                    '--config',
                    required=True,
                    help='Name of config file to execute')
parser.add_argument('-l',
                    '--log_dir',
                    default=None,
                    help='Log directory name (default: unix timestamp at program start)')
parser.add_argument('-s',
                    '--seed',
                    type=int,
                    default=None,
                    help='numpy.random.seed() for simulation')
parser.add_argument('-v',
                    '--verbose',
                    action='store_true',
                    help='Maximum verbosity!')
parser.add_argument('--config_help',
                    action='store_true',
                    help='Print argument options for this config file')
parser.add_argument('-a',
                    '--agent_name',
                    default=None,
                    help='Specify the agent to test with')

args, remaining_args = parser.parse_known_args()

if args.config_help:
    parser.print_help()
    sys.exit()

log_dir = args.log_dir  # Requested log directory.
seed = args.seed  # Random seed specification on the command line.
if not seed: seed = int(pd.Timestamp.now().timestamp() * 1000000) % (2 ** 32 - 1)
np.random.seed(seed)

util.silent_mode = not args.verbose
LimitOrder.silent_mode = not args.verbose

simulation_start_time = dt.datetime.now()
print("Simulation Start Time: {}".format(simulation_start_time))
print("Configuration seed: {}\n".format(seed))
########################################################################################################################
############################################### AGENTS CONFIG ##########################################################

# Historical date to simulate.
historical_date = pd.to_datetime('2019-06-28')
symbol = 'JPM'

agent_count, agents, agent_types = 0, [], []
starting_cash = 10000000  # Cash in this simulator is always in CENTS.

# 1) 1 Exchange Agent
mkt_open = historical_date + pd.to_timedelta('09:30:00')
mkt_close = historical_date + pd.to_timedelta('16:00:00')

agents.extend([ExchangeAgent(id=0,
                             name="EXCHANGE_AGENT",
                             type="ExchangeAgent",
                             mkt_open=mkt_open,
                             mkt_close=mkt_close,
                             symbols=[symbol],
                             log_orders=False,
                             pipeline_delay=0,
                             computation_delay=0,
                             stream_history=10,
                             book_freq=0,
                             random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                       dtype='uint64')))])
agent_types.extend("ExchangeAgent")
agent_count += 1

# 2) 1 Market Maker Agent
num_mm_agents = 1
agents.extend([MarketMakerAgent(id=j,
                                name="MARKET_MAKER_AGENT_{}".format(j),
                                type='MarketMakerAgent',
                                symbol=symbol,
                                starting_cash=starting_cash,
                                min_size=500,
                                max_size=1000,
                                subscribe=True,
                                log_orders=False,
                                random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                          dtype='uint64')))
               for j in range(agent_count, agent_count + num_mm_agents)])

agent_types.extend('MarketMakerAgent')
agent_count += num_mm_agents

# 3) 50 Zero Intelligence Agents
symbols = {symbol: {'r_bar': 1e5,
                    'kappa': 1.67e-12,
                    'agent_kappa': 1.67e-15,
                    'sigma_s': 0,
                    'fund_vol': 1e-8,
                    'megashock_lambda_a': 2.77778e-13,
                    'megashock_mean': 1e3,
                    'megashock_var': 5e4,
                    'random_state': np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                 dtype='uint64'))}}
oracle = SparseMeanRevertingOracle(mkt_open, mkt_close, symbols)

num_zi_agents = 50
agents.extend([ZeroIntelligenceAgent(id=j,
                                     name="ZI_AGENT_{}".format(j),
                                     type="ZeroIntelligenceAgent",
                                     symbol=symbol,
                                     starting_cash=starting_cash,
                                     sigma_n=10000,
                                     sigma_s=symbols[symbol]['fund_vol'],
                                     kappa=symbols[symbol]['agent_kappa'],
                                     r_bar=symbols[symbol]['r_bar'],
                                     q_max=10,
                                     sigma_pv=5e4,
                                     R_min=0,
                                     R_max=100,
                                     eta=1,
                                     lambda_a=1e-12,
                                     log_orders=False,
                                     random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                               dtype='uint64')))
               for j in range(agent_count, agent_count + num_zi_agents)])
agent_types.extend("ZeroIntelligenceAgent")
agent_count += num_zi_agents

# 4) 25 Heuristic Belief Learning Agents
num_hbl_agents = 25
agents.extend([HeuristicBeliefLearningAgent(id=j,
                                            name="HBL_AGENT_{}".format(j),
                                            type="HeuristicBeliefLearningAgent",
                                            symbol=symbol,
                                            starting_cash=starting_cash,
                                            sigma_n=10000,
                                            sigma_s=symbols[symbol]['fund_vol'],
                                            kappa=symbols[symbol]['agent_kappa'],
                                            r_bar=symbols[symbol]['r_bar'],
                                            q_max=10,
                                            sigma_pv=5e4,
                                            R_min=0,
                                            R_max=100,
                                            eta=1,
                                            lambda_a=1e-12,
                                            L=2,
                                            log_orders=False,
                                            random_state=np.random.RandomState(
                                                seed=np.random.randint(low=0, high=2 ** 32,
                                                                       dtype='uint64')))
               for j in range(agent_count, agent_count + num_hbl_agents)])
agent_types.extend("HeuristicBeliefLearningAgent")
agent_count += num_hbl_agents

# 5) 24 Momentum Agents:
num_momentum_agents = 24
agents.extend([MomentumAgent(id=j,
                             name="MOMENTUM_AGENT_{}".format(j),
                             type="MomentumAgent",
                             symbol=symbol,
                             starting_cash=starting_cash,
                             min_size=1,
                             max_size=10,
                             subscribe=True,
                             log_orders=False,
                             random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                       dtype='uint64')))
               for j in range(agent_count, agent_count + num_momentum_agents)])
agent_types.extend("MomentumAgent")
agent_count += num_momentum_agents

"""
# 6) 1 Example Subscription Agent:
agents.extend([SubscriptionAgent(id=agent_count,
                                 name="EXAMPLE_SUBSCRIPTION_AGENT",
                                 type="SubscriptionAgent",
                                 symbol=symbol,
                                 starting_cash=starting_cash,
                                 levels=5,
                                 freq=10e9,
                                 log_orders=False,
                                 random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                           dtype='uint64')))])
agent_types.extend("SubscriptionAgent")
agent_count += 1
"""
# 7) User defined agent
# Load the agent to evaluate against the market 
if args.agent_name:
    mod_name = args.agent_name.rsplit('.', 1)[0]
    class_name = args.agent_name.split('.')[-1]
    m = importlib.import_module(args.agent_name, package=None)
    testagent = getattr(m, class_name)

    agents.extend([testagent(id=agent_count,
                             name=args.agent_name,
                             type="AgentUnderTest",
                             symbol=symbol,
                             starting_cash=starting_cash,
                             min_size=1,
                             max_size=10,
                             log_orders=False,
                             random_state=np.random.RandomState(
                                 seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')))])
    agent_count += 1
    agent_types.extend('AgentUnderTest')

########################################################################################################################
########################################### KERNEL AND OTHER CONFIG ####################################################

kernel = Kernel("RMSC02 Kernel", random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                                  dtype='uint64')))

kernelStartTime = historical_date
kernelStopTime = mkt_close + pd.to_timedelta('00:01:00')

defaultComputationDelay = 50  # 50 nanoseconds

# LATENCY

latency_rstate = np.random.RandomState(seed=np.random.randint(low=0, high=2**32))
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
                             kwargs=model_args
                             )
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
