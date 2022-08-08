# Test-1 :
# - 1     Exchange Agent
# - 1     Market Maker Agent
# - 30    ZI Agent (retail)
# - 70   HBL Agent (institutional)

import argparse
import numpy as np
import pandas as pd
import sys
import datetime as dt
import importlib
import copy 

from Kernel import Kernel   
from util import util
from util.order import LimitOrder
from util.oracle.SparseMeanRevertingOracle import SparseMeanRevertingOracle
from model.LatencyModel import LatencyModel

from agent.ExchangeAgent import ExchangeAgent
from agent.market_makers.MarketMakerAgent import MarketMakerAgent

from agent.RetailExecutionAgent import RetailExecutionAgent   
from agent.HeuristicBeliefLearningAgent import HeuristicBeliefLearningAgent
from agent.examples.MomentumAgent import MomentumAgent
from agent.NoiseAgent import NoiseAgent

########################################################################################################################
############################################### GENERAL CONFIG #########################################################

parser = argparse.ArgumentParser(
    description='initial test config')

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
                    default=1,
                    help='numpy.random.seed() for simulation')
parser.add_argument('-d',
                    '--historical_date',
                    default=pd.to_datetime('2020-01-01'),
                    help='date to test')
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
parser.add_argument('-t',
                    '--ticker',
                    default="IBM",
                    help='Stock Ticker')
parser.add_argument('-e',
                    '--experiment_length',
                    default='70:00:00', #TODO     test odd times that dont split evenly into days
                    help='Experiment length')               
parser.add_argument('-n',
                    '--noise',
                    default=1,
                    help='Include noise agents')        
parser.add_argument('-i',
                    '--iterations',
                    default=1,
                    help='Number of repeats of experiment')                             
        
args, remaining_args = parser.parse_known_args()
if args.config_help:
    parser.print_help()
    sys.exit()


 
log_dir = args.log_dir  # Requested log directory. # TODO: if it exists, overwrite
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
historical_date = args.historical_date # midnight on date
symbol = args.ticker

agent_count, agents, agent_types = 0, [], []


mm_cash = 50000000000 # $500,000,000

def random_retail_start_cash(retail_cash = 250000): # $2,500
    # Draws start cash from a normal distribution around retail_cash
    return int(np.random.normal(retail_cash, retail_cash * 0.1)) # TODO: improve?

def random_institution_start_cash(institution_cash = 5000000000): # $50,000,000
    # Draws start cash from a normal distribution around institution_cash
    return int(np.random.normal(institution_cash, institution_cash * 0.1)) # TODO: improve?

# Oracle
mkt_open = historical_date + pd.to_timedelta('9:00:00')
mkt_close = historical_date +  pd.to_timedelta('16:00:00')
day_length = mkt_close - mkt_open
days = pd.to_timedelta(args.experiment_length)/day_length

symbols = {symbol: {'r_bar': 1e4,   # base price of asset
                    'kappa': 1.67e-12,  
                    'agent_kappa': 1.67e-15,
                    'sigma_s': 0,
                    'fund_vol': 1e-8,
                    'megashock_lambda_a': 2.77778e-13,
                    'megashock_mean': 1e3,
                    'megashock_var': 5e4,
                    'random_state': np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 16,
                                                                                 dtype='uint64'))}}
oracle = SparseMeanRevertingOracle(mkt_open, mkt_close, symbols)

# 1) 1 Exchange Agent
stream_history_length = 25000

agents.extend([ExchangeAgent(id=0,
                             name="EXCHANGE_AGENT",
                             type="ExchangeAgent",
                             mkt_open=mkt_open,
                             mkt_close=mkt_close,
                             symbols=[symbol],
                             log_orders=False,
                             stream_history=stream_history_length,
                             pipeline_delay=0,
                             computation_delay=0,
                             wide_book=True,
                             book_freq=0,
                             days=days,
                             random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 16,
                                                                                       dtype='uint64')))])
agent_types.extend("ExchangeAgent")
agent_count += 1

# 2) 1 Market Maker Agent
num_mm_agents = 1
agents.extend([MarketMakerAgent(id=j,
                                name="MARKET_MAKER_AGENT_{}".format(j),
                                type='MarketMakerAgent',
                                symbol=symbol,
                                starting_cash=mm_cash,
                                min_size=1,
                                max_size=1000,
                                log_orders=False,
                                random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 16,
                                                                                          dtype='uint64')))
               for j in range(agent_count, agent_count + num_mm_agents)])

agent_types.extend('MarketMakerAgent')
agent_count += num_mm_agents

# 3) 30 retail Agents   -   30% of total agents

num_retail_agents = 30
agents.extend([RetailExecutionAgent(id=j,
                                     name="RETAIL_{}".format(j),
                                     type="RetailExecutionAgent",
                                     symbol=symbol,
                                     starting_cash=random_retail_start_cash(),
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
                                     execution=True,
                                     retail_delay=2000000000, # 2 second delay on messages
                                     random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 16,
                                                                                               dtype='uint64')))
               for j in range(agent_count, agent_count + num_retail_agents)])
agent_types.extend("RetailExecutionAgent")
agent_count += num_retail_agents

# 4) 70 Heuristic Belief Learning Agents    - smarter, represent institutions 
num_hbl_agents = 70
agents.extend([HeuristicBeliefLearningAgent(id=j,
                                            name="HBL_AGENT_{}".format(j),
                                            type="HeuristicBeliefLearningAgent",
                                            symbol=symbol,
                                            starting_cash=random_institution_start_cash(), 
                                            sigma_n=10000,      
                                            sigma_s=symbols[symbol]['fund_vol'],
                                            kappa=symbols[symbol]['agent_kappa'],
                                            r_bar=symbols[symbol]['r_bar'],
                                            q_max=10*100,               # willing to hold more positions than retail
                                            sigma_pv=5e4,
                                            R_min=0,
                                            R_max=100,
                                            eta=1,
                                            lambda_a=1e-12,
                                            L=2,
                                            log_orders=False,
                                            random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                                      dtype='uint64')))
               for j in range(agent_count, agent_count + num_hbl_agents)])
agent_types.extend("HeuristicBeliefLearningAgent")
agent_count += num_hbl_agents

# 5) Noise Agents
if bool(int(args.noise)):
    num_noise = 900
    agents.extend([NoiseAgent(id=j,
                            name="NoiseAgent {}".format(j),
                            type="NoiseAgent",
                            symbol=symbol,
                            wakeup_time=util.get_wake_time(mkt_open, mkt_close),
                            log_orders=False,
                            random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')))
                for j in range(agent_count, agent_count + num_noise)])
    agent_count += num_noise
    agent_types.extend(['NoiseAgent'])

########################################################################################################################
########################################### KERNEL AND OTHER CONFIG ####################################################

kernel = Kernel("Test1 Kernel", random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 16,
                                                                                                  dtype='uint64')))
offset = pd.to_timedelta(1, unit='h') # ensure kernel can complete shut down process after final mkt close
kernelStartTime = historical_date
kernelStopTime = mkt_close + pd.to_timedelta(days, unit='D') + offset
defaultComputationDelay = 50  # nanoseconds

# LATENCY

latency_rstate = np.random.RandomState(seed=np.random.randint(low=0, high=2**16))
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
for sim in range(int(args.iterations)):
    print("Simulation iteration {} starting".format(sim))
    agents1 = copy.deepcopy(agents)
    if log_dir is not None:
        log_dir = args.log_dir + '_{}'.format(sim + 1)
    kernel.runner(agents=agents1,
                startTime=kernelStartTime,
                stopTime=kernelStopTime,
                agentLatencyModel=latency_model,
                defaultComputationDelay=defaultComputationDelay,
                oracle=oracle,
                log_dir=log_dir)
    print("Simulation iteration {} ending".format(sim + 1))

simulation_end_time = dt.datetime.now()
print("Simulation End Time: {}".format(simulation_end_time))
print("Time taken to run simulation: {}".format(simulation_end_time - simulation_start_time))
