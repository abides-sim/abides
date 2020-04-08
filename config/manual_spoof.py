import argparse
import numpy as np
import pandas as pd
import sys
import datetime as dt
from dateutil.parser import parse

from Kernel import Kernel
from math import floor
from model.LatencyModel import LatencyModel
from util import util
from util.order import LimitOrder
from util.oracle.SparseMeanRevertingOracle import SparseMeanRevertingOracle

from agent.ExchangeAgent import ExchangeAgent
from agent.NoiseAgent import NoiseAgent
from agent.OrderBookImbalanceAgent import OrderBookImbalanceAgent
from agent.SpoofingAgent import SpoofingAgent
from agent.ValueAgent import ValueAgent
from agent.ZeroIntelligenceAgent import ZeroIntelligenceAgent
from agent.market_makers.MarketMakerAgent import MarketMakerAgent
from agent.examples.MomentumAgent import MomentumAgent

results = {}

########################################################################################################################
############################################### GENERAL CONFIG #########################################################
# Config 2 - AAMAS Paper

parser = argparse.ArgumentParser(description='Detailed options for random_fund_value config.')

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
parser.add_argument('-k', '--skip_log',
                   action='store_true',
                    help='Skip writing agent logs to disk')
parser.add_argument('-l',
                    '--log_dir',
                    default=None,
                    help='Log directory name (default: unix timestamp at program start)')
parser.add_argument('-n',
                    '--runs',
                    type=int,
                    default=1,
                    help='Number of runs across which to report returns.')
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

# Specialized for OBI.
parser.add_argument('-f', '--obi_freq', type=int, default=1e9 * 60,
                    help='OBI subscription frequency')
parser.add_argument('-r', '--flat_range', type=float, default=0.1,
                    help='OBI dead zone for staying flat')
parser.add_argument('-y', '--entry_threshold', type=float, default=0.17,
                    help='OBI imbalance to enter a position')
parser.add_argument('-a', '--trail_dist', type=float, default=0.085,
                    help='OBI trailing stop distance to exit positions')
parser.add_argument('-e', '--levels', type=int, default=10,
                    help='OBI order book levels to consider')

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

# Specialized for loop_obi.
obi_freq = args.obi_freq
flat_range = args.flat_range
entry_threshold = args.entry_threshold
trail_dist = args.trail_dist
levels = args.levels

log_orders = False
book_freq = None
#book_freq = 'S'

simulation_start_time = dt.datetime.now()
print("Simulation Start Time: {}".format(simulation_start_time))
print("Configuration seed: {}\n".format(seed))


for run in range(args.runs):

  ########################################################################################################################
  ############################################### AGENTS CONFIG ##########################################################
  
  # Historical date to simulate.
  historical_date = pd.to_datetime(args.historical_date)
  mkt_open = historical_date + pd.to_timedelta('09:30:00')
  #mkt_close = historical_date + pd.to_timedelta('16:00:00')
  mkt_close = historical_date + pd.to_timedelta('11:00:00')
  agent_count, agents, agent_types = 0, [], []
  
  # Hyperparameters
  symbol = args.ticker
  starting_cash = 10000000  # Cash in this simulator is always in CENTS.
  
  r_bar = 1e5
  #sigma_n = r_bar / 10
  sigma_n = 1e6
  kappa = 1.67e-12
  agent_kappa = 1.67e-15
  lambda_a = 1e-12
  
  # Oracle
  symbols = {symbol: {'r_bar': r_bar,
                      'kappa': kappa,
                      'agent_kappa': agent_kappa,
                      'sigma_s': 0,
                      'fund_vol': 1e-4,
                      'megashock_lambda_a': 2.77778e-13,
                      'megashock_mean': 1e3,
                      'megashock_var': 5e4,
                      'random_state': np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64'))}}
  
  oracle = SparseMeanRevertingOracle(mkt_open, mkt_close, symbols)
  
  # 1) Exchange Agent
  num_exch = 1
  exchange_ids = range(agent_count, agent_count + num_exch)
  agents.extend([ExchangeAgent(id=j,
                               name=f"EXCHANGE_AGENT {j}",
                               type="ExchangeAgent",
                               mkt_open=mkt_open,
                               mkt_close=mkt_close,
                               symbols=[symbol],
                               log_orders=True,
                               pipeline_delay=0,
                               computation_delay=0,
                               stream_history=10,
                               book_freq=book_freq,
                               random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')))
                 for j in range(agent_count, agent_count + num_exch)])
  agent_types.extend("ExchangeAgent")
  agent_count += 1
  
  # 2) Noise Agents (5000)
  #num_noise = 12500
  num_noise = 0
  agents.extend([NoiseAgent(id=j,
                            name="NoiseAgent {}".format(j),
                            type="NoiseAgent",
                            symbol=symbol,
                            starting_cash=starting_cash,
                            wakeup_time=util.get_wake_time(mkt_open, mkt_close),
                            log_orders=log_orders,
                            random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')))
                 for j in range(agent_count, agent_count + num_noise)])
  agent_count += num_noise
  agent_types.extend(['NoiseAgent'])
  
  # 2a) Zero Intelligence Agents (500)
  num_zi = 500
  zi_obs_noise = 1000000

  zi_strategy = [ (0, 250, 1), (0, 500, 1), (0, 1000, 0.8), (0, 1000, 1),
                  (0, 2000, 0.8), (250, 500, 0.8), (250, 500, 1) ]

  zi = [ floor(num_zi / len(zi_strategy)) ] * len(zi_strategy)

  i = 0
  while sum(zi) < num_zi:
    zi[i] += 1
    i += 1

  for i in range(len(zi_strategy)):
    x = zi_strategy[i]
    strat_name = "Type {} [{} <= R <= {}, eta={}]".format(i+1, x[0], x[1], x[2])
    agent_types.extend([ 'ZeroIntelligenceAgent' ])


  for n, x in zip(zi, zi_strategy):
    strat_name = "ZeroIntelligenceAgent"
    while n > 0:
      agents.append(ZeroIntelligenceAgent(id=agent_count,
	            name=f"ZI Agent {agent_count}",
		    type=strat_name,
		    log_orders=False,
		    symbol=symbol,
		    starting_cash=starting_cash,
		    sigma_n=sigma_n,
		    r_bar=r_bar,
		    kappa=agent_kappa,
		    sigma_s=1e-4,
		    q_max=10,
		    sigma_pv=5e6,
		    R_min=x[0],
		    R_max=x[1],
		    eta=x[2],
		    lambda_a=lambda_a,
		    random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64'))))
      agent_count += 1
      n -= 1

  # 3) Value Agents (100)
  num_value = 500
  agents.extend([ValueAgent(id=j,
                            name="Value Agent {}".format(j),
                            type="ValueAgent",
                            symbol=symbol,
                            starting_cash=starting_cash,
                            sigma_n=sigma_n,
                            r_bar=r_bar,
                            kappa=agent_kappa,
                            lambda_a=lambda_a,
                            random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')))
                 for j in range(agent_count, agent_count + num_value)])
  agent_count += num_value
  agent_types.extend(['ValueAgent'])
  
  # 4) Market Maker Agent (1)
  num_mm_agents = 0
  agents.extend([MarketMakerAgent(id=j,
                                  name="MARKET_MAKER_AGENT_{}".format(j),
                                  type='MarketMakerAgent',
                                  symbol=symbol,
                                  starting_cash=starting_cash,
                                  min_size=100,
                                  max_size=101,
                                  wake_up_freq="1min",
                                  log_orders=log_orders,
                                  random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                            dtype='uint64')))
                 for j in range(agent_count, agent_count + num_mm_agents)])
  agent_count += num_mm_agents
  agent_types.extend('MarketMakerAgent')
  
  
  # 5) Momentum Agents (25)
  num_momentum_agents = 0
  agents.extend([MomentumAgent(id=j,
                               name="MOMENTUM_AGENT_{}".format(j),
                               type="MomentumAgent",
                               symbol=symbol,
                               starting_cash=starting_cash,
                               min_size=1,
                               max_size=10,
                               log_orders=log_orders,
                               random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                         dtype='uint64')))
                 for j in range(agent_count, agent_count + num_momentum_agents)])
  agent_count += num_momentum_agents
  agent_types.extend("MomentumAgent")
  
  # 6) Spoofing Agents
  num_spoofing = 0
  agents.extend([SpoofingAgent(id=j,
                            name="Spoofing Agent {}".format(j),
                            type="SpoofingAgent",
                            symbol=symbol,
                            starting_cash=starting_cash,
                            random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')))
                 for j in range(agent_count, agent_count + num_spoofing)])
  agent_count += num_spoofing
  agent_types.extend(['SpoofingAgent'])
  
  # 7) OBI Agents
  num_obi = 10
  obi_ids = range(agent_count, agent_count + num_obi)
  agent_count += num_obi

  # Latency must be set up before OBI agents, because each one must know its latency
  # for experimental reporting (not strategy) purposes.

  # 1-D experiment
  #min_latency = np.random.uniform(low = 1000000, high = 13000000, size = agent_count)
  #for idx, i in enumerate(obi_ids):
    #min_latency[i] = 21000 + (idx * 5000)
  #for i in exchange_ids:
    #min_latency[i] = 0

  pairwise = (agent_count, agent_count)

  min_latency = np.random.uniform(low = 1000000, high = 13000000, size = pairwise)

  # Mirror the matrix (using the lower triangle as the reference).
  min_latency = np.tril(min_latency) + np.triu(min_latency.T, 1)

  # Make some agents special (in terms of network connectivity).
  for idx, i in enumerate(obi_ids):
    for j in exchange_ids:
      #lat = 21000 + (idx * 5000)
      lat = 21000 + (idx * 21000)
    
      min_latency[i,j] = lat
      min_latency[j,i] = lat

  # Now create the OBI agents.
  agents.extend([OrderBookImbalanceAgent(id=j,
                                         name="OBI Agent {}".format(j),
                                         type="OrderBookImbalanceAgent",
                                         symbol = symbol,
                                         starting_cash = starting_cash,
                                         levels = levels,
                                         entry_threshold = entry_threshold,
                                         trail_dist = trail_dist,
                                         freq = obi_freq,
                                         latency = min_latency[j,0],
                                         strat_start = mkt_open + pd.to_timedelta('00:30:00'),
                                         unique_name = f"{log_dir}_{run}",
                                         random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')))
                 for j in obi_ids ])
  agent_types.extend(['OrderBookImbalanceAgent'])
  
  ########################################################################################################################
  ########################################### KERNEL AND OTHER CONFIG ####################################################
  
  kernel = Kernel("Market Replay Kernel", random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32,
                                                                                                    dtype='uint64')))
  
  kernelStartTime = historical_date
  kernelStopTime = historical_date + pd.to_timedelta('16:01:00')
  
  defaultComputationDelay = 1


  # Instantiate the latency model.
  model_args = { 'connected'   : True,
                 'min_latency' : min_latency,
                 #'jitter'      : 0.3,
                 #'jitter_clip' : 0.05,
                 #'jitter_unit' : 5,
                 'jitter'      : 0.5,
                 'jitter_clip' : 0.4,
                 'jitter_unit' : 10,
               }

  latency_model = LatencyModel ( latency_model = 'cubic',
                                 random_state = np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')),
                                 kwargs = model_args )


  
  saved_states, new_results = kernel.runner(agents=agents,
                startTime=kernelStartTime,
                stopTime=kernelStopTime,
                agentLatencyModel=latency_model,
                defaultComputationDelay=defaultComputationDelay,
                defaultLatency=0,
                oracle=oracle,
                log_dir=args.log_dir,
                skip_log=args.skip_log)
  
  for t in new_results:
    if t in results: results[t] += new_results[t]
    else: results[t] = new_results[t]
  

print (f"\nFinal mean results across {args.runs} runs.\n")

for t in sorted(results):
  print (f"{t}: {int(round(results[t] / args.runs))}")

print ()

simulation_end_time = dt.datetime.now()
print("Simulation End Time: {}".format(simulation_end_time))
print("Time taken to run simulation: {}".format(simulation_end_time - simulation_start_time))
