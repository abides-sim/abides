import logging as log
import numpy as np
import pandas as pd
import datetime as dt
import psutil
import sys
from sqlalchemy import create_engine

sys.path.append('../')

import optuna
from optuna.samplers import RandomSampler, TPESampler

from Kernel import Kernel
from util import util
from util.order import LimitOrder
from util.oracle.ExternalFileOracle import ExternalFileOracle

from agent.ExchangeAgent import ExchangeAgent

# (1) ZI Agents
from agent.NoiseAgent import NoiseAgent
from agent.ValueAgent import ValueAgent

# (2) Strategic Agents
from agent.market_makers.MarketMakerAgent import MarketMakerAgent
from agent.market_makers.POVMarketMakerAgent import POVMarketMakerAgent
from agent.examples.MomentumAgent import MomentumAgent

util.silent_mode = True
LimitOrder.silent_mode = True

ABS_FIXED_PROPS = {
    'security': 'ABS',
    'date': '20190628',
    'start_time': '09:30:00',
    'stop_time': '16:00:00',
    'fundamental_file_path': '/efs/data/get_real_data/mid_prices/ORDERBOOK_IBM_FREQ_ALL_20190628_mid_price.bz2',
    'book_freq': None
}

R_BAR = 100


def abides(agents, props, oracle, name):
    """ run an abides simulation using a list of agents. Note this implementation assumes zero-latency.

    :param agents: list of agents in the ABM simulation
    :param props: simulation-specific properties
    :param oracle: the data oracle for the simulation
    :param name: simulation name
    :return: agent_states (saved states for each agent at the end of the simulation)
    """
    simulation_start_time = dt.datetime.now()

    kernel = Kernel(name,
                    random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32)))

    agent_states = kernel.runner(agents=list(agents.values()),
                                 startTime=pd.Timestamp(f'{props["date"]} {props["start_time"]}'),
                                 stopTime=pd.Timestamp(f'{props["date"]} {props["stop_time"]}'),
                                 agentLatency=np.zeros((len(agents), len(agents))),
                                 latencyNoise=[0.0],
                                 defaultComputationDelay=0,
                                 defaultLatency=0,
                                 oracle=oracle,
                                 log_dir=f'calibration_log_{name}')

    simulation_end_time = dt.datetime.now()

    log.info(f"Time taken to run simulation {name}: {simulation_end_time - simulation_start_time}")

    return agent_states


def config(params):
    """ create the list of agents for the simulation

    :param params: abides config parameters
    :return: list of agents given a set of parameters
    """

    ABS_AGENT_PROPS = {

        'noise_agents': {'num': params['n_noise'],
                         'cash': params['cash_noise']},

        'value_agents': {'num': params['n_value'],
                         'cash': params['cash_value'],

                         'lambda_a': params['lambda_a'],
                         'r_bar': params['r_bar'],
                         'sigma_n': params['sigma_n'],
                         'sigma_s': params['sigma_s'],
                         'kappa': params['kappa']
                         },

        'market_maker_agent': {'num': params['n_market_makers'],
                               'cash': params['cash_market_makers'],
                               'freq': '60s',
                               'min_size': params['market_makers_min_size'],
                               'max_size': params['market_makers_max_size']
                               },

        'momentum_agents': {'num': params['n_momentum'],
                            'cash': params['cash_momentum'],
                            'freq': '60s',
                            'min_size': params['momentum_min_size'],
                            'max_size': params['momentum_max_size']
                            }
    }

    abs_agents = {
        0: ExchangeAgent(id=0, name="EXCHANGE_AGENT", type="ExchangeAgent",
                         symbols=[ABS_FIXED_PROPS['security']],
                         mkt_open=pd.to_datetime(f'{ABS_FIXED_PROPS["date"]} {ABS_FIXED_PROPS["start_time"]}'),
                         mkt_close=pd.to_datetime(f'{ABS_FIXED_PROPS["date"]} {ABS_FIXED_PROPS["stop_time"]}'),
                         pipeline_delay=0, computation_delay=0, book_freq=ABS_FIXED_PROPS['book_freq'],
                         random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32))),
    }

    noise_agent_ids = [i for i in range(len(abs_agents),
                                        len(abs_agents) + ABS_AGENT_PROPS['noise_agents']['num'])]
    for id in noise_agent_ids:
        abs_agents[id] = NoiseAgent(id=id, name=f"NOISE_AGENT_{id}", type="NoiseAgent",
                                    symbol=ABS_FIXED_PROPS['security'],
                                    starting_cash=ABS_AGENT_PROPS['noise_agents']['cash'],
                                    wakeup_time=util.get_wake_time(
                                        pd.to_datetime(f'{ABS_FIXED_PROPS["date"]} {ABS_FIXED_PROPS["start_time"]}'),
                                        pd.to_datetime(f'{ABS_FIXED_PROPS["date"]} {ABS_FIXED_PROPS["stop_time"]}')),
                                    random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32)))

    value_agent_ids = [i for i in range(len(abs_agents),
                                        len(abs_agents) + ABS_AGENT_PROPS['value_agents']['num'])]
    for id in value_agent_ids:
        abs_agents[id] = ValueAgent(id=id, name=f"VALUE_AGENT_{id}", type="ValueAgent",
                                    symbol=ABS_FIXED_PROPS['security'],
                                    starting_cash=ABS_AGENT_PROPS['value_agents']['cash'],
                                    r_bar=ABS_AGENT_PROPS['value_agents']['r_bar'],
                                    sigma_n=ABS_AGENT_PROPS['value_agents']['sigma_n'],
                                    sigma_s=ABS_AGENT_PROPS['value_agents']['sigma_s'],
                                    kappa=ABS_AGENT_PROPS['value_agents']['kappa'],
                                    lambda_a=ABS_AGENT_PROPS['value_agents']['lambda_a'],
                                    random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32)))

    market_maker_agent_ids = [i for i in range(len(abs_agents),
                                               len(abs_agents) + ABS_AGENT_PROPS['market_maker_agent']['num'])]
    for id in market_maker_agent_ids:
        abs_agents[id] = MarketMakerAgent(id=id, name=f"MARKET_MAKER_AGENT_{id}", type='MarketMakerAgent',
                                          symbol=ABS_FIXED_PROPS['security'],
                                          starting_cash=ABS_AGENT_PROPS['market_maker_agent']['cash'],
                                          min_size=ABS_AGENT_PROPS['market_maker_agent']['min_size'],
                                          max_size=ABS_AGENT_PROPS['market_maker_agent']['max_size'],
                                          wake_up_freq=ABS_AGENT_PROPS['market_maker_agent']['freq'],
                                          random_state=np.random.RandomState(
                                              seed=np.random.randint(low=0, high=2 ** 32)))

    momentum_agent_ids = [i for i in
                          range(len(abs_agents), len(abs_agents) + ABS_AGENT_PROPS['momentum_agents']['num'])]
    for id in momentum_agent_ids:
        abs_agents[id] = MomentumAgent(id=id, name=f"MOMENTUM_AGENT_{id}", type='MomentumAgent',
                                       symbol=ABS_FIXED_PROPS['security'],
                                       starting_cash=ABS_AGENT_PROPS['momentum_agents']['cash'],
                                       min_size=ABS_AGENT_PROPS['momentum_agents']['min_size'],
                                       max_size=ABS_AGENT_PROPS['momentum_agents']['max_size'],
                                       wake_up_freq=ABS_AGENT_PROPS['momentum_agents']['freq'],
                                       random_state=np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32)))

    return abs_agents


def objective(trial):
    """ The objective function to be optimized given parameters of the agent-based model

    :param trial: a single execution of the objective function
    :return: objective function
    """

    # define parameters of the model

    # 1) Noise Agents:
    n_noise = trial.suggest_int('n_noise', 1, 5000)
    cash_noise = 1e3  # trial.suggest_int('cash_noise', 1e3, 1e5)

    # 2) Value Agents:
    n_value = trial.suggest_int('n_value', 1, 100)
    cash_value = 1e4  # trial.suggest_int('cash_value', 1e3, 1e5)

    # true mean fundamental value
    r_bar = R_BAR  # trial.suggest_uniform('r_bar', 0.0, 100.0)

    # mean reversion parameter
    kappa = 1.67e-15  # trial.suggest_loguniform('kappa', 1e-15, 1e-5)

    # shock variance
    sigma_s = 1e5  # trial.suggest_uniform('sigma_s', 1e3, 1e5)

    # observation noise variance
    sigma_n = 1e4  # trial.suggest_uniform('sigma_n', 0.0, 1e4)

    # mean arrival rate of ZI agents
    lambda_a = 7e-11 # trial.suggest_loguniform('lambda_a', 1e-13, 1e-11)

    # 3) Market Maker Agents:
    n_market_makers = 1 # trial.suggest_int('n_market_makers', 1, 10)
    cash_market_makers = 1e5  # trial.suggest_int('cash_market_makers', 1e5, 1e7)
    market_makers_min_size = 100
    market_makers_max_size = 101

    # 4) Momentum Agents:
    n_momentum = 25 # trial.suggest_int('n_momentum', 1, 50)
    cash_momentum = 1e4  # trial.suggest_int('cash_momentum', 1e4, 1e6)
    momentum_min_size = 1
    momentum_max_size = 10

    params = {
        'n_noise': n_noise,
        'cash_noise': cash_noise,

        'n_value': n_value,
        'cash_value': cash_value,
        'r_bar': r_bar,
        'kappa': kappa,
        'sigma_s': sigma_s,
        'sigma_n': sigma_n,
        'lambda_a': lambda_a,

        'n_market_makers': n_market_makers,
        'cash_market_makers': cash_market_makers,
        'market_makers_min_size' : market_makers_min_size,
        'market_makers_max_size' : market_makers_max_size,

        'n_momentum': n_momentum,
        'cash_momentum': cash_momentum,
        'momentum_min_size' : momentum_min_size,
        'momentum_max_size' : momentum_max_size

    }

    # 2) get list of agents using params
    agents = config(params)

    # 3) define oracle (TO DO: remove in future iterations)
    oracle = ExternalFileOracle({ABS_FIXED_PROPS['security']: {
        'fundamental_file_path': ABS_FIXED_PROPS['fundamental_file_path'],
        'random_state': np.random.RandomState(seed=np.random.randint(low=0, high=2 ** 32))
    }})

    # 4) run abides with the set of agents and properties and observe the resulting agent states
    result = abides(agents=agents, props=ABS_FIXED_PROPS, oracle=oracle, name=trial.number)

    exchange, gains = result[0], result[1:]

    return sum(gains) # simple objective to maximize the gains of all agents


if __name__ == "__main__":

    start_time = dt.datetime.now()
    log.basicConfig(level=log.INFO)

    system_name = '  ABIDES: Calibration using Optuna framework'
    log.info('=' * len(system_name))
    log.info(system_name)
    log.info('=' * len(system_name))
    log.info(' ')

    study_name = 'abides_study'
    log.info(f'Study : {study_name}')

    seed = 1
    log.info(f'Seed: {seed}')
    np.random.seed(seed)

    hist_fund = pd.read_pickle(ABS_FIXED_PROPS['fundamental_file_path'])
    mkt_open = pd.Timestamp(f'{ABS_FIXED_PROPS["date"]} {ABS_FIXED_PROPS["start_time"]}')
    R_BAR = hist_fund.loc[hist_fund.index >= mkt_open].iloc[0]

    n_trials = 2
    n_jobs = 1 # psutil.cpu_count()
    log.info(f'Number of Trials : {n_trials}')
    log.info(f'Number of Parallel Jobs : {n_jobs}')

    # sampler = RandomSampler(seed=seed)
    sampler = TPESampler(seed=seed)  # Make the sampler behave in a deterministic way.

    # study: A study corresponds to an optimization task, i.e., a set of trials.
    study = optuna.create_study(study_name=study_name,
                                direction='maximize',
                                sampler=sampler,
                                pruner=optuna.pruners.MedianPruner(),
                                storage=f'sqlite:///{study_name}.db',
                                load_if_exists=True)
    study.optimize(objective,
                   n_trials=n_trials,
                   n_jobs=n_jobs,
                   show_progress_bar=True)

    log.info(f'Best Parameters: {study.best_params}')
    log.info(f'Best Value: {study.best_value}')

    df = study.trials_dataframe()

    df.to_pickle(f'{study_name}_df.bz2')

    end_time = dt.datetime.now()
    log.info(f'Total time taken for the study: {end_time - start_time}')
