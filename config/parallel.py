import argparse
import os
from multiprocessing import Pool
import psutil
import datetime as dt
import numpy as np


def run_in_parallel(num_simulations, num_parallel, config, log_folder, verbose):

    global_seeds = np.random.randint(0, 2 ** 32, num_simulations)
    print(f'Global Seeds: {global_seeds}')

    processes = [f'python -u abides.py -c {config} -l {log_folder}_seed_{seed} {"-v" if verbose else ""} -s {seed}'
                 for seed in global_seeds]

    pool = Pool(processes=num_parallel)
    pool.map(run_process, processes)


def run_process(process):
    os.system(process)


if __name__ == "__main__":
    start_time = dt.datetime.now()

    parser = argparse.ArgumentParser(description='Main config to run multiple ABIDES simulations in parallel')
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed controlling the generated global seeds')
    parser.add_argument('--num_simulations', type=int, default=1,
                        help='Total number of simulations to run')
    parser.add_argument('--num_parallel', type=int, default=None,
                        help='Number of simulations to run in parallel')
    parser.add_argument('--config', required=True,
                        help='Name of config file to execute')
    parser.add_argument('--log_folder', required=True,
                        help='Log directory name')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Maximum verbosity!')

    args, remaining_args = parser.parse_known_args()

    seed = args.seed
    num_simulations = args.num_simulations
    num_parallel = args.num_parallel if args.num_parallel else psutil.cpu_count() # count of the CPUs on the machine
    config = args.config
    log_folder = args.log_folder
    verbose = args.verbose

    print(f'Total number of simulation: {num_simulations}')
    print(f'Number of simulations to run in parallel: {num_parallel}')
    print(f'Configuration: {config}')

    np.random.seed(seed)

    run_in_parallel(num_simulations=num_simulations,
                    num_parallel=num_parallel,
                    config=config,
                    log_folder=log_folder,
                    verbose=verbose)

    end_time = dt.datetime.now()
    print(f'Total time taken to run in parallel: {end_time - start_time}')