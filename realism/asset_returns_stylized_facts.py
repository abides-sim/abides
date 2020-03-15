import sys
import os
import random
import pickle
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from realism_utils import get_trades
from glob import glob
from pathlib import Path
import argparse
sys.path.append('..')
from order_flow_stylized_facts import get_plot_colors
from util.formatting.convert_order_stream import dir_path

# Create cache folder if it does not exist
try: os.mkdir("cache")
except: pass

from metrics.minutely_returns import MinutelyReturns
from metrics.aggregation_normality import AggregationNormality
from metrics.autocorrelation import Autocorrelation
from metrics.volatility_clustering import VolatilityClustering
from metrics.kurtosis import Kurtosis
from metrics.volume_volatility_correlation import VolumeVolatilityCorrelation
from metrics.returns_volatility_correlation import ReturnsVolatilityCorrelation

all_metrics = [MinutelyReturns, AggregationNormality, Autocorrelation, VolatilityClustering, Kurtosis, VolumeVolatilityCorrelation, ReturnsVolatilityCorrelation]


def get_sims(sim_dir, n, my_metric):
    sims = []
    exchanges = [a for a in Path(sim_dir).rglob('*.bz2') if "exchange" in str(a).lower()]
    mult = math.ceil(n * samples_per_day / len(exchanges))

    for exchange in exchanges:
        ohlcv = get_trades(exchange)
        sims += my_metric.compute(ohlcv)

    sims = sims * mult  # Duplicate data to match size of historical data
    random.shuffle(sims)
    return sims


def plot_metrics(samples_per_day, real_dir, sim_dirs, sim_colors, recompute):
    # Loop through all metrics
    for my_metric in all_metrics:
        print(my_metric)
        my_metric = my_metric()

        # Calculate metrics for real data
        real_data = sorted(os.listdir(real_dir))
        n = len(real_data)
        pickled_real = "cache/{}_real.pickle".format(my_metric.__class__.__name__)

        # HISTORICAL METRIC
        if (not os.path.exists(pickled_real)) or recompute: # Pickled historical metric not found in cache.
            reals = []
            for f in real_data: # For each historical trading day...
                print(f)
                f = os.path.join(real_dir, f)
                df = pd.read_pickle(f, compression="bz2")
                df.reset_index(level=-1, inplace=True)
                df.level_1 = pd.to_datetime(df.level_1)
                symbols = df.index.unique().tolist()
                random.shuffle(symbols)
                symbols = symbols[:samples_per_day] # ...sample `samples_per_day` symbols.
                df = df[df.index.isin(symbols)] # Only keep data for sampled symbols.
                for sym in symbols:
                    select = df[df.index == sym].set_index("level_1") # Set timestamp as index.
                    vol = select["volume"]
                    select = select.drop("volume", axis=1)
                    select = (np.round((1000*select/select.iloc[0]).dropna())*100).astype("int")
                    select["volume"] = vol
                    reals += my_metric.compute(select) # Compute metric on sampled data.
            pickle.dump(reals, open(pickled_real, "wb"))
        else: # Pickled historical metric found in cache.
            reals = pickle.load(open(pickled_real, "rb"))

        # SIMULATED METRIC
        first = True if real_dir is None else False
        for i, sim_dir in enumerate(sim_dirs):
            # Calculate metrics for simulated data (via sampling)
            pickled_sim = "cache/{}_{}.pickle".format(my_metric.__class__.__name__, sim_dir.replace("/",""))
            if (not os.path.exists(pickled_sim)) or recompute: # Pickled simulated metric not found in cache.
                sims = get_sims(sim_dir, n, my_metric)
                sims = sims[:len(reals)] # Ensure length of simulated and historical data matches
                pickle.dump(sims, open(pickled_sim, "wb"))
            else: # Pickled simulated metric found in cache.
                sims = pickle.load(open(pickled_sim, "rb"))

            sim_name = sim_dir.rstrip('/').split("/")[-1]
            result = {(sim_name, sim_colors[i]): sims}

            # Create plot for each config and metric
            my_metric.visualize(result, reals, plot_real=not first)
            first = True

        plt.title(plt.gca().title.get_text())
        try: os.mkdir("visualizations")
        except: pass
        plt.savefig("visualizations/{}_{}.png".format(my_metric.__class__.__name__, sim_name),bbox_inches='tight')
        plt.clf()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Processes historical data and simulated stream files and produce plots'
                                                 ' of stylized fact metrics for asset return distributions.')
    parser.add_argument('-r', '--historical-data-dir', type=dir_path, required=False, help="Directory containing preprocessed"
                                                                                    "historical data for asset return"
                                                                                    "stylized facts")
    parser.add_argument('-s', '--simulated-data-dir', type=dir_path, action='append', required=False,
                        help="Directory containing .bz2 output log files from ABIDES Exchange Agent. Note that the "
                             "filenames MUST contain the word 'Exchange' in any case. One can add many simulated data "
                             "directories")
    parser.add_argument('-z', '--recompute', action="store_true", help="Rerun computations without caching.")


    args, remaining_args = parser.parse_known_args()

    # Settings
    samples_per_day = 30
    real_dir = args.historical_data_dir
    sim_dirs = args.simulated_data_dir
    sim_colors = get_plot_colors(sim_dirs, start_idx=1)

    plot_metrics(samples_per_day, real_dir, sim_dirs, sim_colors, args.recompute)
