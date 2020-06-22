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
p = str(Path(__file__).resolve().parents[1])  # directory one level up from this file
sys.path.append(p)
from realism_utils import get_plot_colors
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


def get_sims(sim_dir, my_metric):
    sims = []
    exchanges = [a for a in Path(sim_dir).rglob('*.bz2') if "exchange" in str(a).lower()]

    for exchange in exchanges:
        ohlcv = get_trades(exchange)
        if ohlcv is not None:
            sims += my_metric.compute(ohlcv)

    random.shuffle(sims)

    return sims


def plot_metrics(sim_dirs, sim_colors, output_dir, recompute):
    # Loop through all metrics
    for my_metric in all_metrics:
        print(my_metric)
        my_metric = my_metric()

        for i, sim_dir in enumerate(sim_dirs):
            # Calculate metrics for simulated data (via sampling)
            pickled_sim = "cache/{}_{}.pickle".format(my_metric.__class__.__name__, sim_dir.replace("/",""))
            if (not os.path.exists(pickled_sim)) or recompute: # Pickled simulated metric not found in cache.
                sims = get_sims(sim_dir, my_metric)
                pickle.dump(sims, open(pickled_sim, "wb"))
            else: # Pickled simulated metric found in cache.
                sims = pickle.load(open(pickled_sim, "rb"))

            sim_name = sim_dir.rstrip('/').split("/")[-1]
            result = {(sim_name, sim_colors[i]): sims}

            # Create plot for each config and metric
            my_metric.visualize(result)

        plt.title(plt.gca().title.get_text())
        try: os.mkdir(output_dir)
        except: pass
        plt.savefig("{}/{}.png".format(output_dir, my_metric.__class__.__name__), bbox_inches='tight')
        plt.clf()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Processes historical data and simulated stream files and produce plots'
                                                 ' of stylized fact metrics for asset return distributions.')
    parser.add_argument('-s', '--simulated-data-dir', type=dir_path, action='append', required=True,
                        help="Directory containing .bz2 output log files from ABIDES Exchange Agent. Note that the "
                             "filenames MUST contain the word 'Exchange' in any case. One can add many simulated data "
                             "directories")
    parser.add_argument('-z', '--recompute', action="store_true", help="Rerun computations without caching.")
    parser.add_argument('-o', '--output-dir', default='visualizations', help='Path to output directory', type=dir_path)

    args, remaining_args = parser.parse_known_args()

    # Settings
    sim_dirs = args.simulated_data_dir
    sim_dirs.sort()
    sim_colors = get_plot_colors(sim_dirs)

    plot_metrics(sim_dirs, sim_colors, args.output_dir, args.recompute)
