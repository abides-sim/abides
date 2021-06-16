import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time
import itertools
import pickle
import os.path

import sys
sys.path.extend(['../util/formatting'])
from convert_order_book import process_orderbook
from convert_order_stream import convert_stream_to_format
import itertools

import matplotlib.dates as mdates
from realism_utils import clip_times, make_orderbook_for_analysis, compute_impact_statistics, get_plot_colors, get_plot_linestyles, forward_fill_series, make_cache_and_visualisation_dir

from pprint import pprint
from bisect import bisect
import json
import argparse

import re


""" 
    Script plots the output for a single day market impact experiment.

"""


def make_plots(plot_data, plot_params_dict):
    """ Draw and save plot from preprocessed data.

    :param plot_data: Data structure processed by __name__.prep_data
    :param plot_params_dict: dict holding some plotting paramaters, see e.g. plot_configs/single_day/pov_single_day_config.example.json["PLOT_PARAMS_DICT"]
    :return:
    """
    fig, axes = plt.subplots(nrows=2, ncols=1)
    fig.set_size_inches(h=14, w=9)

    colors = list(get_plot_colors(plot_data + ['dummy']*2))
    color = colors.pop(0)
    linestyles = get_plot_linestyles(len(plot_data) + 2)
    linestyle = linestyles.pop(0)

    orderbook_df = plot_data[0]['no_execution_df']
    orderbook_df["MID_PRICE"].plot(ax=axes[0], label=plot_params_dict['baseline_label'], color=color, linestyle=linestyle)

    for plot_data_dict in plot_data:
        color = colors.pop(0)
        linestyle = linestyles.pop(0)
        pov = f'{100 * plot_data_dict["pov"]} %'
        orderbook_df = plot_data_dict['no_execution_df']
        orderbook_with_execution_df = plot_data_dict['yes_execution_df']

        # mid_price
        orderbook_with_execution_df["MID_PRICE"].plot(ax=axes[0], label=
            f'{plot_params_dict["execution_label"]}{pov}', color=color, linestyle=linestyle)

        # normalised difference
        mid_price_yes_execution, mid_price_no_execution = forward_fill_series(orderbook_with_execution_df["MID_PRICE"],
                                                                              orderbook_df["MID_PRICE"])

        diff = 10000 * (mid_price_yes_execution - mid_price_no_execution) / mid_price_no_execution

        diff = diff.to_frame()
        diff = diff.loc[~diff.index.duplicated(keep='last')]
        diff = diff[diff.columns[0]]  # to series for plotting

        diff.plot(ax=axes[1], label=f'{plot_params_dict["execution_label"]}{pov}', color=color, linestyle=linestyle)


    axes[0].axvspan(pd.Timestamp(plot_params_dict['shade_start_datetime']),
                    pd.Timestamp(plot_params_dict['shade_end_datetime']), alpha=0.2, color='grey')
    axes[1].axvspan(pd.Timestamp(plot_params_dict['shade_start_datetime']),
                    pd.Timestamp(plot_params_dict['shade_end_datetime']), alpha=0.2, color='grey')
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[-1].xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
    axes[0].xaxis.set_visible(False)
    axes[0].legend()

    axes[-1].set_xlabel('Time', size=15)
    axes[0].set_ylabel('Mid-price ($)', size=15)
    axes[1].set_ylabel('Normalized Difference (bps)', size=15)

    fig.tight_layout()
    fig.subplots_adjust(top=0.7)

    fig.savefig(plot_params_dict["output_file_path"], format='png', dpi=300, transparent=False, bbox_inches='tight',
                pad_inches=0.03)


def check_date_in_string(s):
    """ Check if date in format YYYY-MM-DD in a string. """
    m = re.search(r'(\d{4}-\d{2}-\d{2})', s)
    if m:
        return True
    else:
        return False


def prepare_shade_dates(start, end, historical_date):
    """ Helper method for prep_data. Formats date string in correct for compute_impact_statistics method"""

    if not check_date_in_string(start):
        shade_start_time = historical_date + pd.to_timedelta(start)
        shade_start_time = shade_start_time.strftime('%Y-%m-%d %H:%M:%S')
    else:
        shade_start_time = start

    if not check_date_in_string(end):
        shade_end_time = historical_date + pd.to_timedelta(end)
        shade_end_time = shade_end_time.strftime('%Y-%m-%d %H:%M:%S')
    else:
        shade_end_time = end

    return shade_start_time, shade_end_time


def prep_data(plot_data, cache_file, only_executed, clipped_start_time, clipped_end_time, plot_params_dict, compute_impact_stats=False):
    """ Prepares and caches POV market impact experiment output files for further aggregation and processing.

    :param plot_data: Data structure holding paths to relevant ABIDES output files, see e.g. plot_configs/single_day/pov_single_day_config.example.json["PLOT_DATA"]
    :param cache_file: Path to file where processed data will be cached.
    :param only_executed: Switch as to only include transacted volume as opposed to including limit orders.
    :param clipped_start_time: Starting time at which to clip data in format 'HH:MM:SS'
    :param clipped_end_time: Finishing time at which to clip data in format 'HH:MM:SS'
    :param compute_impact_statistics: Switch whether to compute impact statistics


    :type plot_data: list
    :type cache_file: str
    :type only_executed: bool
    :type clipped_start_time: str
    :type clipped_end_time: str
    :type compute_impact_stats: bool

    :return:
    """
    out_data = []

    for data_dict in plot_data:

        print(f"Processing data for POV {data_dict['participation_of_volume']}")
        abides_orderbook_df = make_orderbook_for_analysis(data_dict['no_execution_exchange_path'], data_dict['no_execution_orderbook_path'], num_levels=1)
        abides_execution_orderbook_df = make_orderbook_for_analysis(data_dict['yes_execution_exchange_path'], data_dict['yes_execution_orderbook_path'],
                                                                    num_levels=1)

        if only_executed:
            abides_orderbook_df = abides_orderbook_df.loc[abides_orderbook_df["TYPE"] == "ORDER_EXECUTED"]
            abides_execution_orderbook_df = abides_execution_orderbook_df.loc[
                abides_execution_orderbook_df["TYPE"] == "ORDER_EXECUTED"]

        historical_date = pd.Timestamp(abides_orderbook_df.index[0].date())
        start = historical_date + pd.to_timedelta(clipped_start_time)
        end = historical_date + pd.to_timedelta(clipped_end_time)

        shade_start_time, shade_end_time = prepare_shade_dates(plot_params_dict['shade_start_datetime'],
                                                               plot_params_dict['shade_end_datetime'],
                                                               historical_date)

        abides_orderbook_df = clip_times(abides_orderbook_df, start, end)
        abides_execution_orderbook_df = clip_times(abides_execution_orderbook_df, start, end)

        date_str = historical_date.strftime('%Y%m%d')
        pov = data_dict["participation_of_volume"]
        seed = data_dict["seed"]

        if compute_impact_stats:
            stats_dict = compute_impact_statistics(abides_orderbook_df, abides_execution_orderbook_df,
                                                   shade_start_time, shade_end_time,
                                                   date_str=date_str,
                                                   pov=pov,
                                                   seed=seed,
                                                   experiment_name=plot_params_dict['experiment_name'],
                                                   execution_agent_name=plot_params_dict['execution_agent_name'],
                                                   log_dir=plot_params_dict['log_dir'],
                                                   spread_lookback=plot_params_dict['spread_lookback']
                                                   )

            print(f"Statistics for participation of volume at level {100 * data_dict['participation_of_volume']}%")
            print("Statistics:")
            pprint(stats_dict)
            out_data.append({
                "no_execution_df": abides_orderbook_df,
                "yes_execution_df": abides_execution_orderbook_df,
                "impact_statistics": stats_dict,
                "pov": data_dict['participation_of_volume']
            })
        else:
            out_data.append({
                "no_execution_df": abides_orderbook_df,
                "yes_execution_df": abides_execution_orderbook_df,
                "pov": data_dict['participation_of_volume']
            })

    with open(cache_file, 'wb') as f:
        pickle.dump(out_data, f)

    return out_data


def main(config_path):
    """ Loads a plot config file for a single day POV market impact experiment and plots the result.

    :param config_path: 'Name of config file to execute. See plot_configs/single_day/pov_single_day_config.example.json for an example.'
    :type config_path: str

    :return:
    """

    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    # TODO: do this without global vars
    global ONLY_EXECUTED, CLIPPED_START_TIME, CLIPPED_FINISH_TIME, PLOT_DATA, CACHE_FILE, USE_CACHE, PLOT_PARAMS_DICT, \
        SPREAD_PLOT_LOG_SCALE, ORDERBOOK_IMBALANCE_PLOT_LOG_SCALE, COMPUTE_IMPACT_STATS

    ONLY_EXECUTED = config_dict["ONLY_EXECUTED"]
    CLIPPED_START_TIME = config_dict["CLIPPED_START_TIME"]
    CLIPPED_FINISH_TIME = config_dict["CLIPPED_FINISH_TIME"]
    PLOT_DATA = config_dict["PLOT_DATA"]
    CACHE_FILE = config_dict["CACHE_FILE"]
    USE_CACHE = config_dict["USE_CACHE"]
    PLOT_PARAMS_DICT = config_dict["PLOT_PARAMS_DICT"]
    SPREAD_PLOT_LOG_SCALE = config_dict["SPREAD_PLOT_LOG_SCALE"]
    ORDERBOOK_IMBALANCE_PLOT_LOG_SCALE = config_dict["ORDERBOOK_IMBALANCE_PLOT_LOG_SCALE"]
    COMPUTE_IMPACT_STATS = config_dict["COMPUTE_IMPACT_STATS"]

    if USE_CACHE and os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'rb') as f:
            print("Using cache...")
            out_data = pickle.load(f)
    else:
        out_data = prep_data(PLOT_DATA, CACHE_FILE, ONLY_EXECUTED, CLIPPED_START_TIME, CLIPPED_FINISH_TIME,
                             PLOT_PARAMS_DICT, compute_impact_stats=COMPUTE_IMPACT_STATS)

    print("Constructing plots...")

    make_plots(out_data, PLOT_PARAMS_DICT)
    print('Plotting complete!')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CLI utility for plotting results of single day POV experiments.')

    parser.add_argument('plot_config',
                        help='Name of config file to execute. See plot_configs/single_day/pov_single_day_config.example.json for an example.',
                        type=str)

    args, remaining_args = parser.parse_known_args()
    config_path = args.plot_config

    make_cache_and_visualisation_dir()

    main(config_path)






