import pandas as pd
import pickle
import glob as glob
import re
from pprint import pprint
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from impact_single_day_pov import prep_data
from realism_utils import forward_fill_series, make_cache_and_visualisation_dir
from multiprocessing import Pool

from functools import reduce

import json
import argparse
from pandas.plotting import register_matplotlib_converters

RESAMPLE_RATE = '0.5S'  # times at which to sample mid price series


def normalise_time(ts, base_date='2016-01-30'):
    """ Force a pandas timestamp to be on a certain date.

        :param ts: timestamp
        :param base_date: string in format YYYY-MM-DD

        :type ts: pd.Timestamp
        :type base_date: str
    """
    pd_base_date = pd.Timestamp(base_date)
    time = ts.time()
    td = pd.to_timedelta(str(time))
    return pd_base_date + td


def extract_seed_date_from_path(p, yes_base, no_base):
    """ Extracts random seed and date from path of ABIDES output file

    :param p: path from which to extract seed and date
    :param yes_base: regex pattern for output files WITH execution agent. Must obey the following:
        - contains the string `yes` and not contain the string `no`
        - pattern must contain two capture groups, where the first captures seed, second captures POV, and third
        captures date.
        If the above are not satisfied you will have to rename experimental output files.
    :param no_base: regex pattern for output files WITHOUT execution agent. Must obey the following:
        - contains the string `no` and not contain the string `yes`
        - pattern must contain two capture groups, where the first captures seed, second captures date.
        If the above are not satisfied you will have to rename experimental output files.
    :return:
    """
    base = yes_base if 'yes' in p else no_base
    yes_no = 'yes' if 'yes' in p else 'no'

    m = re.search(base, p)

    if yes_no == 'yes':
        seed = m.group(1)
        pov = m.group(2)
        date = m.group(3)
        out_dict = {
            'type': yes_no,
            'seed': seed,
            'pov': pov,
            'date': date
        }
        return out_dict
    elif yes_no == 'no':
        seed = m.group(1)
        date = m.group(2)
        out_dict = {
            'type': yes_no,
            'seed': seed,
            'date': date
        }
        return out_dict
    else:
        raise ValueError(f'Path {p} not in required format')


def generate_plot_data_cache_dicts(log_dirs_no_glob, log_dirs_yes_glob, yes_base, no_base):
    """ Generates data structure containing information about ABIDES output paths and experimental parameters, for
        downstream processing

    :param log_dirs_no_glob: Glob pattern listing directories corresponding to experiments WITHOUT execution agent.
    :param log_dirs_yes_glob: Glob pattern listing directories corresponding to experiments WITH execution agent.
    :param yes_base: regex pattern for output files WITH execution agent. Must obey the following:
        - contains the string `yes` and not contain the string `no`
        - pattern must contain two capture groups, where the first captures seed, second captures POV, and third
        captures date.
        If the above are not satisfied you will have to rename experimental output files.
    :param no_base: regex pattern for output files WITHOUT execution agent. Must obey the following:
        - contains the string `no` and not contain the string `yes`
        - pattern must contain two capture groups, where the first captures seed, second captures date.
        If the above are not satisfied you will have to rename experimental output files.
    :return:
    """
    log_dirs_no = glob.glob(log_dirs_no_glob)
    log_dirs_yes = glob.glob(log_dirs_yes_glob)

    log_dirs_no.sort()
    log_dirs_yes.sort()

    # match nos with same seed and date to yeses with same seed and date
    no_yes_dict = dict()

    for dn in log_dirs_no:
        info_dict_no = extract_seed_date_from_path(dn, yes_base, no_base)
        no_seed = info_dict_no['seed']
        no_date = info_dict_no['date']

        no_yes_dict[dn] = {
            'seed': no_seed,
            'date': no_date,
            'execution_paths': []
        }

        for dy in log_dirs_yes:
            info_dict_yes = extract_seed_date_from_path(dy, yes_base, no_base)
            yes_seed = info_dict_yes['seed']
            yes_date = info_dict_yes['date']
            pov = info_dict_yes['pov']
            if yes_seed == no_seed and yes_date == no_date:
                no_yes_dict[dn]['execution_paths'].append({
                    'path': dy,
                    'pov': pov
                })

    out_list = []

    for dn, dn_info_dict in no_yes_dict.items():

        seed = dn_info_dict['seed']
        date = dn_info_dict['date']
        PLOT_DATA = []

        for execution_d in dn_info_dict['execution_paths']:
            pov = execution_d['pov']
            execution_path = execution_d['path']
            plot_data_dict = {
                'participation_of_volume': pov,
                'no_execution_exchange_path': f'{dn}/{NO_EXECUTION_EXCHANGE_NAME}',
                'no_execution_orderbook_path': f'{dn}/{NO_EXECUTION_ORDERBOOK_NAME}',
                'yes_execution_exchange_path': f'{execution_path}/{YES_EXECUTION_EXCHANGE_NAME}',
                'yes_execution_orderbook_path': f'{execution_path}/{YES_EXECUTION_ORDERBOOK_NAME}',
                'seed': seed
            }

            PLOT_DATA.append(plot_data_dict)

        CACHE_DIR = f'cache/{CACHE_PREFIX}_{seed}_{date}'

        out_tuple = (PLOT_DATA, CACHE_DIR)
        out_list.append(out_tuple)

    return out_list


def process_tuple(tup):
    """ Helper method for __name__.make_cached. """
    PLOT_DATA, CACHE_DIR = tup
    PLOT_PARAMS_DICT = {
        'shade_start_datetime': SHADE_START_TIME,
        'shade_end_datetime': SHADE_END_TIME,
        'spread_lookback': SPREAD_LOOKBACK,
        'experiment_name': EXPERIMENT_NAME,
        'log_dir': LOG_DIR,
        'execution_agent_name': EXECUTION_AGENT_NAME
    }
    prep_data(PLOT_DATA, CACHE_DIR, ONLY_EXECUTED, CLIPPED_START_TIME, CLIPPED_FINISH_TIME, PLOT_PARAMS_DICT,
              compute_impact_stats=COMPUTE_IMPACT_STATS)


def get_differences(tup):
    """ Computes mid price differential for execution experiment.

    :param tup: tuple with elements:
        - [0] dictionary holding data to be processed.
        - [1] counter for which set of parameters are being used
    :return:
    """
    dict_in, count = tup
    orderbook_df = dict_in['no_execution_df']
    orderbook_with_execution_df = dict_in['yes_execution_df']

    impact_statistics = dict_in['impact_statistics'] if COMPUTE_IMPACT_STATS else None

    pov = dict_in['pov']

    print(f"Processing df pair {count+1}")

    mid_price_yes_execution, mid_price_no_execution = forward_fill_series(orderbook_with_execution_df["MID_PRICE"],
                                                                          orderbook_df["MID_PRICE"])

    mid_price_diff = 10000 * (mid_price_yes_execution - mid_price_no_execution) / mid_price_no_execution

    out_df = mid_price_diff.to_frame()
    out_df = out_df.loc[~out_df.index.duplicated(keep='last')]
    out_df = out_df.rename(columns={
        0: f'MID_PRICE_{count}',
    })

    # make date the same
    out_df.index = pd.DatetimeIndex(out_df.index.to_series().apply(normalise_time))
    print(f"Finished processing df pair {count+1}")
    out_dict = {
        'out_df': out_df,
        'impact_statistics': impact_statistics,
        'pov': pov
    }

    return out_dict


def concat_horizontal(df, s, resample_rate=RESAMPLE_RATE):
    """ Concats a pd.Series object to a pd.DataFrame horizontally. They need to each have a DatetimeIndex"""

    df_out = pd.merge(df.resample(resample_rate).last(), s.resample(resample_rate).last(), how='outer', left_index=True, right_index=True)
    return df_out


def aggregate_orderbook_stats(saved_orderbooks, pov, cache_file_suffix='', num_workers=24):
    """ Compute quantiles for mid-price and liquidity measures.

        :param saved_orderbooks: output of __name__.process_orderbooks_for_liquidity_plots
        :param num_workers: How many CPU cores to use for executing this function

        :type saved_orderbooks: list(tuple(pd.DataFrame))
        :type num_workers: int

    """

    print(f"Aggregating data for POV {pov}...")
    with Pool(num_workers) as p:
        tuple_list = [(v, idx) for idx, v in enumerate(saved_orderbooks)]
        stat_dicts = p.map(get_differences, tuple_list)

    stat_dfs = [s['out_df'] for s in stat_dicts]
    print("Merging DataFrames...")

    df_final = reduce(lambda left, right: concat_horizontal(left, right), stat_dfs)
    df_final = df_final.ffill()
    df_final = df_final.dropna()
    df_final = df_final.loc[~df_final.index.duplicated(keep='last')]

    print("Computing summary statistics...")
    mid_price_cols = [f'MID_PRICE_{x}' for x in range(len(saved_orderbooks))]
    mid_price_median = df_final[mid_price_cols].median(axis=1)
    mid_price_05_q = df_final[mid_price_cols].quantile(q=0.05, axis=1)
    mid_price_25_q = df_final[mid_price_cols].quantile(q=0.25, axis=1)
    mid_price_75_q = df_final[mid_price_cols].quantile(q=0.75, axis=1)
    mid_price_95_q = df_final[mid_price_cols].quantile(q=0.95, axis=1)

    df_data = {
        'mid_price_median': mid_price_median,
        'mid_price_05_q': mid_price_05_q,
        'mid_price_25_q': mid_price_25_q,
        'mid_price_75_q': mid_price_75_q,
        'mid_price_95_q': mid_price_95_q,
    }

    df_out = pd.DataFrame(data=df_data, index=df_final.index)

    with open(f'cache/aggregated_execution_pov_{pov}_{cache_file_suffix}.pkl', 'wb') as f:
        pickle.dump(df_out, f)

    print('Done!')
    return df_out


def plot_aggregated(aggregated, plot_params_dict):
    """ Make aggregated mid-price and liquidity plots.

        :param aggregated: output of __name__.aggregate_orderbook_stats
        :param plot_params_dict: dictionary containing some plotting parameters

        :type aggregated: pd.DataFrame
        :type plot_params_dict: dict

    """

    fig, axes = plt.subplots(nrows=1, ncols=1)
    fig.set_size_inches(h=9, w=12)

    # mid_price
    aggregated["mid_price_median"].plot(ax=axes, color='blue', label='Median')
    axes.axhline(y=0, color='black', linestyle='--', linewidth=0.7, zorder=1)
    axes.fill_between(x=aggregated.index, y1=aggregated["mid_price_median"], y2=aggregated['mid_price_95_q'],
                      color='blue', alpha=plot_params_dict['alpha_90'], label='90%')
    axes.fill_between(x=aggregated.index, y2=aggregated["mid_price_median"], y1=aggregated['mid_price_05_q'],
                      color='blue', alpha=plot_params_dict['alpha_90'])
    axes.fill_between(x=aggregated.index, y2=aggregated["mid_price_median"], y1=aggregated['mid_price_75_q'],
                      color='blue', alpha=plot_params_dict['alpha_50'], label='50%')
    axes.fill_between(x=aggregated.index, y2=aggregated["mid_price_median"], y1=aggregated['mid_price_25_q'],
                      color='blue', alpha=plot_params_dict['alpha_50'])

    date = aggregated.index[0].date()
    midnight = pd.Timestamp(date)
    xmin = midnight + pd.to_timedelta(plot_params_dict['xmin'])
    xmax = midnight + pd.to_timedelta(plot_params_dict['xmax'])

    axes.set_ylim(plot_params_dict['ymin'], plot_params_dict['ymax'])
    axes.set_xlim(xmin, xmax)

    shade_start = midnight + pd.to_timedelta(plot_params_dict['shade_start_time'])
    shade_end = midnight + pd.to_timedelta(plot_params_dict['shade_end_time'])
    axes.axvspan(shade_start, shade_end, alpha=0.2, color='grey')

    axes.set_xlabel('Time', size=18)

    axes.set_ylabel('Mid-price normalized\n difference (bps)', size=18)

    # axes.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    # axes.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
    axes.legend(loc='upper right', fontsize=18)
    axes.tick_params(axis='both', which='major', labelsize=16)
    axes.tick_params(axis='both', which='minor', labelsize=16)

    fig.suptitle(plot_params_dict['execution_label'], fontsize=20, fontweight='bold')
    fig.tight_layout()
    fig.subplots_adjust(top=0.94)

    fig.savefig(plot_params_dict['output_file_path'], format='png', dpi=300, transparent=False, bbox_inches='tight',
                pad_inches=0.03)


def make_cached(params, num_workers):
    """ Process ABIDES output data into format suitable for aggregation and cache.

    :param params: Data structure constructed by __name__.generate_plot_data_cache_dicts
    :param num_workers: Number of CPU cores to use
    :return:
    """
    with Pool(num_workers) as p:
        p.map(process_tuple, params)


def load_cached():
    """ Loads cached execution experiment data from __name__.make_cached into memory

    :return:
    """
    data_to_process = []

    for path in glob.glob(f'cache/{CACHE_PREFIX}_*_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]'):
        with open(path, 'rb') as f:
            try:
                data = pickle.load(f)
                data_to_process.append(data)
            except pickle.UnpicklingError:
                continue

    return data_to_process


def aggregate_data(data_to_process, num_workers):
    """ Aggregates cached experimental data.

    :param data_to_process: output of __name__.load_cached
    :return:
    """
    aggregated_data = dict()

    for pov in POVs:
        data_pov = []
        for data in data_to_process:
            for dd in data:
                if dd['pov'] == str(pov):
                    data_pov.append(dd)

        aggregated = aggregate_orderbook_stats(data_pov, pov, cache_file_suffix=CACHE_PREFIX, num_workers=num_workers)
        aggregated_data.update({
            pov: aggregated
        })

    with open(f'cache/{CACHE_PREFIX}_pov_multiday_agg.pkl', 'wb') as f:
        pickle.dump(aggregated_data, f)

    return aggregated_data


def plot_all_aggregated(aggregated_data, params):
    """ Draw plots of aggregated data.

    :param aggregated_data: output of __name__.aggregate_data
    :param params: output of __name__.generate_plot_data_cache_dicts
    :return:
    """
    for pov in POVs:
        aggregated = aggregated_data[pov]

        PLOT_PARAMS_DICT = {
            'baseline_label': BASELINE_LABEL,
            'execution_label': f'POV {DIRECTION} order @ {100 * pov} %, freq. {FREQUENCY}, {len(params)} traces',
            'shade_start_time': SHADE_START_TIME,
            'shade_end_time': SHADE_END_TIME,
            'ymax': YMAX,
            'ymin': YMIN,
            'xmin': XMIN,
            'xmax': XMAX,
            'alpha_90': ALPHA_90,
            'alpha_50': ALPHA_50,
            'output_file_path': f'visualizations/{CACHE_PREFIX}_pov_{pov}_multiday.png'
        }

        plot_aggregated(aggregated, PLOT_PARAMS_DICT)


def main(config_path, num_workers, recompute):
    """ Load config file for multiday POV market impact experiment and draws plots, doing necessary data processing if necessary.

    :param config_path: path to config file, see e.g. plot_configs/pov_plot_config.example.json
    :param num_workers: number of CPU cores to use during processing
    :param recompute: switch as to whether to repeat whole data processing cycle

    :type config_path: str
    :type num_workers: int
    :type recompute: bool

    :return:
    """
    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    # TODO: do this without global vars
    global CLIPPED_START_TIME, CLIPPED_FINISH_TIME, ONLY_EXECUTED, NO_EXECUTION_EXCHANGE_NAME, \
        NO_EXECUTION_ORDERBOOK_NAME, YES_EXECUTION_EXCHANGE_NAME, YES_EXECUTION_ORDERBOOK_NAME, \
        CACHE_PREFIX, POVs, FREQUENCY, DIRECTION, BASELINE_LABEL, SHADE_START_TIME, SHADE_END_TIME, \
        YMAX, YMIN, XMIN, XMAX, ALPHA_90, ALPHA_50, yes_base, no_base, log_dirs_no_glob, log_dirs_yes_glob, \
        SPREAD_LOOKBACK, EXPERIMENT_NAME, LOG_DIR, EXECUTION_AGENT_NAME, COMPUTE_IMPACT_STATS

    CLIPPED_START_TIME = config_dict["CLIPPED_START_TIME"]
    CLIPPED_FINISH_TIME = config_dict["CLIPPED_FINISH_TIME"]
    ONLY_EXECUTED = config_dict["ONLY_EXECUTED"]
    NO_EXECUTION_EXCHANGE_NAME = config_dict["NO_EXECUTION_EXCHANGE_NAME"]
    NO_EXECUTION_ORDERBOOK_NAME = config_dict["NO_EXECUTION_ORDERBOOK_NAME"]
    YES_EXECUTION_EXCHANGE_NAME = config_dict["YES_EXECUTION_EXCHANGE_NAME"]
    YES_EXECUTION_ORDERBOOK_NAME = config_dict["YES_EXECUTION_ORDERBOOK_NAME"]
    CACHE_PREFIX = config_dict["CACHE_PREFIX"]
    POVs = config_dict["POVs"]
    FREQUENCY = config_dict["FREQUENCY"]
    DIRECTION = config_dict["DIRECTION"]
    BASELINE_LABEL = config_dict["BASELINE_LABEL"]
    SHADE_START_TIME = config_dict["SHADE_START_TIME"]
    SHADE_END_TIME = config_dict["SHADE_END_TIME"]
    YMAX = config_dict["YMAX"]
    YMIN = config_dict["YMIN"]
    XMIN = config_dict["XMIN"]
    XMAX = config_dict["XMAX"]
    ALPHA_90 = config_dict["ALPHA_90"]
    ALPHA_50 = config_dict["ALPHA_50"]
    yes_base = config_dict["yes_base"]
    no_base = config_dict["no_base"]
    log_dirs_no_glob = config_dict["log_dirs_no_glob"]
    log_dirs_yes_glob = config_dict["log_dirs_yes_glob"]
    SPREAD_LOOKBACK = config_dict["SPREAD_LOOKBACK"]
    EXPERIMENT_NAME = config_dict["EXPERIMENT_NAME"]
    LOG_DIR = config_dict["LOG_DIR"]
    EXECUTION_AGENT_NAME = config_dict["EXECUTION_AGENT_NAME"]
    COMPUTE_IMPACT_STATS = config_dict["COMPUTE_IMPACT_STATS"]

    params = generate_plot_data_cache_dicts(log_dirs_no_glob, log_dirs_yes_glob, yes_base, no_base)
    if not params:
        print("No files to process. Please check fields `yes_base`, `no_base`, `log_dirs_no_glob` and `log_dirs_yes_"
              "glob` in config file")

    if recompute:
        print(f"Recomputing aggregate data for experiment {CACHE_PREFIX}")
        make_cached(params, num_workers)
        data_to_process = load_cached()
        aggregated_data = aggregate_data(data_to_process, num_workers)

    else:
        # check for cached aggregate data
        try:
            print("Searching for cached aggregate file.")
            with open(f'cache/{CACHE_PREFIX}_pov_multiday_agg.pkl', 'rb') as f:
                aggregated_data = pickle.load(f)
            print("Aggregate cached file found")
        except FileNotFoundError:
            print("Cached aggregate file not found. Searching for individual processed files.")
            try:
                data_to_process = load_cached()
                if not data_to_process: raise FileNotFoundError("No individual cached files")
                print("Aggregating cached files...")
                aggregated_data = aggregate_data(data_to_process, num_workers)
            except (FileNotFoundError, TypeError) as e:
                print("Individual cached processed files not found, commencing processing now.")
                make_cached(params, num_workers)
                data_to_process = load_cached()
                aggregated_data = aggregate_data(data_to_process, num_workers)

    print("Plotting aggregated data...")
    register_matplotlib_converters()
    plot_all_aggregated(aggregated_data, params)
    print("Done!")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CLI utility for plotting results of multiday POV experiments.')

    parser.add_argument('plot_config',
                        help='Name of config file to execute. See plot_configs/multiday/pov_plot_config.example.json for an example.',
                        type=str)
    parser.add_argument('--num_workers',
                        '-n',
                        help='Number of cores to use in computation',
                        required=False,
                        default=8,
                        type=int
                        )
    parser.add_argument('--recompute',
                        '-r',
                        help='Switch to reaggregate data.',
                        action='store_true',
                        )

    args, remaining_args = parser.parse_known_args()

    make_cache_and_visualisation_dir()

    config_path = args.plot_config
    num_workers = args.num_workers
    recompute = args.recompute

    main(config_path, num_workers, recompute)

