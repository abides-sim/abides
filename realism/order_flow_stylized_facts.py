import argparse
import sys
sys.path.append("..")
from util.formatting.convert_order_stream import dir_path
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
from realism_utils import get_plot_colors
import numpy as np
from scipy import stats
from matplotlib.dates import DateFormatter
from pandas.plotting import register_matplotlib_converters
import itertools
import os
import pickle


class Constants:
    """ Stores constants for use in plotting code. """

    # Plot params -- Generic
    fig_height = 10
    fig_width = 15
    tick_label_size = 20
    legend_font_size = 20
    axes_label_font_size = 20
    title_font_size = 22
    scatter_marker_styles_sizes = [('x', 60), ('+', 60), ('o', 14), (',', 60)]

    # Plot params -- Interarrival_times
    interarrival_times_xlabel = "Quote interarrival time /s"
    interarrival_times_ylabel = "Empirical density"
    interarrival_times_filename = "interarrivals_times"
    interarrival_fit_lower_bound = 0
    interarrival_fit_upper_bound = 10
    interarrival_linewidth = 5

    # Plot params -- Binned trade counts
    binned_trade_counts_xlabel = "Normalized number of quotes"
    binned_trade_counts_ylabel = "Empirical density"
    binned_trade_counts_filename = "binned_trade_counts"
    binned_count_linewidth = 5

    # Plot params -- Intraday order volume
    intraday_volume_filename = "intraday_volume"
    intraday_volume_linewidth = 5


YEAR_OFFSET = 0.1  # adds offset of ${YEAR_OFFSET} years to each trace to differentiate between seeds


def unpickle_stream_dfs_to_stream_list(dir_containing_pickles):
    """ Extracts pickled dataframes over a number of dates to a dict containing dataframes and their dates.

        :param dir_containing_pickles: path of directory containing pickled data frames, in format `orders_SYMB_YYYYMMDD.pkl`
        :type dir_containing_pickles: str

        :return bundled_streams: list of dicts, where each dict has symbol, date and orders_df
    """

    bundled_streams = []
    symbol_regex = r".*\/orders_(\w*)_(\d{8}).*.pkl"

    stream_file_list = glob.glob(f"{dir_containing_pickles}/orders*.pkl")
    for stream_pkl in stream_file_list:
        print(f"Processing {stream_pkl}")
        match = re.search(symbol_regex, stream_pkl) 
        symbol = match.group(1)
        date_YYYYMMDD = match.group(2)
        orders_df = pd.read_pickle(stream_pkl) 
        bundled_streams.append({
            "symbol": symbol,
            "date": date_YYYYMMDD,
            "orders_df": orders_df
        })
    return bundled_streams


def set_up_plotting():
    """ Sets matplotlib variables for plotting. """
    plt.rc('xtick', labelsize=Constants.tick_label_size)
    plt.rc('ytick', labelsize=Constants.tick_label_size)
    plt.rc('legend', fontsize=Constants.legend_font_size)
    plt.rc('axes', labelsize=Constants.axes_label_font_size)


def bundled_stream_interarrival_times(bundled_streams):
    """ From bundled streams return dict with interarrival times collated by symbol. """

    interarrivals_dict = dict()
    year_offset = 0

    for idx, elem in enumerate(bundled_streams):
        print(f"Processing elem {idx + 1} of {len(bundled_streams)}")

        year_offset_td = pd.Timedelta(int(365 * (year_offset * YEAR_OFFSET)), unit='day')
        orders_df = elem["orders_df"]
        symbol = elem["symbol"]
        arrival_times = orders_df.index.to_series()
        # offset arrival times by ${YEAR_OFFSET} to separate same day, different seed
        arrival_times = arrival_times + year_offset_td
        arrival_times.index = arrival_times.index + year_offset_td

        interarrival_times = arrival_times.diff()
        interarrival_times = interarrival_times.iloc[1:].apply(pd.Timedelta.total_seconds)
        interarrival_times = interarrival_times.rename("Interarrival time /s")

        if symbol not in interarrivals_dict.keys():
            interarrivals_dict[symbol] = interarrival_times
        else:
            interarrivals_dict[symbol] = interarrivals_dict[symbol].append(interarrival_times)

        year_offset += 1

    return interarrivals_dict


def plot_interarrival_times(interarrivals_dict, output_dir, scale='log'):
    """ Plots histogram of the interarrival times for symbols. """

    fig, ax = plt.subplots(figsize=(Constants.fig_width, Constants.fig_height))

    if scale == 'log':
        ax.set(xscale="symlog", yscale="log")

    ax.set_ylabel(Constants.interarrival_times_ylabel)
    ax.set_xlabel(Constants.interarrival_times_xlabel)

    symbols = list(interarrivals_dict.keys())
    symbols.sort()
    colors = get_plot_colors(symbols)
    alphas = [1] * len(symbols)

    x_s = []

    for symbol, color, alpha in zip(symbols, colors, alphas):
        interarrival_times_series = interarrivals_dict[symbol]
        x = interarrival_times_series.sort_values()
        x_s.append(x)
        plt.hist(x, bins="sqrt", density=True, label=symbol, color=color, alpha=alpha, histtype="step",
                 linewidth=Constants.interarrival_linewidth)

    ylim = ax.get_ylim()
    xlim = ax.get_xlim()

    xx = np.linspace(*xlim, 200)

    # Plot fitted curves, leave out zeroes for better fit
    for x, symbol, color in zip(x_s, symbols, colors):
        x = x[(x > Constants.interarrival_fit_lower_bound) & (x < Constants.interarrival_fit_upper_bound)]
        weibull_params = stats.weibull_min.fit(x, floc=0)

        x_left = xx[xx < x.min()][1:]
        x_mid = x.to_numpy()
        x_right = xx[xx > x.max()]
        xxx = np.concatenate((x_left, x_mid, x_right))

        plt.plot(xxx, stats.weibull_min.pdf(xxx, *weibull_params), linestyle="--", color=color,
                 label=f"{symbol} Weibull fit", linewidth=Constants.interarrival_linewidth)

    plt.legend(fontsize=Constants.legend_font_size)
    ax.set_ylim(ylim)

    fig.savefig(f'{output_dir}/{Constants.interarrival_times_filename}.png', format='png', dpi=300,
                transparent=False, bbox_inches='tight', pad_inches=0.03)


def count_trades_within_bins(interarrival_times_series, binwidth=1):
    """ Bins trades into specified-width bins and counts the number.

        :param interarrival_times_series: pandas Series object corresponding to the interarrival times, indexed on timestamp
        :param binwidth: width of time bin in seconds

    """
    bins = pd.interval_range(start=interarrival_times_series.index[0].floor('min'), end=interarrival_times_series.index[-1].ceil('min'),
                             freq=pd.DateOffset(seconds=binwidth)) 
    binned = pd.cut(interarrival_times_series.index, bins=bins)
    counted = interarrival_times_series.groupby(binned).count()
    return counted


def bundled_stream_binned_trade_counts(bundled_interarrivals_dict, binwidth):
    trades_within_bins_dict = dict()

    for symbol, interarrival_times in bundled_interarrivals_dict.items():
        series_list = [group[1] for group in interarrival_times.groupby(interarrival_times.index.date)]
        for idx, series in enumerate(series_list):
            print(f"Processing series {idx + 1} of {len(series_list)} for symbol {symbol}")
            counted_trades = count_trades_within_bins(series, binwidth=binwidth)
            counted_trades_copy = counted_trades.copy(deep=True)
            
            if idx == 0:
                base_counted = counted_trades_copy
                hist_index = base_counted.index
                #print(f'base_counted.index: {base_counted.index}')
                #print(f'hist_index: {hist_index}')

            else:
                #print(f'counted_trades_copy.index: {counted_trades_copy.index}')
                #print(f'hist_index: {hist_index}')    

                try:
                    counted_trades_copy.index = hist_index
                    base_counted = base_counted.add(counted_trades_copy)
                except ValueError: # length mismatch of hist bins (currently ignore)
                    continue

        trades_within_bins_dict[symbol] = base_counted

    return trades_within_bins_dict


def plot_binned_trade_counts(trades_within_bins_dict, binwidth, output_dir):
    """ Plot binned counts of trade volume. """

    fig, ax = plt.subplots(figsize=(Constants.fig_width, Constants.fig_height))

    ax.set(yscale="log")

    ax.set_ylabel(Constants.binned_trade_counts_ylabel)
    ax.set_xlabel(Constants.binned_trade_counts_xlabel)

    symbols = list(trades_within_bins_dict.keys())
    symbols.sort()
    colors = get_plot_colors(symbols)
    alphas = [1] * len(symbols)

    x_s = []
    for symbol, color, alpha in zip(symbols, colors, alphas):
        binned_trades_counts = trades_within_bins_dict[symbol].copy(deep=True)
        binned_trades_counts = binned_trades_counts / binned_trades_counts.sum()
        x = binned_trades_counts.sort_values()
        x_s.append(x)
        plt.hist(x, bins="sqrt", density=True, label=symbol, color=color, alpha=alpha, histtype="step",
                 linewidth=Constants.binned_count_linewidth)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx = np.linspace(*xlim, 200)

    # Plot fitted curves
    for x, symbol, color in zip(x_s, symbols, colors):
        gamma_params = stats.gamma.fit(x.values[x.values > 0], floc=0)
        plt.plot(xx, stats.gamma.pdf(xx, *gamma_params), linestyle="--", color=color, label=f"{symbol} gamma fit",
                 linewidth=Constants.binned_count_linewidth)

    ax.set_ylim(ylim)

    plt.title(f"Order volume within time window $\\tau = ${binwidth} seconds, normalized",
              size=Constants.title_font_size)
    plt.legend(fontsize=Constants.legend_font_size)

    fig.savefig(f'{output_dir}/{Constants.binned_trade_counts_filename}_tau_{binwidth}.png',
                format='png', dpi=300, transparent=False, bbox_inches='tight', pad_inches=0.03)


def get_scatter_plot_params_dict(symbols):
    """ Creates dictionary of parameters used by intraday seasonality plots. """
    colors = get_plot_colors(symbols)
    scatter_styles_sizes = itertools.cycle(Constants.scatter_marker_styles_sizes)
    scatter_plot_params_dict = dict()

    for symbol, color, style_and_size in zip(symbols, colors, scatter_styles_sizes):
        scatter_plot_params_dict.update({
            symbol : {
                'color': color,
                'marker': style_and_size[0],
                'marker_size': style_and_size[1]
            }
        })

    return scatter_plot_params_dict


def plot_intraday_seasonality(trades_within_bins_dict, binsize, output_dir):
    """ Plots intraday order volume over a day for multiple symbols."""

    fig, ax = plt.subplots(figsize=(Constants.fig_width, Constants.fig_height))
    ax.set_ylabel("Normalized activity")
    ax.set_xlabel("Time of day")

    myFmt = DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(myFmt)

    symbols = list(trades_within_bins_dict.keys())
    symbols.sort()

    for symbol, binned_trades in trades_within_bins_dict.items():

        x_fit = []

        for elem in binned_trades.index:
            py_datetime = elem.right.to_pydatetime()
            seconds_from_epoch = py_datetime.timestamp()
            x_fit.append(seconds_from_epoch)

        y = binned_trades.values / np.mean(binned_trades.values)

        start_time_from_epoch = x_fit[0]
        x_fit = np.array(x_fit) - start_time_from_epoch
        quad_coeff = np.polyfit(x_fit, y, 2)
        intraday_quadratic_fitted_y = np.poly1d(quad_coeff)(x_fit)

        plot_ticker_style = get_scatter_plot_params_dict(symbols)
        color = plot_ticker_style[symbol]["color"]
        marker = plot_ticker_style[symbol]["marker"]
        marker_size = plot_ticker_style[symbol]["marker_size"]

        x = []
        for elem in binned_trades.index:
            x.append(elem.right.time())

        plt.scatter(x, y, marker=marker, color=color, label=symbol, s=marker_size)
        plt.plot(x, intraday_quadratic_fitted_y, color=color, label=f"{symbol} quadratic fit",
                 linewidth=Constants.intraday_volume_linewidth)

    plt.legend(fontsize=Constants.legend_font_size)
    plt.title(f"Number of limit orders submitted in $\Delta t = {binsize}$ seconds, normalized by mean volume.", size=Constants.title_font_size, pad=20)

    ax.set_xticklabels(["", "09:45", "11:00", "12:30", "14:00", "15:15"])
    ax.set_ylim(-0.1, 3)

    fig.savefig(f'{output_dir}/{Constants.intraday_volume_filename}.png', format='png', dpi=300,
                transparent=False, bbox_inches='tight', pad_inches=0.03)


if __name__ == "__main__":

    # Create cache and visualizations folders if they do not exist
    try: os.mkdir("cache")
    except: pass
    try: os.mkdir("visualizations")
    except: pass

    parser = argparse.ArgumentParser(description='Process order stream files and produce plots of relevant metrics.')
    parser.add_argument('targetdir', type=dir_path, help='Path of directory containing order stream files. Note that they must have been preprocessed'
                                                         ' by formatting scripts into format orders_{symbol}_{date_str}.pkl')
    parser.add_argument('-o', '--output-dir', default='visualizations', help='Path to output directory', type=dir_path)
    parser.add_argument('-f','--facts-to-plot', choices=['all'], type=str, default='all',
                        help="Decide which stylized facts should be plotted.")

    parser.add_argument('-z', '--recompute', action="store_true", help="Rerun computations without caching.")
    args, remaining_args = parser.parse_known_args()

    print("### Order stream stylized facts plots ###")

    set_up_plotting()

    bundled_orders_dict = unpickle_stream_dfs_to_stream_list(args.targetdir)

    ## interarrival times
    pickled_bundled_interarrivals_dict = "cache/bundled_interarrivals_dict.pkl"
    if (not os.path.exists(pickled_bundled_interarrivals_dict)) or args.recompute:
        print("Computing interarrivals times...")
        bundled_interarrivals_dict = bundled_stream_interarrival_times(bundled_orders_dict)
        pickle.dump(bundled_interarrivals_dict, open(pickled_bundled_interarrivals_dict, "wb"))
    else:
        bundled_interarrivals_dict = pickle.load(open(pickled_bundled_interarrivals_dict, "rb"))

    print("Plotting interarrivals times...")
    plot_interarrival_times(bundled_interarrivals_dict, args.output_dir)

    ## transaction volume binned
    pickled_binned_30_days_1_minute = "cache/binned_30_days_1_minute.pkl"
    if (not os.path.exists(pickled_binned_30_days_1_minute)) or args.recompute:
        print("Computing 1 minute transaction volumes...")
        binned_30_days_1_minute = bundled_stream_binned_trade_counts(bundled_interarrivals_dict, 60)
        pickle.dump(binned_30_days_1_minute, open(pickled_binned_30_days_1_minute, "wb"))
    else:
        binned_30_days_1_minute = pickle.load(open(pickled_binned_30_days_1_minute, "rb"))

    print("Plotting 1 minute transaction volumes...")
    plot_binned_trade_counts(binned_30_days_1_minute, 60, args.output_dir)

    pickled_binned_30_days_5_minute = "cache/binned_30_days_5_minute.pkl"
    if (not os.path.exists(pickled_binned_30_days_5_minute)) or args.recompute:
        print("Computing 5 minute transaction volumes...")
        binned_30_days_5_minute = bundled_stream_binned_trade_counts(bundled_interarrivals_dict, 300)
        pickle.dump(binned_30_days_5_minute, open(pickled_binned_30_days_5_minute, "wb"))
    else:
        binned_30_days_5_minute = pickle.load(open(pickled_binned_30_days_5_minute, "rb"))

    print("Plotting 5 minute transaction volumes...")
    plot_binned_trade_counts(binned_30_days_5_minute, 300, args.output_dir)

    ## intraday seasonality
    print("Plotting intraday seasonality...")
    plot_intraday_seasonality(binned_30_days_5_minute, 300, args.output_dir)

