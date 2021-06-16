import argparse
import os
import sys
sys.path.append("..")
from util.formatting.convert_order_stream import dir_path
from order_flow_stylized_facts import unpickle_stream_dfs_to_stream_list, YEAR_OFFSET
import pickle
from realism_utils import get_plot_colors
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from util.util import OrderSizeDistribution

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

    # Plot params -- Limit Order size
    limit_order_sizes_xlabel = "Limit Order Size"
    limit_order_sizes_ylabel = "Empirical density"
    limit_order_sizes_filename = "limit_order_sizes"
    limit_order_size_fit_lower_bound = 0
    limit_order_size_fit_upper_bound = 10
    limit_order_size_linewidth = 5


def bundled_stream_limit_order_sizes(bundled_streams):
    """ From bundled streams return dict with limit order sizes collated by symbol. """

    limit_order_sizes_dict = dict()

    for idx, elem in enumerate(bundled_streams):
        print(f"Processing elem {idx + 1} of {len(bundled_streams)}")
        orders_df = elem["orders_df"]
        symbol = elem["symbol"]
        limit_orders = orders_df[orders_df['TYPE'] == "LIMIT_ORDER"]["SIZE"]

        if symbol not in limit_order_sizes_dict.keys():
            limit_order_sizes_dict[symbol] = limit_orders
        else:
            limit_order_sizes_dict[symbol] = limit_order_sizes_dict[symbol].append(limit_orders)

    return limit_order_sizes_dict


def plot_limit_order_sizes(limit_order_sizes_dict, output_dir, scale='log'):
    """ Plots histogram of the limit order sizes for symbols. """

    fig, ax = plt.subplots(figsize=(Constants.fig_width, Constants.fig_height))

    if scale == 'log':
        ax.set(xscale="log", yscale="log")

    ax.set_ylabel(Constants.limit_order_sizes_ylabel)
    ax.set_xlabel(Constants.limit_order_sizes_xlabel)

    symbols = list(limit_order_sizes_dict.keys())
    symbols.sort()
    colors = get_plot_colors(symbols)
    alphas = [1] * len(symbols)

    x_s = []

    for symbol, color, alpha in zip(symbols, colors, alphas):
        limit_order_sizes_series = limit_order_sizes_dict[symbol]
        x = limit_order_sizes_series.sort_values()
        x_s.append(x)
        plt.hist(x, bins="sqrt", density=True, label=symbol, color=color, alpha=alpha, histtype="step",
                 linewidth=Constants.limit_order_size_linewidth)

    ylim = ax.get_ylim()
    xlim = ax.get_xlim()

    xx = np.linspace(*xlim, 200)

    # # Plot fitted curves, leave out zeroes for better fit
    for x, symbol, color in zip(x_s, symbols, colors):
        # x = x[(x > Constants.interarrival_fit_lower_bound) & (x < Constants.interarrival_fit_upper_bound)]
        # power_law_params = stats.powerlaw.fit(x)#, floc=0)
        x_left = xx[xx < x.min()][1:]
        x_mid = x.to_numpy()
        x_right = xx[xx > x.max()]
        xxx = np.concatenate((x_left, x_mid, x_right))
        #
        # plt.plot(xxx, stats.powerlaw.pdf(xxx, *power_law_params), linestyle="--", color=color,
        #          label=f"{symbol} Power law fit a={power_law_params[0]:.3f}, loc={power_law_params[1]:.3f}, "
        #                f"scale={power_law_params[2]:.3f}", linewidth=Constants.limit_order_size_linewidth)

        num_spikes = 5
        probs = [1 / (num_spikes + 1)] * (num_spikes + 1)
        print(f"probs: {probs}")
        # order_size_dist = OrderSizeDistribution(0.15, *probs)
        shapes = OrderSizeDistribution.get_shape_list(probs)
        order_size_fit_params = OrderSizeDistribution(shapes=shapes).fit(x, 0.15, *probs, loc=1, scale=50000)
        print(f"order_size_fit_params: {order_size_fit_params}")
        plt.plot(xxx, OrderSizeDistribution.pdf(xxx, *order_size_fit_params), linestyle="--", color=color,
                 label=f"{symbol} Order size distribution fit", linewidth=Constants.limit_order_size_linewidth)

    plt.legend(fontsize=Constants.legend_font_size)
    ax.set_ylim(ylim)

    fig.savefig(f'{output_dir}/{Constants.limit_order_sizes_filename}.png', format='png', dpi=300,
                transparent=False, bbox_inches='tight', pad_inches=0.03)


def set_up_plotting():
    """ Sets matplotlib variables for plotting. """
    plt.rc('xtick', labelsize=Constants.tick_label_size)
    plt.rc('ytick', labelsize=Constants.tick_label_size)
    plt.rc('legend', fontsize=Constants.legend_font_size)
    plt.rc('axes', labelsize=Constants.axes_label_font_size)


if __name__ == "__main__":

    # Create cache and visualizations folders if they do not exist
    try: os.mkdir("cache")
    except: pass
    try: os.mkdir("visualizations")
    except: pass

    parser = argparse.ArgumentParser(description='Process order stream files and produce plots of order size (limit and executed).')
    parser.add_argument('targetdir', type=dir_path, help='Path of directory containing order stream files. Note that they must have been preprocessed'
                                                         ' by formatting scripts into format orders_{symbol}_{date_str}.pkl')
    parser.add_argument('-o', '--output-dir', default='visualizations', help='Path to output directory', type=dir_path)

    parser.add_argument('-z', '--recompute', action="store_true", help="Rerun computations without caching.")
    args, remaining_args = parser.parse_known_args()

    bundled_orders_dict = unpickle_stream_dfs_to_stream_list(args.targetdir)

    print("### Order size stylized facts plots ###")

    ## limit order sizes
    pickled_bundled_limit_order_sizes_dict = "cache/bundled_limit_order_sizes_dict.pkl"
    if (not os.path.exists(pickled_bundled_limit_order_sizes_dict)) or args.recompute:
        print("Computing limit order sizes...")
        bundled_limit_order_sizes_dict = bundled_stream_limit_order_sizes(bundled_orders_dict)
        pickle.dump(bundled_limit_order_sizes_dict, open(pickled_bundled_limit_order_sizes_dict, "wb"))
    else:
        bundled_limit_order_sizes_dict = pickle.load(open(pickled_bundled_limit_order_sizes_dict, "rb"))

    print("Plotting limit order sizes...")
    plot_limit_order_sizes(bundled_limit_order_sizes_dict, args.output_dir)
