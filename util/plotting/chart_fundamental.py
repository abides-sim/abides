import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter
from dateutil.parser import parse
from pandas.plotting import register_matplotlib_converters
import argparse
import sys
sys.path.append('..')
from formatting.convert_order_stream import dir_path


class Constants:
    fig_width = 13
    fig_height = 9
    tick_label_size = 20
    axes_label_font_size = 20
    title_font_size = 22
    legend_font_size = 20
    filename = 'fundamental'


def set_up_plotting():
    """ Sets matplotlib variables for plotting. """
    plt.rc('xtick', labelsize=Constants.tick_label_size)
    plt.rc('ytick', labelsize=Constants.tick_label_size)
    plt.rc('axes', labelsize=Constants.axes_label_font_size)
    plt.rc('axes', titlesize=Constants.title_font_size)
    plt.rc('legend', fontsize=Constants.legend_font_size)


def plot_fundamental(fundamentals_df_list, legend_labels, plot_title, output_dir):

    fig, ax = plt.subplots(figsize=(Constants.fig_width, Constants.fig_height))

    ax.set_ylabel("Fundamental value (cents)")
    ax.set_xlabel("Time of day")

    myFmt = DateFormatter("%H:%M")
    ax.xaxis.set_major_formatter(myFmt)
    plt.title(plot_title)

    for df, label in zip(fundamentals_df_list, legend_labels):
        x = df.index
        y = df['FundamentalValue']
        plt.plot(x, y, label=label)

    plt.legend()

    fig.savefig(f'{output_dir}/{Constants.filename}.png', format='png', dpi=300, transparent=False, bbox_inches='tight',
                pad_inches=0.03)
    plt.show()


def validate_input(fundamentals, legend_labels):
    if len(fundamentals) != len(legend_labels):
        raise ValueError("Number of fundamental files and number of legend labels specified must be equal.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Chart fundamental for ABIDES simulations.')
    parser.add_argument('-f', '--fundamental-file', action='append', required=True, help='bz2 file containing the fundamental'
                                                                                         ' over time. Note can add multiple instances of this variable for'
                                                                                         ' and overlayed chart.')
    parser.add_argument('-l', '--legend-label', action='append', required=False, help='label for the legend entry '
                                                                                      'corresponding to fundamental-file. Must have as many legen-label variables as'
                                                                                      ' fundamental-file variables.')
    parser.add_argument('-t', '--title', action='store', default='', help='Chart title.')
    parser.add_argument('-o', '--output-dir', default='.', help='Path to output directory', type=dir_path)

    args, remaining_args = parser.parse_known_args()

    register_matplotlib_converters()
    set_up_plotting()
    validate_input(args.fundamental_file, args.legend_label)

    fundamentals_df_list = [pd.read_pickle(f) for f in args.fundamental_file]
    legend_labels = args.legend_label
    plot_title = args.title
    output_dir = args.output_dir

    plot_fundamental(fundamentals_df_list, legend_labels, plot_title, output_dir)

