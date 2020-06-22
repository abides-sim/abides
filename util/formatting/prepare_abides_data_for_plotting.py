from util.formatting.convert_order_stream import convert_stream_to_format
import os
import argparse
from dateutil.parser import parse
from util.formatting.convert_order_stream import dir_path
import pandas as pd


def process_abides_order_stream(stream_bz2, symbol, out_dir, date):
    """ Writes ABIDES stream data into pandas DataFrame required by plotting programs. """
    stream_df = pd.read_pickle(stream_bz2, compression='bz2').reset_index()
    write_df = convert_stream_to_format(stream_df, fmt="plot-scripts")
    write_df = write_df.set_index('TIMESTAMP')
    date_str = date.strftime('%Y%m%d')
    file_name = f"{out_dir}/orders_{symbol}_{date_str}.pkl"
    write_df.to_pickle(file_name)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process ABIDES stream data into plotting format and pickles it.')
    parser.add_argument('stream', type=str, help='ABIDES order stream in bz2 format. '
                                                 'Typical example is `ExchangeAgent.bz2`')
    parser.add_argument('-o', '--output-dir', default='.', help='Path to output directory', type=dir_path)
    parser.add_argument('ticker', type=str, help="Ticker label to give the ABIDES data.")
    parser.add_argument('date', type=parse, help="Date of the trading day in format YYYYMMDD.")

    args, remaining_args = parser.parse_known_args()
    process_abides_order_stream(args.stream, args.ticker, args.output_dir, args.date)
