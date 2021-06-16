import sys
from pathlib import Path

p = str(Path(__file__).resolve().parents[2])  # directory two levels up from this file
sys.path.append(p)

from util.formatting.convert_order_book import process_orderbook, is_wide_book
from util.formatting.convert_order_stream import dir_path
import pandas as pd
import os
import argparse


def save_mid_price(orderbook_file_path, output_dir):
    """ Save order book mid price, computed from ABIDES orderbook log. """

    orderbook_df = pd.read_pickle(orderbook_file_path)
    processed_df = process_orderbook(orderbook_df, 1)

    # Compute mid price and associate to timestamp
    mid_price = (processed_df['ask_price_1'] + processed_df['bid_price_1']) / 2
    if not is_wide_book(orderbook_df):
        mid_price.index = orderbook_df.index.levels[0]
    else:
        mid_price.index = orderbook_df.index

    # Save file
    base = os.path.basename(orderbook_file_path)
    base_no_ext = os.path.splitext(base)[0]
    filename = os.path.join(output_dir, f"{base_no_ext}_mid_price.bz2")
    mid_price.to_pickle(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Processes ABIDES orderbook data into pickled pd.DataFrame containing '
                                                 'mid price at timestep.')
    parser.add_argument('-o', '--output-dir', default='.', help='Path to output directory', type=dir_path)
    parser.add_argument('book', type=str, help='ABIDES order book output in bz2 format. ')
    args, remaining_args = parser.parse_known_args()

    save_mid_price(args.book, args.output_dir)
