import argparse
import os
import pandas as pd
import numpy as np

import sys
from pathlib import Path
p = str(Path(__file__).resolve().parents[2])  # directory two levels up from this file
sys.path.append(p)

from util.formatting.convert_order_stream import get_year_month_day, get_start_end_time, dir_path, check_positive
from tqdm import tqdm


def get_larger_int_and_gap(a, b):
    return (True, a - b) if a >= b else (False, b - a)


def get_int_from_string(s):
    int_list = [int(s) for s in s.split('_') if s.isdigit()]
    return int_list[0]


def process_row(row, quote_levels):
    """ Method takes row of unstacked orderbook log and processes into a dictionary representing a row of the LOBSTER-
        ised DataFrame.
    """

    row_arr = row[1].to_numpy()

    bid_orders_idx = np.nonzero(row_arr < 0)
    ask_orders_idx = np.nonzero(row_arr > 0)

    ask_values = quote_levels[ask_orders_idx].to_numpy()
    ask_volumes = row_arr[ask_orders_idx]

    bid_values = quote_levels[bid_orders_idx].to_numpy()
    bid_values = np.flip(bid_values)
    bid_volumes = row_arr[bid_orders_idx]
    bid_volumes = np.flip(-bid_volumes)

    num_bids = bid_values.size
    num_asks = ask_values.size

    more_bids_then_asks, difference = get_larger_int_and_gap(num_bids, num_asks)

    if more_bids_then_asks:
        ask_values = np.pad(ask_values.astype(np.float32), (0, difference), 'constant', constant_values=np.nan)
        ask_volumes = np.pad(ask_volumes.astype(np.float32), (0, difference), 'constant', constant_values=np.nan)
    else:
        bid_values = np.pad(bid_values.astype(np.float32), (0, difference), 'constant', constant_values=np.nan)
        bid_volumes = np.pad(bid_volumes.astype(np.float32), (0, difference), 'constant', constant_values=np.nan)

    ask_volumes_dict = {f"ask_size_{idx + 1}": ask_volumes[idx] for idx in range(len(ask_volumes))}
    ask_values_dict = {f"ask_price_{idx + 1}": ask_values[idx] for idx in range(len(ask_values))}
    bid_volumes_dict = {f"bid_size_{idx + 1}": bid_volumes[idx] for idx in range(len(bid_volumes))}
    bid_values_dict = {f"bid_price_{idx + 1}": bid_values[idx] for idx in range(len(bid_values))}

    row_dict = {**ask_volumes_dict, **ask_values_dict, **bid_volumes_dict, **bid_values_dict}
    return row_dict


def reorder_columns(unordered_cols):
    """ Reorders column list to coincide with columns of LOBSTER csv file format. """

    ask_price_cols = [label for label in unordered_cols if 'ask_price' in label]
    ask_size_cols = [label for label in unordered_cols if 'ask_size' in label]
    bid_price_cols = [label for label in unordered_cols if 'bid_price' in label]
    bid_size_cols = [label for label in unordered_cols if 'bid_size' in label]

    bid_price_cols.sort(key=get_int_from_string)
    bid_size_cols.sort(key=get_int_from_string)
    ask_price_cols.sort(key=get_int_from_string)
    ask_size_cols.sort(key=get_int_from_string)

    bid_price_cols = np.array(bid_price_cols)
    bid_size_cols = np.array(bid_size_cols)
    ask_price_cols = np.array(ask_price_cols)
    ask_size_cols = np.array(ask_size_cols)

    new_col_list_size = ask_price_cols.size + ask_size_cols.size + bid_price_cols.size + bid_size_cols.size
    new_col_list = np.empty((new_col_list_size,), dtype='<U16')

    new_col_list[0::4] = ask_price_cols
    new_col_list[1::4] = ask_size_cols
    new_col_list[2::4] = bid_price_cols
    new_col_list[3::4] = bid_size_cols
    return new_col_list


def finalise_processing(orderbook_df, level):
    """ Clip to requested level and fill NaNs according to LOBSTER spec. """

    bid_columns = [col for col in orderbook_df.columns if "bid" in col]
    ask_columns = [col for col in orderbook_df.columns if "ask" in col]

    orderbook_df[bid_columns] = orderbook_df[bid_columns].fillna(value=-9999999999)
    orderbook_df[ask_columns] = orderbook_df[ask_columns].fillna(value=9999999999)

    num_levels = int((len(orderbook_df.columns) - 1) / 4) + 1
    columns_to_drop = []
    columns_to_drop.extend([f'ask_price_{idx}' for idx in range(1, num_levels + 1) if idx > level])
    columns_to_drop.extend([f'ask_size_{idx}' for idx in range(1, num_levels + 1) if idx > level])
    columns_to_drop.extend([f'bid_price_{idx}' for idx in range(1, num_levels + 1) if idx > level])
    columns_to_drop.extend([f'bid_size_{idx}' for idx in range(1, num_levels + 1) if idx > level])

    orderbook_df = orderbook_df.drop(columns=columns_to_drop)

    return orderbook_df


def is_wide_book(df):
    """ Checks if orderbook dataframe is in wide or skinny format. """
    if isinstance(df.index, pd.core.index.MultiIndex):
        return False
    else:
        return True


def process_orderbook(df, level):
    """ Method takes orderbook log and transforms into format amenable to "LOBSTER-ification"

    :param df: pd.DataFrame orderbook output by ABIDES
    :param level: Maximum displayed level in book
    :return:
    """

    if not is_wide_book(df):  # orderbook skinny format
        unstacked = df.unstack(level=-1)
        quote_levels = unstacked.columns.levels[1]
    else:  # orderbook wide format
        unstacked = df
        quote_levels = df.columns

    # Make DataFrame reshaped to LOBSTER format
    rows_list = []
    for row in tqdm(unstacked.iterrows(), total=len(unstacked.index), desc="Processing order book"):
        row_dict = process_row(row, quote_levels)
        rows_list.append(row_dict)

    nearly_lobster = pd.DataFrame(rows_list)

    # Reorder columns
    unordered_cols = list(nearly_lobster.columns)
    new_col_list = reorder_columns(unordered_cols)
    nearly_lobster = nearly_lobster[new_col_list]

    # Clip to requested level and fill NaNs
    orderbook_df = finalise_processing(nearly_lobster, level)

    return orderbook_df


def save_formatted_order_book(orderbook_bz2, ticker, level, out_dir='.'):
    """ Saves orderbook data from ABIDES in LOBSTER format.

        :param orderbook_bz2: file path of order book bz2 output file.
        :type orderbook_bz2: str
        :param ticker: label of security
        :type ticker: str
        :param level: maximum level of order book to display
        :type level: int
        :param out_dir: path to output directory
        :type out_dir: str

        :return:

        ============

        Orderbook File:     (Matrix of size: (Nx(4xNumberOfLevels)))
        ---------------

        Name:   TICKER_Year-Month-Day_StartTime_EndTime_orderbook_LEVEL.csv

        Columns:

            1.) Ask Price 1:    Level 1 Ask Price   (Best Ask)
            2.) Ask Size 1:     Level 1 Ask Volume  (Best Ask Volume)
            3.) Bid Price 1:    Level 1 Bid Price   (Best Bid)
            4.) Bid Size 1:     Level 1 Bid Volume  (Best Bid Volume)
            5.) Ask Price 2:    Level 2 Ask Price   (2nd Best Ask)
            ...


    """

    orderbook_df = pd.read_pickle(orderbook_bz2)

    if not is_wide_book(orderbook_df):  # skinny format
        trading_day = get_year_month_day(pd.Series(orderbook_df.index.levels[0]))
        start_time, end_time = get_start_end_time(orderbook_df, 'orderbook_skinny')
    else:  # wide format
        trading_day = get_year_month_day(pd.Series(orderbook_df.index))
        start_time, end_time = get_start_end_time(orderbook_df, 'orderbook_wide')

    orderbook_df = process_orderbook(orderbook_df, level)

    # Save to file

    #filename = f'{ticker}_{trading_day}_{start_time}_{end_time}_orderbook_{str(level)}.csv'
    filename = f'orderbook.csv'
    filename = os.path.join(out_dir, filename)

    orderbook_df.to_csv(filename, index=False, header=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process ABIDES order book data into the LOBSTER format.')
    parser.add_argument('book', type=str, help='ABIDES order book in bz2 format. '
                                               'Typical example is `orderbook_TICKER.bz2`')
    parser.add_argument('-o', '--output-dir', default='.', help='Path to output directory', type=dir_path)
    parser.add_argument('ticker', type=str, help="Ticker label")
    parser.add_argument('level', type=check_positive, help="Maximum orderbook level.")

    args, remaining_args = parser.parse_known_args()

    save_formatted_order_book(args.book, args.ticker, args.level, out_dir=args.output_dir)
