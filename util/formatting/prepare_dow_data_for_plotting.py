import pandas as pd
from datetime import datetime
from util.formatting.convert_order_stream import dir_path
import argparse
from dateutil.parser import parse
from datetime import timedelta
import os


class Oracle:
    """ Class that creates a pandas DataFrame ready for processing by plotting framework.

        Thanks @Mahmoud for the code!
    """
    COLUMNS = ['TIMESTAMP', 'ORDER_ID', 'PRICE', 'SIZE', 'BUY_SELL_FLAG']
    DIRECTION = {0: 'BUY', 1: 'SELL'}

    def __init__(self, symbol, date, start_time, end_time, orders_file_path):
        self.symbol             = symbol
        self.date               = date
        self.start_time         = start_time
        self.end_time           = end_time
        self.orders_file_path   = orders_file_path
        self.orders_df          = self.processOrders()

    def processOrders(self):

        def convertDate(date_str):
            try:
                return datetime.strptime(date_str, '%Y%m%d%H%M%S.%f')
            except ValueError:
                return convertDate(date_str[:-1])

        orders_df = pd.read_csv(self.orders_file_path).iloc[1:]
        all_columns = orders_df.columns[0].split('|')
        orders_df = orders_df[orders_df.columns[0]].str.split('|', 16, expand=True)
        orders_df.columns = all_columns
        orders_df = orders_df[Oracle.COLUMNS]
        orders_df['BUY_SELL_FLAG'] = orders_df['BUY_SELL_FLAG'].astype(int).replace(Oracle.DIRECTION)
        orders_df['TIMESTAMP'] = orders_df['TIMESTAMP'].astype(str).apply(convertDate)
        orders_df['SIZE'] = orders_df['SIZE'].astype(int)
        orders_df['PRICE'] = orders_df['PRICE'].astype(float)
        orders_df = orders_df.loc[(orders_df.TIMESTAMP >= self.start_time) & (orders_df.TIMESTAMP < self.end_time)]
        orders_df.set_index('TIMESTAMP', inplace=True)
        return orders_df


def check_dates_valid(start_date, end_date):
    valid_date_order = pd.to_datetime(start_date) <= pd.to_datetime(end_date)
    if not valid_date_order:
        raise ValueError("Start date > final date")


def get_date_range(start_date, end_date):
    """ https://stackoverflow.com/a/7274316 """
    delta = end_date - start_date
    date_range = [(start_date + timedelta(days=i)).strftime('%Y%m%d') for i in range(delta.days + 1)]
    return date_range


def dow_data_to_pickle(dow_data_dir, symbol, start_date, end_date, out_dir):
    """ Saves files of the form orders_{symbol}_{date}.pkl """

    date_range = get_date_range(start_date, end_date)

    for date in date_range:
        print(f"Processing file for symbol {symbol} on date: {date}")
        mkt_open = pd.to_datetime(date) + pd.to_timedelta('09:30:00')
        mkt_close = pd.to_datetime(date) + pd.to_timedelta('16:00:00')
        oracle = Oracle(symbol=symbol, date=date, start_time=mkt_open, end_time=mkt_close,
                        orders_file_path=f'{dow_data_dir}/{symbol}/{symbol}.{date}')
        oracle.orders_df.to_pickle(f'{out_dir}/orders_{symbol}_{date}.pkl')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Processes DOW30 data into pickled pd.DataFrame files required as '
                                                 'input for plotting.')
    parser.add_argument('dow_data_dir', type=dir_path, help='Path to directory containing all of the DOW30 data in EFS.')
    parser.add_argument('-o', '--output-dir', default='.', help='Path to output directory', type=dir_path)
    parser.add_argument('ticker', type=str, help="Ticker label")
    parser.add_argument('start_date', type=parse, help='First date of DOW30 data to use for a given symbol in format YYYYMMDD.')
    parser.add_argument('end_date', type=parse, help='Final date of DOW30 data to use for a given symbol in format YYYYMMDD.')

    args, remaining_args = parser.parse_known_args()

    check_dates_valid(args.start_date, args.end_date)
    dow_data_to_pickle(args.dow_data_dir, args.ticker, args.start_date, args.end_date, args.output_dir)
