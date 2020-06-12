import argparse
import pandas as pd
from pandas.io.json import json_normalize
import json
import os


def extract_events_from_stream(stream_df, event_type):
    """ Extracts specific event from stream.

    """
    events = stream_df.loc[stream_df.EventType == event_type][['EventTime', 'Event']]
    events_json = events['Event'].to_json(orient="records")
    json_struct = json.loads(events_json)
    # TODO : get rid of structs containing all `int` types
    event_extracted = json_normalize(json_struct)
    event_extracted = pd.merge(events['EventTime'].reset_index(), event_extracted, left_index=True, right_index=True)

    if not event_extracted.empty:
        event_extracted = event_extracted[['EventTime', 'order_id', 'limit_price', 'quantity', 'is_buy_order']]
        event_extracted.rename(columns={'EventTime': 'TIMESTAMP',
                                        'order_id': 'ORDER_ID',
                                        'limit_price': 'PRICE',
                                        'quantity': 'SIZE',
                                        'is_buy_order': 'BUY_SELL_FLAG'}, inplace=True)
    else:
        event_extracted = pd.DataFrame({
            'TIMESTAMP': [],
            'ORDER_ID': [],
            'PRICE': [],
            'SIZE': [],
            'BUY_SELL_FLAG': []
        })

    return event_extracted


def seconds_since_midnight(s):
    """ Converts a pandas Series object of datetime64[ns] timestamps to Series of seconds from midnight on that day.

        Inspired by https://stackoverflow.com/a/38050344
    """
    d = pd.to_datetime(s.dt.date)
    delta_t = s - d
    return delta_t.dt.total_seconds()


def convert_stream_to_format(stream_df, fmt="LOBSTER"):
    """ Converts imported ABIDES DataFrame into LOBSTER FORMAT.
    """
    event_dfs = []
    market_events = {
        "LIMIT_ORDER": 1,
        # "MODIFY_ORDER": 2, # causing errors in market replay
        "ORDER_CANCELLED": 3,
        "ORDER_EXECUTED": 4
    }
    reversed_market_events = {val: key for key, val in market_events.items()}

    for event_name, lobster_code in market_events.items():
        event_df = extract_events_from_stream(stream_df, event_name)
        if event_df.empty:
            continue
        else:
            event_df["Time"] = seconds_since_midnight(event_df["TIMESTAMP"])
            event_df["Type"] = lobster_code
            event_dfs.append(event_df)

    lobster_df = pd.concat(event_dfs)

    if fmt == "plot-scripts":

        lobster_df["Type"].replace(reversed_market_events, inplace=True)
        lobster_df.rename(columns={'Type': "TYPE"}, inplace=True)
        lobster_df = lobster_df.sort_values(by=['TIMESTAMP'])
        lobster_df = lobster_df[['TIMESTAMP', 'ORDER_ID', 'PRICE', 'SIZE', 'BUY_SELL_FLAG', 'TYPE']]
        return lobster_df

    elif fmt == "LOBSTER":
        lobster_df["Order ID"] = lobster_df["ORDER_ID"]
        lobster_df["Size"] = lobster_df["SIZE"]
        lobster_df["Price"] = 100 * lobster_df["PRICE"]
        lobster_df["Direction"] = (lobster_df["BUY_SELL_FLAG"] * 2) - 1

        lobster_df = lobster_df[["Time", "Type", "Order ID", "Size", "Price", "Direction"]]
        lobster_df = lobster_df.sort_values(by=['Time'])
        return lobster_df

    else:
        raise ValueError('Format needs to be "plot-scripts" or "LOBSTER"')


def get_year_month_day(s):
    """ Returns date as string from pandas series of timestamps.

        :param s:
        :type s: pandas.Series(datetime64[ns])

        :return s_date_str: str in format YYYY-MM-DD

    """
    t = s.loc[s.first_valid_index()]
    return f'{t.year}-{"{:02}".format(t.month)}-{"{:02}".format(t.day)}'


def get_start_end_time(df, fmt):
    """ Returns first and last timestamp of pandas DataFrame in plot-scripts format or LOBSTER format. """

    if fmt == "plot-scripts":
        t = seconds_since_midnight(df['TIMESTAMP'])
        return int(round(t.iloc[0])), int(round(t.iloc[-1]))
    elif fmt == "LOBSTER":
        return int(round(df['Time'].iloc[0])), int(round(df['Time'].iloc[-1]))
    elif fmt == "orderbook_skinny":
        t = seconds_since_midnight(df.index.levels[0].to_series())
        return int(round(t.iloc[0])), int(round(t.iloc[-1]))
    elif fmt == "orderbook_wide":
        t = seconds_since_midnight(df.index.to_series())
        return int(round(t.iloc[0])), int(round(t.iloc[-1]))
    else:
        raise ValueError('Format needs to be "plot-scripts" or "LOBSTER" or "orderbook_skinny" or "orderbook_wide"')


def save_formatted_order_stream(stream_bz2, ticker, level, fmt, suffix, out_dir='.'):
    """ Saves ABIDES logged order stream into csv in requested format.

        :param stream_bz2: file path of Exchange Agent bz2 output file.
        :type stream_bz2: str
        :param ticker: label of security
        :type ticker: str
        :param level: maximum level of order book to display
        :type level: int
        :param fmt: Specifies the output format, current options are "plot-scripts" and "LOBSTER".
        :type fmt: str
        :param suffix: suffix to add to file name before extension
        :type suffix: str
        :param out_dir: path to output directory
        :type out_dir: str

        :return:

        =============

        PLOT-SCRIPTS FORMAT                                              (Matrix of size: (Nx5))
        --------------

        Name: TICKER_Year-Month-Day_StartTime_EndTime_message_LEVEL.csv

        Columns:

            1) TIMESTAMP

            2) ORDER_ID

            3) PRICE

            4) SIZE

            5) BUY_SELL_FLAG


        LOBSTER FORMAT (compliant with version 01 Sept 2013):       (Matrix of size: (Nx6))
        --------------

        Name:   TICKER_Year-Month-Day_StartTime_EndTime_message_LEVEL.csv

            StartTime and EndTime give the theoretical beginning
            and end time of the output file in milliseconds after
            mid night. LEVEL refers to the number of levels of the
            requested limit order book.


        Columns:

            1.) Time:
                    Seconds after midnight with decimal
                    precision of at least milliseconds
                    and up to nanoseconds depending on
                    the requested period
            2.) Type:
                    1: Submission of a new limit order
                    2: Cancellation (Partial deletion
                       of a limit order)
                    3: Deletion (Total deletion of a limit order)
                    4: Execution of a visible limit order
                    5: Execution of a hidden limit order
                    7: Trading halt indicator
                       (Detailed information below)
            3.) Order ID:
                    Unique order reference number
                    (Assigned in order flow)
            4.) Size:
                    Number of shares
            5.) Price:
                    Dollar price times 10000
                    (i.e., A stock price of $91.14 is given
                    by 911400)
            6.) Direction:
                    -1: Sell limit order
                    1: Buy limit order

                    Note:
                    Execution of a sell (buy) limit
                    order corresponds to a buyer (seller)
                    initiated trade, i.e. Buy (Sell) trade.

    """

    stream_df = pd.read_pickle(stream_bz2).reset_index()
    write_df = convert_stream_to_format(stream_df, fmt=fmt)

    # Save to file
    trading_day = get_year_month_day(stream_df['EventTime'])
    start_time, end_time = get_start_end_time(write_df, fmt)


    if fmt == "plot-scripts":
        filename = f'orders_{ticker}_{trading_day.replace("-", "")}{suffix}.pkl'
        filename = os.path.join(out_dir, filename)
        write_df = write_df.set_index('TIMESTAMP')
        write_df.to_pickle(filename)
    elif fmt == "LOBSTER":
        filename = f'{ticker}_{trading_day}_{start_time}_{end_time}_message_{str(level)}{suffix}.csv'
        filename = os.path.join(out_dir, filename)
        write_df.to_csv(filename, index=False, header=False)
    else:
        raise ValueError('Format needs to be "plot-scripts" or "LOBSTER"')




def dir_path(string):
    """ https://stackoverflow.com/a/51212150 """
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def check_positive(value):
    """ https://stackoverflow.com/a/14117511 """
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is an invalid positive int value")
    return ivalue


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process ABIDES stream data into either plotting or LOBSTER formats.')
    parser.add_argument('stream', type=str, help='ABIDES order stream in bz2 format. '
                                                 'Typical example is `ExchangeAgent.bz2`')
    parser.add_argument('-o', '--output-dir', default='.', help='Path to output directory', type=dir_path)
    parser.add_argument('ticker', type=str, help="Ticker label")
    parser.add_argument('level', type=check_positive, help="Maximum orderbook level.")
    parser.add_argument('format', choices=['plot-scripts', 'LOBSTER'], type=str,
                        help="Output format of stream")
    parser.add_argument('--suffix', type=str, help="optional suffix to add to filename.", default="")


    args, remaining_args = parser.parse_known_args()

    save_formatted_order_stream(args.stream, args.ticker, args.level, args.format, args.suffix, out_dir=args.output_dir)
