import sys
import pandas as pd
import numpy as np
from pathlib import Path
p = str(Path(__file__).resolve().parents[1])  # directory one level up from this file
sys.path.append(p)
from util.formatting.convert_order_book import process_orderbook, is_wide_book
from util.formatting.convert_order_stream import convert_stream_to_format
import itertools
from bisect import bisect
from matplotlib.cm import get_cmap
import os
import warnings
from util.util import get_value_from_timestamp


MID_PRICE_CUTOFF = 10000  # Price above which mid price is set as `NaN` and subsequently forgotten. WARNING: This
                         # effectively hides dropout of liquidity on ask side. Set to sys.max_size to reset.
LIQUIDITY_DROPOUT_WARNING_MSG = "No liquidity on one side of the order book during this experimental trace."


def get_trades(sim_file):
  
  # Code taken from `read_simulated_trades`
  try:
    df = pd.read_pickle(sim_file, compression='bz2')
  except (OSError, EOFError):
      return None
  
  df = df[df['EventType'] == 'LAST_TRADE']
  if len(df) <= 0:
    print("There appear to be no simulated trades.")
    sys.exit()
  df['PRICE'] = [y for x,y in df['Event'].str.split(',')]
  df['SIZE'] = [x for x,y in df['Event'].str.split(',')]
  df['PRICE'] = df['PRICE'].str.replace('$','').astype('float64')
  df['SIZE'] = df['SIZE'].astype('float64')

  # New code for minutely resampling and renaming columns.
  df = df[["PRICE","SIZE"]].resample("1T")
  df_open = df["PRICE"].first().ffill()
  df_close = df["PRICE"].last().ffill()
  df_high = df["PRICE"].max().ffill()
  df_low = df["PRICE"].min().ffill()
  df_vol = df["SIZE"].sum()
  ohlcv = pd.DataFrame({
      "open": df_open,
      "high": df_high,
      "low": df_low,
      "close": df_close,
      "volume": df_vol
  })
  ohlcv = ohlcv.iloc[:390,:]
  return ohlcv


def clip_times(df, start, end):
    """ Keep only rows within certain time bounds of dataframe.

        :param df: DataFrame with DatetimeIndex
        :param start: lower bound
        :param end: upper bound

        :type df: pd.DataFrame
        :type start: pd.Timestamp
        :type end: pd.Timestamp

    """
    return df.loc[(df.index > start) & (df.index < end)]


def mid_price_cutoff(df):
    """ Removes outliers for mid-price induced by no liquidity. """

    out = df.loc[(df["MID_PRICE"] < MID_PRICE_CUTOFF) & (df["MID_PRICE"] > - MID_PRICE_CUTOFF)]

    if len(df.index) > len(out.index):
        warnings.warn(LIQUIDITY_DROPOUT_WARNING_MSG, UserWarning, stacklevel=1)

    return out


def augment_with_VWAP(merged):
    """ Method augments orderbook with volume weighted average price.
    """

    merged = merged.reset_index()
    executed_df = merged.loc[merged["TYPE"] == "ORDER_EXECUTED"]
    executed_df = executed_df.dropna()
    vwap = (executed_df['PRICE'].multiply(executed_df['SIZE'])).cumsum() / executed_df['SIZE'].cumsum()
    vwap = vwap.to_frame(name='VWAP')
    merged = pd.merge(merged.reset_index(), vwap, how='left', left_index=True, right_index=True)
    merged['VWAP'] = merged['VWAP'].fillna(method='ffill')
    merged['VWAP'] = merged['VWAP']
    merged = merged.set_index('index')
    merged = merged.drop(columns=['level_0'])
    del merged.index.name

    return merged


def make_orderbook_for_analysis(stream_path, orderbook_path, num_levels=5, ignore_cancellations=True, hide_liquidity_collapse=True):
    """  Make orderbook amenable to mid-price + liquidity plots from ABIDES input.

         :param stream_path: path to ABIDES Exchange output, e.g. ExchangeAgent0.bz2. Note ABIDES must have been run with --log-orders=True
         :param orderbook_path: path to ABIDES order book output, e.g. ORDERBOOK_TICKER_FULL.bz2. Note ABIDES must have been run with --book-freq not set to None
         :param num_levels: number of levels of orderbook to keep in DataFrame.
         :param ignore_cancellations: flag to only include executed trades
         :param hide_liquidity_collapse: flag to remove times in order book with no liquidity on one side of book

         :type stream_path: str
         :type orderbook_path: str
         :type num_levels: int
         :type ignore_cancellations: bool
         :type hide_liquidity_collapse: bool

    """

    stream_df = pd.read_pickle(stream_path)
    orderbook_df = pd.read_pickle(orderbook_path)

    stream_processed = convert_stream_to_format(stream_df.reset_index(), fmt='plot-scripts')
    stream_processed = stream_processed.set_index('TIMESTAMP')

    ob_processed = process_orderbook(orderbook_df, num_levels)

    if not is_wide_book(orderbook_df):  # orderbook in skinny format
        ob_processed.index = orderbook_df.index.levels[0]
    else:  # orderbook in wide format
        ob_processed.index = orderbook_df.index

    columns = list(itertools.chain(
        *[[f'ask_price_{level}', f'ask_size_{level}', f'bid_price_{level}', f'bid_size_{level}'] for level in
          range(1, num_levels + 1)]))
    merged = pd.merge(stream_processed, ob_processed, left_index=True, right_index=True, how='left')
    merge_cols = ['ORDER_ID', 'PRICE', 'SIZE', 'BUY_SELL_FLAG', 'TYPE'] + columns
    merged = merged[merge_cols]
    merged['PRICE'] = merged['PRICE'] / 100

    # clean
    # merged = merged.dropna()
    merged = merged.ffill()

    # Ignore cancellations
    if ignore_cancellations:
        merged = merged[merged.SIZE != 0]

    merged['MID_PRICE'] = (merged['ask_price_1'] + merged['bid_price_1']) / (2 * 100)
    merged['SPREAD'] = (merged['ask_price_1'] - merged['bid_price_1']) / 100
    merged['ORDER_VOLUME_IMBALANCE'] = merged['ask_size_1'] / (merged['bid_size_1'] + merged['ask_size_1'])

    if hide_liquidity_collapse:
        merged = mid_price_cutoff(merged)

    # add VWAP
    merged = augment_with_VWAP(merged)
    print("Orderbook construction complete!")

    return merged


def get_daily_spread(orderbook_df, start_time='09:30:00', end_time='16:00:00'):
    """ Get mean spread for the day's trading.

        :param orderbook_df: preprocessed orderbook (see __name__.make_orderbook_for_analysis) for data without execution agent.
        :param start_time: time to "start" trading day -- in format HH:MM:SS
        :param end_time: time to "finish" trading day -- in format HH:MM:SS

        :type orderbook_df: pd.DataFrame
        :type start_time: str
        :type end_time: str
    """
    historical_date = pd.Timestamp(orderbook_df.index[0].date())
    start = historical_date + pd.to_timedelta(start_time)
    end = historical_date + pd.to_timedelta(end_time)
    day_spread = clip_times(orderbook_df["SPREAD"], start, end)
    mean_daily_spread = day_spread.mean()
    return mean_daily_spread


def find_nearest_ts_idx(df, np_dt64):
    """ https://stackoverflow.com/a/42266882 """
    timestamps = np.array(df.index)
    upper_index = bisect(timestamps, np_dt64, hi=len(timestamps) - 1)  # find the upper index of the closest time stamp
    df_index = df.index.get_loc(min(timestamps[upper_index], timestamps[upper_index - 1], key=lambda x: abs(
        x - np_dt64)))  # find the closest between upper and lower timestamp
    return df_index


def first_elem(s):
    """ Extracts first element of pandas Series s, or returns s if not a series. """
    try:
        return s.iloc[0]
    except AttributeError:
        return s


def get_relevant_prices(orderbook_df, orderbook_with_execution_df, start_ts, end_ts):

    start_idx_orig = find_nearest_ts_idx(orderbook_df, start_ts.to_datetime64())
    end_idx_orig = find_nearest_ts_idx(orderbook_df, end_ts.to_datetime64())
    end_idx_execution = find_nearest_ts_idx(orderbook_with_execution_df, end_ts.to_datetime64())

    start_mid_price_orig = first_elem(orderbook_df['MID_PRICE'].iloc[start_idx_orig])
    end_mid_price_orig = first_elem(orderbook_df['MID_PRICE'].iloc[end_idx_orig])
    end_mid_price_execution = first_elem(orderbook_with_execution_df['MID_PRICE'].iloc[end_idx_execution])

    return start_mid_price_orig, end_mid_price_orig, end_mid_price_execution


def get_execution_agent_vwap(experiment_name, agent_name, date, seed, pov, log_dir):
    """ Function computes the VWAP for an execution agent's orders, when ran from the `execution_iabs_plots` config.

        :param experiment_name: name for experiment
        :param agent_name: name of agent, e.g. POV_EXECUTION_AGENT
        :param date: date of experiment in format YYYY-MM-DD
        :param seed: seed used to run experiment
        :param pov: Participation of volume for agent
        :param log_dir: location of directory with all ABIDES logs

        :type experiment_name: str
        :type agent_name: str
        :type date: str
        :type seed: int
        :type pov: float
        :type log_dir: str

    """
    file_path = f'{log_dir}/{experiment_name}_yes_{seed}_{pov}_{date}/{agent_name}.bz2'
    exec_df = pd.read_pickle(file_path)

    executed_orders = exec_df.loc[exec_df['EventType'] == 'ORDER_EXECUTED']
    executed_orders['PRICE'] = executed_orders['Event'].apply(lambda x: x['fill_price'])
    executed_orders['SIZE'] = executed_orders['Event'].apply(lambda x: x['quantity'])

    executed_orders["VWAP"] = (executed_orders['PRICE'].multiply(executed_orders['SIZE'])).cumsum() / executed_orders[
        'SIZE'].cumsum()
    executed_orders["VWAP"] = executed_orders["VWAP"] / 100
    final_vwap = executed_orders.iloc[-1].VWAP

    return final_vwap


def compute_impact_statistics(orderbook_df, orderbook_with_execution_df, start_time, end_time, date_str, pov, seed, experiment_name,
                              spread_lookback='1min', execution_agent_name='POV_EXECUTION_AGENT', log_dir='../log'):
    """ Computes dictionary of run statistics for comparison.

        :param orderbook_df: preprocessed orderbook (see __name__.make_orderbook_for_analysis) for data without execution agent.
        :param orderbook_with_execution_df: preprocessed orderbook (see __name__.make_orderbook_for_analysis) for data with execution agent.

        :type orderbook_df: pd.DataFrame
        :type orderbook_with_execution_df: pd.DataFrame
    """

    start_ts = pd.Timestamp(start_time)
    end_ts = pd.Timestamp(end_time)

    start_mid_price_orig, end_mid_price_orig, end_mid_price_execution = get_relevant_prices(orderbook_df, orderbook_with_execution_df, start_ts, end_ts)
    end_mid_price_execution_bps = 10000 * end_mid_price_execution / end_mid_price_orig

    end_shade_str = end_ts.strftime('%H:%M:%S')
    end_shade_str_lookback = (end_ts - pd.to_timedelta(spread_lookback)).strftime('%H:%M:%S')

    mean_daily_spread = get_daily_spread(orderbook_df, end_shade_str_lookback, end_shade_str)

    daily_VWAP_price = get_value_from_timestamp(orderbook_df["VWAP"], end_ts)

    execution_agent_vwap = get_execution_agent_vwap(experiment_name, execution_agent_name, date_str, seed, pov, log_dir)

    mid_price_shift_bps = 10000 * (end_mid_price_execution - end_mid_price_orig) / end_mid_price_orig
    vwap_plus_half_spread_dollars = daily_VWAP_price + 0.5 * mean_daily_spread
    vwap_plus_half_spread_bps = 10000 * vwap_plus_half_spread_dollars / end_mid_price_orig

    stats_dict = {
        "start_mid_price_orig ($)": start_mid_price_orig,
        "end_mid_price_orig ($)": end_mid_price_orig,
        "end_mid_price_execution ($)": end_mid_price_execution,
        "end_mid_price_execution (bps)": end_mid_price_execution_bps,
        'mid_price_difference ($)': (end_mid_price_execution - end_mid_price_orig),
        "mid_price_impact_bps": mid_price_shift_bps,
        "daily_VWAP_price ($)": daily_VWAP_price,
        "daily_VWAP_price (bps)": 10000 * daily_VWAP_price / end_mid_price_orig,
        "mean_daily_spread ($)": mean_daily_spread,
        "mean_daily_spread (bps)": 10000 * mean_daily_spread / end_mid_price_orig,
        "VWAP + half spread ($)":  vwap_plus_half_spread_dollars,
        "VWAP + half spread (bps)": vwap_plus_half_spread_bps,
        "execution_impact_from_VWAP_plus_half_spread ($)": end_mid_price_execution - vwap_plus_half_spread_dollars,
        "execution_impact_from_VWAP_plus_half_spread (bps)": end_mid_price_execution_bps - vwap_plus_half_spread_bps,
        "execution_agent_vwap ($)": execution_agent_vwap,
        "execution_agent_vwap (bps)": 10000 * execution_agent_vwap / end_mid_price_orig
    }

    return stats_dict


def get_plot_colors(symbols, start_idx=0):
    name = "Set1"
    cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
    colors = cmap.colors  # type: list
    return colors[start_idx:(len(symbols) + start_idx)]


def get_plot_linestyles(n):
    """ https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/linestyles.html """
    linestyle_tuple = [
        ('solid', (0, ())),
        ('dotted', (0, (1, 1))),  # Same as (0, (1, 1)) or '.'
        ('densely dotted', (0, (1, 1))),
        ('dashed', (0, (5, 5))),
        ('less densely dashed', (0, (3, 1))),
        ('dashdotted', (0, (3, 5, 1, 5))),
        ('densely dashdotted', (0, (3, 1, 1, 1))),
        ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
        ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))
         ]
    out_list = [y for (x,y) in linestyle_tuple]
    return out_list[:n]


def forward_fill_series(s1, s2):
    """ For two pandas series with DateTimeIndex , return corresponding series with the same numer of entries, forward-filled.

        :type s1: pd.Series
        :type s2: pd.Series
    """

    def dedup_index(s):
        """ Deduplicate index values of pd.Series"""
        df = s.to_frame()
        df = df.loc[~df.index.duplicated(keep='last')]
        s_out = df[df.columns[0]]
        return s_out

    s1_out = dedup_index(s1)
    s2_out = dedup_index(s2)

    missing_times_from_s2 = s1_out.index.difference(s2_out.index)
    missing_times_from_s1 = s2_out.index.difference(s1_out.index)

    dummy_to_add_to_s1 = pd.Series([np.NaN] * missing_times_from_s1.size, index=missing_times_from_s1)
    dummy_to_add_to_s2 = pd.Series([np.NaN] * missing_times_from_s2.size, index=missing_times_from_s2)

    s2_out = s2_out.append(dummy_to_add_to_s2)
    s1_out = s1_out.append(dummy_to_add_to_s1)

    s1_out = s1_out.sort_index()
    s2_out = s2_out.sort_index()

    s1_out = s1_out.ffill()
    s2_out = s2_out.ffill()

    return s1_out, s2_out


def make_cache_and_visualisation_dir():
    # Create cache folder if it does not exist
    try:
        os.mkdir("cache")
    except:
        pass
    try:
        os.mkdir("visualizations")
    except:
        pass
