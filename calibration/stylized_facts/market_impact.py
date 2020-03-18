import argparse
import pandas as pd
import numpy as np
import datetime as dt
import logging as log

from calibration.log_util import get_orderbook, get_transacted_orders


def calculate_market_impact(transacted_orders_df, orderbook_df, start_time, end_time, tao):
    """
    
    :param transacted_orders_df: Transacted Orders Pandas Dataframe
    :param orderbook_df: L2 Orderbook snapshots Pandas Dataframe
    :param start_time: Start Time Pandas Timestamp
    :param end_time: End Time Pandas Timestamp
    :param tao: time interval
    :return: market impact Pandas Dataframe

    Market impact of order placement is a expected to grow as a function of order volume.

    - For each time interval $\tau$, define $V_{\text{buy},\tau}$ and $V_{\text{ask},\tau}$ to be  buy and sell order volumes in $\tau$
    respectively.

    - Define participation of volume in $\tau$ as
    $$ P_{\tau}=\frac{\left| V_{\text{buy},\tau} - V_{\text{ask},\tau} \right|}{V_{\text{buy},\tau} + V_{\text{ask},\tau}}. $$
    Note that $0 \leq P_{\tau} \leq 1$.

    - Also define $\Delta m_\tau$ to be the observable mid-price move in $\tau$.

    - Discretize the range for $P_{\tau}$ into bins $B_i, i=1, \ldots, N$ such that $B_i =\{\tau: \frac{i-1}{N} \leq P_{\tau}\leq \frac{i}{N} \}$.
    - For each $B_i$, define
    \begin{eqnarray*}
        M_i = \frac{1}{|B_i|}\sum_{\tau \in B_i} \Delta m_\tau \quad\text{and}\quad P_i =
        \frac{1}{|B_i|}\sum_{\tau \in B_i} \Delta P_\tau
    \end{eqnarray*}
      to be the average price move and average participation of volume in bins with similar volume participation.
    - One can then fit a relationship of the form $M_i \sim \alpha P_i ^{\beta}$ through the data.
    - Source: Get Real: Realism Metrics for Robust Limit Order Book Market Simulations https://arxiv.org/abs/1912.04941

    """
    # TODO: Make code cleaner and more effecient

    def create_bins(tao, start_time, end_time, transacted_orders_df, is_buy):
        bins = pd.interval_range(start=start_time, end=end_time, freq=pd.DateOffset(seconds=tao))
        binned = pd.cut(transacted_orders_df.loc[transacted_orders_df.BUY_SELL_FLAG == is_buy].index, bins=bins)
        binned_volume = transacted_orders_df.loc[transacted_orders_df.BUY_SELL_FLAG == is_buy].groupby(binned).SIZE.agg(np.sum)
        return binned_volume

    def calculate_mid_move(row):
        try:
            t_start = row.name.left
            t_end = row.name.right
            mid_t_start = mid_resampled.loc[mid_resampled.index == t_start].item()
            mid_t_end = mid_resampled.loc[mid_resampled.index == t_end].item()
            if row.ti < 0:
                row.mi = -1 * ((mid_t_end - mid_t_start) / mid_t_start) * 10000 # bps
            else:
                row.mi = (mid_t_end - mid_t_start) / mid_t_start * 10000 # bps
            return row.mi
        except:
            pass

    mid = (orderbook_df.ask_price_1 + orderbook_df.bid_price_1) / 2
    mid_resampled = mid.resample(f'{tao}s').ffill()

    binned_buy_volume = create_bins(tao=int(tao), start_time=start_time, end_time=end_time, transacted_orders_df=transacted_orders_df,
                                    is_buy=True).fillna(0)
    binned_sell_volume = create_bins(tao=int(tao), start_time=start_time, end_time=end_time, transacted_orders_df=transacted_orders_df,
                                     is_buy=False).fillna(0)

    midf = pd.DataFrame()
    midf['buy_vol'] = binned_buy_volume
    midf['sell_vol'] = binned_sell_volume
    midf['ti'] = midf['buy_vol'] - midf['sell_vol']  # Trade Imbalance
    midf['pov'] = abs(midf['ti']) / (midf['buy_vol'] + midf['sell_vol'])  # Participation of Volume in tao
    midf['mi'] = None
    midf.index = pd.interval_range(start=start_time, end=end_time, freq=pd.DateOffset(seconds=int(tao)))

    midf.mi = midf.apply(calculate_mid_move, axis=1)

    pov_bins = np.linspace(start=0, stop=1, num=1000, endpoint=False)
    pov_binned = pd.cut(x=midf['pov'], bins=pov_bins)

    midf['pov_bins'] = pov_binned

    midf_gpd = midf.sort_values(by='pov_bins')
    midf_gpd.index = midf_gpd.pov_bins
    del midf_gpd['pov_bins']

    df = pd.DataFrame(index=midf_gpd.index)
    df['mi'] = midf_gpd['mi']
    df['pov'] = midf_gpd['pov']
    df = df.groupby(df.index).mean()

    return df


if __name__ == "__main__":

    script_start_time = dt.datetime.now()
    log.basicConfig(level=log.INFO)

    system_name = '  ABIDES: Market Impact'
    log.info('=' * len(system_name))
    log.info(system_name)
    log.info(' ')

    parser = argparse.ArgumentParser(description='Market Impact Curve as described in AlmgrenChriss 05 paper')

    parser.add_argument('--tao', type=int, required=True, help='Number of seconds in each bin')

    # ABS:
    '''
    parser.add_argument('--seed', type=int, default=None, required=True, help='Seed')
    parser.add_argument('--rw', type=int, default=None, required=True, help='Random walk number')
    
    security = 'ABS'
    date = '2020-01-01'
    
    abides_logs_parent_folder = '../log/'
    abides_log_folder = abides_logs_parent_folder + f'aamas.hist_fund_diverse_rw_{args.rw}_seed_{args.seed}/'

    '''

    # Market Replay:
    parser.add_argument('--security', required=True, help='Name of the stock/symbol')
    parser.add_argument('--date', required=True, help='Historical date')

    args, remaining_args = parser.parse_known_args()
    security = args.security
    date = args.date

    log.info(f'Calculating Market Impact for {security} and date {date}')
    log.info(f'Loading the order book transacted orders and L2 snapshots')

    start_time = pd.Timestamp(date) + pd.to_timedelta('09:30:00')
    end_time = pd.Timestamp(date) + pd.to_timedelta('16:00:00')

    csv_orderbooks_parent_folder = '/efs/data/get_real_data/lobsterized/orderbook/'
    abides_logs_parent_folder = '/efs/data/get_real_data/marketreplay-logs/log/'
    abides_log_folder = abides_logs_parent_folder + f'marketreplay_{security}_{date.replace("-", "")}/'

    orderbook_df = get_orderbook(abides_log_folder=abides_log_folder,
                                 csv_orderbooks_parent_folder=csv_orderbooks_parent_folder,
                                 security=security,
                                 date=date,
                                 start_time=start_time,
                                 end_time=end_time,
                                 num_price_levels=50)

    transacted_orders_df = get_transacted_orders(abides_log_folder=abides_log_folder)

    df = calculate_market_impact(transacted_orders_df, orderbook_df, start_time, end_time, tao=args.tao)
    df.to_pickle(abides_log_folder + f'market_impact_df_tao_{args.tao}.bz2')

    log.info(f'Processed market impact data for {security} {date}')
    log.info('=' * len(system_name))