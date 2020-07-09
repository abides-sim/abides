import argparse
import pandas as pd
import numpy as np
import sys
p = str(Path(__file__).resolve().parents[2])  # directory two levels up from this file
sys.path.append(p)

from realism.realism_utils import make_orderbook_for_analysis


def create_orderbooks(exchange_path, ob_path):
    MID_PRICE_CUTOFF = 10000
    processed_orderbook = make_orderbook_for_analysis(exchange_path, ob_path, num_levels=1,
                                                      hide_liquidity_collapse=False)
    cleaned_orderbook = processed_orderbook[(processed_orderbook['MID_PRICE'] > - MID_PRICE_CUTOFF) &
                                            (processed_orderbook['MID_PRICE'] < MID_PRICE_CUTOFF)]
    transacted_orders = cleaned_orderbook.loc[cleaned_orderbook.TYPE == "ORDER_EXECUTED"]

    transacted_orders = transacted_orders.reset_index()
    transacted_orders = transacted_orders.sort_values(by=['index', 'ORDER_ID']).iloc[1::2]
    transacted_orders.set_index('index', inplace=True)
    return processed_orderbook, transacted_orders, cleaned_orderbook


def calculate_market_impact(orders_df, ob_df, start_time, end_time, tao):

    def create_bins(tao, start_time, end_time, orders_df, is_buy):
        bins = pd.interval_range(start=start_time, end=end_time, freq=pd.DateOffset(seconds=tao))
        binned = pd.cut(orders_df.loc[orders_df.BUY_SELL_FLAG == is_buy].index, bins=bins)
        binned_volume = orders_df.loc[orders_df.BUY_SELL_FLAG == is_buy].groupby(binned).SIZE.agg(np.sum)
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

    ob_df = ob_df.reset_index().drop_duplicates(subset='index', keep='last').set_index('index')

    mid = ob_df.MID_PRICE
    mid_resampled = mid.resample(f'{tao}s').ffill()

    binned_buy_volume = create_bins(tao=int(tao), start_time=start_time, end_time=end_time, orders_df=orders_df,
                                    is_buy=True).fillna(0)
    binned_sell_volume = create_bins(tao=int(tao), start_time=start_time, end_time=end_time, orders_df=orders_df,
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

    parser = argparse.ArgumentParser(description='Market Impact Curve as described in AlmgrenChriss 05 paper')

    parser.add_argument('--stock', default=None, required=True, help='stock (ABM)')
    parser.add_argument('--date', default=None, required=True, help='date (20200101)')
    parser.add_argument('--log', type=str, default=None, required=True, help='log folder')
    parser.add_argument('--tao', type=int, required=True, help='Number of seconds in each bin')

    args, remaining_args = parser.parse_known_args()

    stock = args.stock
    date = args.date
    start_time = pd.Timestamp(date) + pd.to_timedelta('09:30:00')
    end_time = pd.Timestamp(date) + pd.to_timedelta('16:00:00')
    abides_log_folder = args.log

    print('Processing market impact data for {}'.format(abides_log_folder))

    processed_orderbook, transacted_orders, cleaned_orderbook = create_orderbooks(
                                            exchange_path=abides_log_folder + '/EXCHANGE_AGENT.bz2',
                                            ob_path=abides_log_folder + '/ORDERBOOK_{}_FULL.bz2'.format(stock))

    df = calculate_market_impact(transacted_orders, cleaned_orderbook, start_time, end_time, tao=args.tao)
    df.to_pickle(abides_log_folder + f'/market_impact_df_tao_{args.tao}.bz2')

    print('Processed market impact data for {}'.format(abides_log_folder))