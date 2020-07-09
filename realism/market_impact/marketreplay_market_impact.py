import argparse
import pandas as pd
import numpy as np

num_levels = 50
columns = [[f'ask_price_{level}', f'ask_size_{level}', f'bid_price_{level}', f'bid_size_{level}'] for level in range(1, num_levels+1)]
columns = [x for b in columns for x in b]


def process_data(abides_log_folder, stock, date):
    csv_orderbooks_parent_folder = '/efs/data/get_real_data/lobsterized/orderbook/'

    # Orderbook snapshots
    ob_df = pd.read_csv(csv_orderbooks_parent_folder + f'orderbook_{stock}_{date}.csv')
    ob_df.columns = columns
    ob_df.index = pd.read_pickle(abides_log_folder + f'ORDERBOOK_{stock}_FREQ_ALL_{date.replace("-", "")}.bz2').index[1:]

    start_time = pd.Timestamp(date) + pd.to_timedelta('09:30:00')
    end_time = pd.Timestamp(date) + pd.to_timedelta('16:00:00')
    ob_df = ob_df.loc[(ob_df.index >= start_time) & (ob_df.index <= end_time)]


    # Transacted Orders
    ea_df = pd.read_pickle(abides_log_folder + 'EXCHANGE_AGENT.bz2')
    ea_df = ea_df.loc[ea_df.EventType == 'ORDER_EXECUTED']

    transacted_orders_df = pd.DataFrame(columns=['TIMESTAMP', 'ORDER_ID', 'PRICE', 'SIZE', 'BUY_SELL_FLAG'])

    i = 0
    for index, row in ea_df.iterrows():
        transacted_orders_df = transacted_orders_df.append(pd.Series(data={
            'TIMESTAMP': index,
            'ORDER_ID': row.Event['order_id'],
            'PRICE': row.Event['fill_price'],
            'SIZE': row.Event['quantity'],
            'BUY_SELL_FLAG': row.Event['is_buy_order']
        }), ignore_index=True)
        i += 1

    transacted_orders_df.set_index('TIMESTAMP', inplace=True)

    transacted_orders_df = transacted_orders_df.sort_values(by=['TIMESTAMP', 'ORDER_ID']).iloc[1::2]

    return ob_df, transacted_orders_df, start_time, end_time


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

    mid = (ob_df.ask_price_1 + ob_df.bid_price_1) / 2
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

    parser.add_argument('--tao',  required=True, help='Number of seconds in each bin')
    parser.add_argument('--ticker', required=True,  help='Name of the stock/symbol')
    parser.add_argument('--date', required=True, help='Historical date')

    args, remaining_args = parser.parse_known_args()

    stock = args.ticker
    date = args.date

    abides_logs_parent_folder = '/efs/data/get_real_data/marketreplay-logs/log/'
    abides_log_folder = abides_logs_parent_folder + f'marketreplay_{stock}_{date.replace("-", "")}/'

    ob_df, orders_df, start_time, end_time = process_data(abides_log_folder=abides_log_folder,
                                                          stock=stock,
                                                          date=date)
    print(f'Processed order book data for {stock} {date}, calculating market impact ...')

    df = calculate_market_impact(orders_df, ob_df, start_time, end_time, tao=args.tao)
    df.to_pickle(abides_log_folder + f'market_impact_df_tao_{args.tao}.bz2')

    print(f'Processed market impact data for {stock} {date}')