import pandas as pd


def get_orderbook(abides_log_folder, csv_orderbooks_parent_folder,
                  security, date, start_time, end_time, num_price_levels):
    """

    :param abides_log_folder:
    :param security: Name of the stock/symbol, e.g. IBM
    :param date: historical date, e.g. 2019-06-28
    :param num_price_levels: number of order book price levels
    :return:
    """
    columns = [x for b in [[f'ask_price_{level}', f'ask_size_{level}',
                            f'bid_price_{level}', f'bid_size_{level}']
                           for level in range(1, num_price_levels + 1)]
               for x in b]

    if csv_orderbooks_parent_folder:
        orderbook_df = pd.read_csv(csv_orderbooks_parent_folder + f'orderbook_{security}_{date.replace("-", "")}.csv')
    else:
        orderbook_df = pd.read_csv(abides_log_folder + f'orderbook_{security}_{date.replace("-", "")}.csv')

    orderbook_df.columns = columns
    orderbook_df.index = pd.read_pickle(abides_log_folder + f'ORDERBOOK_{security}_FREQ_ALL_{date.replace("-", "")}.bz2').index[1:]
    # orderbook_df.index = pd.read_pickle(abides_log_folder + 'ORDERBOOK_ABS_FULL.bz2').index.get_level_values(0).unique()[1:]

    orderbook_df = orderbook_df.loc[(orderbook_df.index >= start_time) & (orderbook_df.index <= end_time)]

    return orderbook_df


def get_transacted_orders(abides_log_folder):
    """

    :param abides_log_folder:
    :return:
    """

    exchange_df = pd.read_pickle(abides_log_folder + 'EXCHANGE_AGENT.bz2')
    exchange_df = exchange_df.loc[exchange_df.EventType == 'ORDER_EXECUTED']

    transacted_orders_df = pd.DataFrame(columns=['TIMESTAMP', 'ORDER_ID', 'PRICE', 'SIZE', 'BUY_SELL_FLAG'])

    # TODO: More effecient way for this
    i = 0
    for index, row in exchange_df.iterrows():
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
    return transacted_orders_df
