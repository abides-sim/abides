import pandas as pd
from datetime import datetime
from util.util import log_print


class OrderBookOracle:
    COLUMNS = ['TIMESTAMP', 'ORDER_ID', 'PRICE', 'SIZE', 'BUY_SELL_FLAG']
    DIRECTION = {0: 'BUY', 1: 'SELL'}

    # Oracle for reading historical exchange orders stream
    def __init__(self, symbol, date, start_time, end_time, orders_file_path):
        self.symbol = symbol
        self.date = date
        self.start_time = start_time
        self.end_time = end_time
        self.orders_file_path = orders_file_path
        self.orders_dict = self.processOrders()
        self.wakeup_times = [*self.orders_dict]
        self.first_wakeup = self.wakeup_times[0]

        print(f"Orders File Path: {self.orders_file_path}")
        print(f"Oracle initialized for {self.symbol} and date {self.date.date} "
              f"between {str(self.start_time.hour)}: {str(self.start_time.minute)} "
              f"and {str(self.end_time.hour)}: {str(self.end_time.minute)}")

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
        orders_df = orders_df[OrderBookOracle.COLUMNS]
        orders_df['BUY_SELL_FLAG'] = orders_df['BUY_SELL_FLAG'].astype(int).replace(OrderBookOracle.DIRECTION)
        orders_df['TIMESTAMP'] = orders_df['TIMESTAMP'].astype(str).apply(convertDate)
        orders_df['SIZE'] = orders_df['SIZE'].astype(int)
        orders_df['PRICE'] = orders_df['PRICE'].astype(float)
        orders_df = orders_df.loc[(orders_df.TIMESTAMP >= self.start_time) & (orders_df.TIMESTAMP < self.end_time)]
        orders_df.set_index('TIMESTAMP', inplace=True)
        log_print(f"Number of Orders: {len(orders_df)}")
        orders_dict = {k: g.to_dict(orient='records') for k, g in orders_df.groupby(level=0)}
        return orders_dict

    def getDailyOpenPrice(self, symbol, mkt_open):
        return self.orders_dict[list(self.orders_dict.keys())[0]][0]['PRICE']

    def observePrice(self, symbol, currentTime, sigma_n=0):
        return self.getDailyOpenPrice(symbol, mkt_open=currentTime)
