import numpy as np
import pandas as pd

from util.util import log_print

class RandomOrderBookOracle:
    order_id = 0

    def __init__(self, symbol,
                 market_open_ts =  pd.Timestamp("2019-06-18 09:30:00"),
                 market_close_ts = pd.Timestamp("2019-06-18 09:35:00"),
                 buy_price_range = [90, 105], sell_price_range = [95, 110], quantity_range = [50, 500],
                 seed=None):
        self.symbol            = symbol
        self.market_open_ts    = market_open_ts
        self.market_close_ts   = market_close_ts
        self.buy_price_range   = buy_price_range
        self.sell_price_range  = sell_price_range
        self.quantity_range    = quantity_range
        self.random_state      = np.random.RandomState(seed=seed)
        np.random.seed(seed)
        self.trades_df         = self.generateTradesDataframe()
        log_print("RandomOrderBookOracle initialized for {} and date: {}".format(self.symbol,
                                                                           str(market_open_ts.date())))

    def generateRandomTimestamps(self):
        start_timestamp = self.market_open_ts + pd.Timedelta('1ms')
        timestamp_list = []
        timestamp_list.append(start_timestamp)
        current_timestamp = start_timestamp
        while current_timestamp < self.market_close_ts:
            delta_time = self.random_state.exponential(scale=1.0 / 0.005)
            current_timestamp = current_timestamp + pd.Timedelta('{}ms'.format(int(round(delta_time))))
            timestamp_list.append(current_timestamp)
        del timestamp_list[-1]
        return timestamp_list

    def generateTradesDataframe(self):
        trades_df = pd.DataFrame(columns=['timestamp', 'type', 'order_id', 'vol', 'price', 'direction'])
        trades_df.timestamp = self.generateRandomTimestamps()
        trades_df.set_index('timestamp', inplace=True)
        trades_df.type = 'NEW'
        for index, row in trades_df.iterrows():
            row['order_id'] = RandomOrderBookOracle.order_id
            RandomOrderBookOracle.order_id += 1
            direction = np.random.randint(0, 2)
            row['direction'] = 'BUY' if direction == 1 else 'SELL'
            row['price'] = np.random.randint(self.buy_price_range[0], self.buy_price_range[1]) \
                if direction == 1 else np.random.randint(self.sell_price_range[0], self.sell_price_range[1])
            row['vol'] = np.random.randint(self.quantity_range[0], self.quantity_range[1])
        RandomOrderBookOracle.order_id = 0
        trades_df.reset_index(inplace=True)
        log_print("RandomOrderBookOracle generated with {} synthetic random trades".format(len(trades_df)))
        return trades_df

    def getDailyOpenPrice(self, symbol, mkt_open):
        price = self.trades_df.iloc[0]['price']
        log_print("Opening price at {} for {}".format(mkt_open, symbol))
        return price