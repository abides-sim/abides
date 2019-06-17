import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

from util.util import log_print, delist

class OrderBookOracle:


    def __init__(self, symbol, date, orderbook_file_path, message_file_path, num_price_levels=10, filter_trades=False):
        self.symbol              = symbol
        self.date                = date
        self.num_price_levels    =  num_price_levels
        self.message_df          = self.readMessageFile(message_file_path)
        self.orderbook_df        = self.readOrderbookFile(orderbook_file_path)
        self.trades_df           = self.filter_trades() if filter_trades else self.message_df
        log_print("OrderBookOracle initialized for {} and date: {}".format(self.symbol, self.date))


    def readMessageFile(self, message_file_path):
        """
        :return: a pandas Dataframe of the trade messages file for the given symbol and date
        """
        log_print("OrderBookOracle Message File: {}".format(message_file_path))

        direction = {-1: 'SELL',
                     1: 'BUY'}

        order_type = {
            1: 'NEW',
            2: 'PARTIAL_CANCELLATION',
            3: 'CANCELLATION',
            4: 'EXECUTE_VISIBLE',
            5: 'EXECUTE_HIDDEN',
            7: 'TRADING_HALT'
        }

        message_df = pd.read_csv(message_file_path)
        message_df.columns = ['timestamp', 'type', 'order_id', 'vol', 'price', 'direction']
        message_df['timestamp'] = pd.to_datetime(self.date) + pd.to_timedelta(message_df['timestamp'], unit='s')
        message_df['direction'] = message_df['direction'].replace(direction)
        message_df['price'] = message_df['price'] / 10000
        message_df['type'] = message_df['type'].replace(order_type)
        return message_df


    def readOrderbookFile(self, orderbook_file_path):
        """
        :return: a pandas Dataframe of the orderbook file for the given symbol and date
        """
        log_print("OrderBookOracle Orderbook File: {}".format(orderbook_file_path))
        all_cols = delist([[f"ask_price_{level}", f"ask_size_{level}", f"bid_price_{level}", f"bid_size_{level}"] for level in range(1, self.num_price_levels+1)])
        price_cols = delist([[f"ask_price_{level}", f"bid_price_{level}"] for level in range(1, self.num_price_levels+1)])
        orderbook_df = pd.read_csv(orderbook_file_path)
        orderbook_df.columns = all_cols
        orderbook_df[price_cols] = orderbook_df[price_cols] / 10000
        orderbook_df = orderbook_df.join(self.message_df[['timestamp']])
        orderbook_df = orderbook_df[['timestamp'] + all_cols]
        #orderbook_df = orderbook_df.drop_duplicates(subset=['timestamp'], keep='last')
        orderbook_df.set_index('timestamp', inplace=True)
        return orderbook_df


    def bids(self):
        """
        :return: bid side of the orderbook (pandas dataframe)
        """
        orderbook_bid_cols = delist([[f"bid_price_{level}", f"bid_size_{level}"] for level in range(1, self.num_price_levels+1)])
        return self.orderbook_df[orderbook_bid_cols]


    def asks(self):
        """
        :return: ask side of the orderbook (pandas dataframe)
        """
        orderbook_ask_cols = delist([[f"ask_price_{level}", f"ask_size_{level}"] for level in range(1, self.num_price_levels+1)])
        return self.orderbook_df[orderbook_ask_cols]


    def orderbook_snapshot(self, t=None):
        """
        :return: orderbook snapshot for a given timestamp (pandas dataframe)
        """
        log_print(f"Orderbook snapshot @ t= {t}")
        orderbook_snapshot = pd.DataFrame(columns=['bid_size', 'bid', 'ask', 'ask_size'], index=range(1, self.num_price_levels+1))
        bids = self.bids().loc[t]
        asks = self.asks().loc[t]
        level = 1
        for i in range(0, len(asks), 2):
            bid_price = bids.iloc[i]
            bid_size = bids.iloc[i + 1]
            ask_price = asks.iloc[i]
            ask_size = asks.iloc[i + 1]
            orderbook_snapshot.loc[level] = [bid_size, bid_price, ask_price, ask_size]
            level += 1
        return orderbook_snapshot


    @staticmethod
    def bestBid(ob_snap_t):
        """Return int
        best bid price for a given orderbook snapshot
        """
        return ob_snap_t.loc[1]['bid']


    @staticmethod
    def bestAsk(ob_snap_t):
        """Return int

        best ask price for a given orderbook snapshot
        """
        return ob_snap_t.loc[1]['ask']


    @staticmethod
    def bestBidSize(ob_snap_t):
        """Return int

        best bid size (volume) for a given orderbook snapshot
        """
        return ob_snap_t.loc[1]['bid_size']


    @staticmethod
    def bestAskSize(ob_snap_t):
        """Return int

        best ask size (volume) for a given orderbook snapshot
        """
        return ob_snap_t.loc[1]['ask_size']


    def midPrice(self, ob_snap_t):
        """Return int

        mid price for a given orderbook snapshot
        """
        return (self.bestBid(ob_snap_t) + self.bestAsk(ob_snap_t)) / 2


    def spread(self, ob_snap_t):
        """Return int

        spread for a given orderbook snapshot
        """
        return self.bestAsk(ob_snap_t) - self.bestBid(ob_snap_t)


    def plotOrderbookSnapshotMetrics(self, t, ob_snap_t):
        """
        at t, plot against l (x-axis): Pb, Pa, Sb, Sa, Pa+Pb/2, Pa-Pb, Pa+Pb
        :param t: timestamp of the orderbook snapshot
        :param ob_snap_t: orderbook snapshot dataframe
        :return: None
        """
        fig, axes = plt.subplots(nrows=2, ncols=3)
        fig.set_size_inches(30, 10)
        fig.suptitle(f"{self.symbol} Orderbook snapshot metrics @ {t}", size=20)

        fig.text(0.05, 0.95,
                 f"Best Bid: {self.bestBid(ob_snap_t)}, Best Ask: {self.bestAsk(ob_snap_t)}, Mid Price: {self.midPrice(ob_snap_t)}, "
                 f"Spread: {self.spread(ob_snap_t)}, Best Bid Size: {self.bestBidSize(ob_snap_t)}, Best Ask Size: {self.bestAskSize(ob_snap_t)}",
                 fontsize=14, verticalalignment='top')
        axes[0, 0].plot(ob_snap_t.index, ob_snap_t.bid)
        axes[0, 0].set_ylabel("Bid Price ( $Pb$ )", size=13)

        axes[0, 1].plot(ob_snap_t.index, ob_snap_t.ask)
        axes[0, 1].set_ylabel("Ask Price ( $Pa$ )", size=13)

        axes[0, 2].plot(ob_snap_t.index, ob_snap_t.ask_size)
        axes[0, 2].plot(ob_snap_t.index, ob_snap_t.bid_size)
        axes[0, 2].set_ylabel("Bid nd Ask Sizes ( $Sb, Sa$ )", size=13)
        axes[0, 2].legend()

        axes[1, 0].plot(ob_snap_t.index, ((ob_snap_t.ask + ob_snap_t.bid) / 2))
        axes[1, 0].set_ylabel("Mid Price ( $(Pa+Pb) / 2$ )", size=13)

        axes[1, 1].plot(ob_snap_t.index, (ob_snap_t.ask - ob_snap_t.bid))
        axes[1, 1].set_ylabel("Spread ( $Pa-Pb$ )", size=13)

        axes[1, 2].plot(ob_snap_t.index, (ob_snap_t.ask + ob_snap_t.bid))
        axes[1, 2].set_ylabel("$Pa + Pb$", size=13)

        for ax in axes:
            for in_ax in ax:
                in_ax.set_xlabel("Price Level", size=13)


    def plotDepth(self, t, ob_snap_t):
        """
        plots the orderbook depth for the given snapshot
        :param t: timestamp of the orderbook snapshot
        :param ob_snap_t: orderbook snapshot dataframe
        :return: None
        """
        fig, axes = plt.subplots(nrows=1, ncols=1)
        fig.set_size_inches(20, 5)
        axes.set_title(f"Orderbook Depth chart for {self.symbol} @ {t}")
        axes.set_xlabel("Price ($)")
        axes.set_ylabel("Cumulative Volume")

        plt.plot(ob_snap_t.bid, ob_snap_t.bid_size.cumsum(), color='green', marker='o')
        axes.fill_between(ob_snap_t.bid.values.astype(float), 0,
                          ob_snap_t.bid_size.cumsum().values.astype(int), color='palegreen')
        plt.bar(ob_snap_t.bid, ob_snap_t.bid_size, width=[0.01] * 10, color='grey')

        plt.plot(ob_snap_t.ask, ob_snap_t.ask_size.cumsum(), color='red', marker='o')
        axes.fill_between(ob_snap_t.ask.values.astype(float), 0,
                          ob_snap_t.ask_size.cumsum().values.astype(int), color='salmon')
        plt.bar(ob_snap_t.ask, ob_snap_t.ask_size, width=[0.01] * 10, color='grey', label='volume')

        plt.axvline(x=self.midPrice(ob_snap_t), label='mid')
        plt.legend()


    def plotPriceLevelVolume(self, orderbook_df):
        """
        plot the price level coloured by volumes available at each level
        :param orderbook_df:
        :return: None
        """

        price_cols = delist([[f"ask_price_{level}", f"bid_price_{level}"] for level in range(1, self.num_price_levels+1)])
        size_cols = delist([[f"ask_size_{level}", f"bid_size_{level}"] for level in range(1, self.num_price_levels+1)])
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.set_size_inches(30, 15)
        ax.set_title(f"Orderbook Price Level Volume for {self.symbol}, {self.num_price_levels} levels", size=22)
        ax.set_xlabel("Time", size=24, fontweight='bold')
        ax.set_ylabel("Price ($)", size=24, fontweight='bold')
        ax.set_facecolor("white")

        mid_price = (orderbook_df.ask_price_1 + orderbook_df.bid_price_1) / 2

        myFmt = DateFormatter("%H:%M")
        ax.xaxis.set_major_formatter(myFmt)
        ax.plot(orderbook_df.index, mid_price, color='black', label='mid price')

        for price_col, size_col in zip(price_cols, size_cols):
            im = ax.scatter(x=orderbook_df.index, y=orderbook_df[price_col], c=np.log(orderbook_df[size_col]), s=0.7,
                            cmap=plt.cm.jet, alpha=0.7)
        cbar = fig.colorbar(im, ax=ax, label='volume')
        cbar.ax.get_yaxis().labelpad = 20
        cbar.ax.set_ylabel('Size', rotation=270, fontsize=20, fontweight='bold')


    def filter_trades(self):
        log_print("Original trades type counts:")
        log_print(self.message_df.type.value_counts())
        trades_df = self.message_df.loc[self.message_df.type.isin(['NEW', 'CANCELLATION', 'PARTIAL_CANCELLATION', 'EXECUTE_VISIBLE'])]
        order_id_types_series = trades_df.groupby('order_id')['type'].apply(list)
        order_id_types_series = order_id_types_series.apply(lambda x: str(x))
        cancel_only_order_ids = list(order_id_types_series[order_id_types_series == "['CANCELLATION']"].index)
        part_cancel_only_order_ids = list(order_id_types_series[order_id_types_series == "['PARTIAL_CANCELLATION']"].index)
        trades_df = trades_df.loc[~trades_df.order_id.isin(cancel_only_order_ids + part_cancel_only_order_ids)]
        log_print("Filtered trades type counts:")
        log_print(trades_df.type.value_counts())
        return trades_df


    def getDailyOpenPrice(self, symbol, mkt_open):
        price = self.message_df.iloc[0]['price']
        log_print("Opening price at {} for {}".format(mkt_open, symbol))
        return price

    def observePrice(self, symbol, currentTime, sigma_n = 0):
        return self.message_df.iloc[0]['price']

