from agent.TradingAgent import TradingAgent
from util.util import log_print

import pandas as pd


class SubscriptionAgent(TradingAgent):
    """
    Simple agent to demonstrate subscription to order book market data.
    """

    def __init__(self, id, name, type, symbol, starting_cash, levels, freq, log_orders=False, random_state=None):
        super().__init__(id, name, type, starting_cash=starting_cash, log_orders=log_orders, random_state=random_state)
        self.symbol = symbol  # symbol traded
        self.levels = levels  # number of price levels to subscribe to/recieve updates for
        self.freq = freq  # minimum number of nanoseconds between market data messages
        self.subscribe = True  # Flag to determine whether to subscribe to data or use polling mechanism
        self.subscription_requested = False
        self.last_update_ts = None  # timestamp of the last agent update.
                                    # This is NOT required but only used to demonstrate how subscription works
        self.state = 'AWAITING_MARKET_DATA'

    def kernelStarting(self, startTime):
        super().kernelStarting(startTime)

    def wakeup(self, currentTime):
        super().wakeup(currentTime)
        if self.subscribe and not self.subscription_requested:
            super().requestDataSubscription(self.symbol, levels=self.levels, freq=self.freq)
            self.subscription_requested = True
            self.last_update_ts = currentTime

    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)
        if self.subscribe and self.state == 'AWAITING_MARKET_DATA' and msg.body['msg'] == 'MARKET_DATA':
            bids, asks = msg.body['bids'], msg.body['asks']
            print("--------------------")
            print("seconds elapsed since last update: {}".format((currentTime - self.last_update_ts).delta / 1e9))
            print("number of bid levels: {}".format(len(bids)))
            print("number of ask levels: {}".format(len(asks)))
            print("bids: {}, asks: {}".format(bids, asks))
            print("Current Agent Timestamp: {}".format(currentTime))
            print("Exchange Timestamp: {}".format(self.exchange_ts[self.symbol]))
            print("--------------------")
            self.last_update_ts = currentTime

    def getWakeFrequency(self):
        return pd.Timedelta('1s')