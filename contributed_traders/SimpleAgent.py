from agent.TradingAgent import TradingAgent
import pandas as pd
import numpy as np
import os
from contributed_traders.util import get_file

class SimpleAgent(TradingAgent):
    """
    Simple Trading Agent that compares the past mid-price observations and places a
    buy limit order if the first window mid-price exponential average >= the second window mid-price exponential average or a
    sell limit order if the first window mid-price exponential average < the second window mid-price exponential average
    """

    def __init__(self, id, name, type, symbol, starting_cash,
                 min_size, max_size, wake_up_freq='60s',
                 log_orders=False, random_state=None):

        super().__init__(id, name, type, starting_cash=starting_cash, log_orders=log_orders, random_state=random_state)
        self.symbol = symbol
        self.min_size = min_size  # Minimum order size
        self.max_size = max_size  # Maximum order size
        self.size = self.random_state.randint(self.min_size, self.max_size)
        self.wake_up_freq = wake_up_freq
        self.mid_list, self.avg_win1_list, self.avg_win2_list = [], [], []
        self.log_orders = log_orders
        self.state = "AWAITING_WAKEUP"
        #self.window1 = 100 
        #self.window2 = 5 

    def kernelStarting(self, startTime):
        super().kernelStarting(startTime)
        # Read in the configuration through util
        with open(get_file('simple_agent.cfg'), 'r') as f:
            self.window1, self.window2 = [int(w) for w in f.readline().split()]
        #print(f"{self.window1} {self.window2}")

    def wakeup(self, currentTime):
        """ Agent wakeup is determined by self.wake_up_freq """
        can_trade = super().wakeup(currentTime)
        if not can_trade: return
        self.getCurrentSpread(self.symbol)
        self.state = 'AWAITING_SPREAD'

    def dump_shares(self):
        # get rid of any outstanding shares we have
        if self.symbol in self.holdings and len(self.orders) == 0:
            order_size = self.holdings[self.symbol]
            bid, _, ask, _ = self.getKnownBidAsk(self.symbol)
            if bid:
                self.placeLimitOrder(self.symbol, quantity=order_size, is_buy_order=False, limit_price=0)

    def receiveMessage(self, currentTime, msg):
        """ Momentum agent actions are determined after obtaining the best bid and ask in the LOB """
        super().receiveMessage(currentTime, msg)
        if self.state == 'AWAITING_SPREAD' and msg.body['msg'] == 'QUERY_SPREAD':
            dt = (self.mkt_close - currentTime) / np.timedelta64(1, 'm')
            if dt < 25:
                self.dump_shares()
            else:
                bid, _, ask, _ = self.getKnownBidAsk(self.symbol)
                if bid and ask:
                    self.mid_list.append((bid + ask) / 2)
                    if len(self.mid_list) > self.window1: self.avg_win1_list.append(pd.Series(self.mid_list).ewm(span=self.window1).mean().values[-1].round(2))
                    if len(self.mid_list) > self.window2: self.avg_win2_list.append(pd.Series(self.mid_list).ewm(span=self.window2).mean().values[-1].round(2))
                    if len(self.avg_win1_list) > 0 and len(self.avg_win2_list) > 0 and len(self.orders) == 0:
                        if self.avg_win1_list[-1] >= self.avg_win2_list[-1]:
                            # Check that we have enough cash to place the order
                            if self.holdings['CASH'] >= (self.size * ask):
                                self.placeLimitOrder(self.symbol, quantity=self.size, is_buy_order=True, limit_price=ask)
                        else:
                            if self.symbol in self.holdings and self.holdings[self.symbol] > 0:
                                order_size = min(self.size, self.holdings[self.symbol])
                                self.placeLimitOrder(self.symbol, quantity=order_size, is_buy_order=False, limit_price=bid)
            self.setWakeup(currentTime + self.getWakeFrequency())
            self.state = 'AWAITING_WAKEUP'

    def getWakeFrequency(self):
        return pd.Timedelta(self.wake_up_freq)
