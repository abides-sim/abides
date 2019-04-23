from agent.TradingAgent import TradingAgent
from message.Message import Message
from util.util import print

import numpy as np
import pandas as pd

class MomentumAgent(TradingAgent):

  def __init__(self, id, name, symbol, startingCash, lookback):
    # Base class init.
    super().__init__(id, name, startingCash)

    self.symbol = symbol
    self.lookback = lookback
    self.state = "AWAITING_WAKEUP"

    self.freq = '1m'
    self.trades = []


  def wakeup (self, currentTime):
    can_trade = super().wakeup(currentTime)

    if not can_trade: return

    self.getLastTrade(self.symbol)
    self.state = "AWAITING_LAST_TRADE"


  def receiveMessage (self, currentTime, msg):
    super().receiveMessage(currentTime, msg)

    if self.state == "AWAITING_LAST_TRADE" and msg.body['msg'] == "QUERY_LAST_TRADE":
      last = self.last_trade[self.symbol]

      self.trades = (self.trades + [last])[:self.lookback]

      if len(self.trades) >= self.lookback:

        m, b = np.polyfit(range(len(self.trades)), self.trades, 1)
        pred = self.lookback * m + b

        holdings = self.getHoldings(self.symbol)

        if pred > last:
          self.placeLimitOrder(self.symbol, 100-holdings, True, self.MKT_BUY)
        else:
          self.placeLimitOrder(self.symbol, 100+holdings, False, self.MKT_SELL)

      self.setWakeup(currentTime + pd.Timedelta(self.freq))
      self.state = 'AWAITING_WAKEUP'

  def getWakeFrequency (self):
    return pd.Timedelta(np.random.randint(low = 0, high = pd.Timedelta(self.freq) / np.timedelta64(1, 'ns')), unit='ns')

