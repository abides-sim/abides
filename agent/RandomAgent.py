from agent.TradingAgent import TradingAgent
import numpy as np
import pandas as pd


class RandomAgent(TradingAgent):


    def __init__(self, id, name, symbol, startingCash,
                 buy_price_range = [90, 105], sell_price_range = [95, 110], quantity_range = [50, 500],
                 random_state = None):
        super().__init__(id, name, startingCash, random_state)
        self.symbol           = symbol
        self.buy_price_range  = buy_price_range
        self.sell_price_range = sell_price_range
        self.quantity_range   = quantity_range


    def kernelStarting(self, startTime):
        super().kernelStarting(startTime)


    def wakeup(self, currentTime):
        super().wakeup(currentTime)
        self.last_trade[self.symbol] = 0
        if not self.mkt_open or not self.mkt_close:
            return
        elif (currentTime > self.mkt_open) and (currentTime < self.mkt_close):
            direction = np.random.randint(0, 2)
            price = np.random.randint(self.buy_price_range[0], self.buy_price_range[1]) \
                if direction == 1 else np.random.randint(self.sell_price_range[0], self.sell_price_range[1])
            quantity = np.random.randint(self.quantity_range[0], self.quantity_range[1])
            self.placeLimitOrder(self.symbol, quantity, direction, price, dollar=False)
        delta_time = self.random_state.exponential(scale=1.0 / 0.005)
        self.setWakeup(currentTime + pd.Timedelta('{}ms'.format(int(round(delta_time)))))


    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)


    def getWakeFrequency(self):
        return pd.Timedelta('1ms')
