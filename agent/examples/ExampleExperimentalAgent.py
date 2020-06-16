from agent.examples.SubscriptionAgent import SubscriptionAgent
import pandas as pd
from copy import deepcopy


class ExampleExperimentalAgentTemplate(SubscriptionAgent):
    """ Minimal working template for an experimental trading agent
    """
    def __init__(self, id, name, type, symbol, starting_cash, levels, subscription_freq, log_orders=False, random_state=None):
        """  Constructor for ExampleExperimentalAgentTemplate.

        :param id: Agent's ID as set in config
        :param name: Agent's human-readable name as set in config
        :param type: Agent's human-readable type as set in config, useful for grouping agents semantically
        :param symbol: Name of asset being traded
        :param starting_cash: Dollar amount of cash agent starts with.
        :param levels: Number of levels of orderbook to subscribe to
        :param subscription_freq: Frequency of orderbook updates subscribed to (in nanoseconds)
        :param log_orders: bool to decide if agent's individual actions logged to file.
        :param random_state: numpy RandomState object from which agent derives randomness
        """
        super().__init__(id, name, type, symbol, starting_cash, levels, subscription_freq, log_orders=log_orders, random_state=random_state)

        self.current_bids = None  # subscription to market data populates this list
        self.current_asks = None  # subscription to market data populates this list

    def wakeup(self, currentTime):
        """ Action to be taken by agent at each wakeup.

            :param currentTime: pd.Timestamp for current simulation time
        """
        super().wakeup(currentTime)
        self.setWakeup(currentTime + self.getWakeFrequency())

    def receiveMessage(self, currentTime, msg):
        """ Action taken when agent receives a message from the exchange

        :param currentTime: pd.Timestamp for current simulation time
        :param msg: message from exchange
        :return:
        """
        super().receiveMessage(currentTime, msg)  # receives subscription market data

    def getWakeFrequency(self):
        """ Set next wakeup time for agent. """
        return pd.Timedelta("1min")

    def placeLimitOrder(self, quantity, is_buy_order, limit_price):
        """ Place a limit order at the exchange.
          :param quantity (int):      order quantity
          :param is_buy_order (bool): True if Buy else False
          :param limit_price: price level at which to place a limit order
          :return:
        """
        super().placeLimitOrder(self.symbol, quantity, is_buy_order, limit_price)

    def placeMarketOrder(self, quantity, is_buy_order):
        """ Place a market order at the exchange.
          :param quantity (int):      order quantity
          :param is_buy_order (bool): True if Buy else False
          :return:
        """
        super().placeMarketOrder(self.symbol, quantity, is_buy_order)

    def cancelAllOrders(self):
        """ Cancels all resting limit orders placed by the experimental agent.
        """
        for _, order in self.orders.items():
            self.cancelOrder(order)


class ExampleExperimentalAgent(ExampleExperimentalAgentTemplate):

    def __init__(self, *args, wake_freq, order_size, short_window, long_window, **kwargs):
        """
        :param args: superclass args
        :param wake_freq: Frequency of wakeup -- str to be parsed by pd.Timedelta
        :param order_size: size of orders to place
        :param short_window: length of mid price short moving average window -- str to be parsed by pd.Timedelta
        :param long_window: length of mid price long moving average window -- str to be parsed by pd.Timedelta
        :param kwargs: superclass kwargs
        """
        super().__init__(*args, **kwargs)
        self.wake_freq = wake_freq
        self.order_size = order_size
        self.short_window = short_window
        self.long_window = long_window
        self.mid_price_history = pd.DataFrame(columns=['mid_price'], index=pd.to_datetime([]))

    def getCurrentMidPrice(self):
        """ Retrieve mid price from most recent subscription data.

        :return:
        """

        try:
            best_bid = self.current_bids[0][0]
            best_ask = self.current_asks[0][0]
            return round((best_ask + best_bid) / 2)
        except (TypeError, IndexError):
            return None

    def receiveMessage(self, currentTime, msg):
        """ Action taken when agent receives a message from the exchange -- action here is for agent to update internal
            log of most recently observed mid-price.

        :param currentTime: pd.Timestamp for current simulation time
        :param msg: message from exchange
        :return:
        """
        super().receiveMessage(currentTime, msg)  # receives subscription market data
        self.mid_price_history = self.mid_price_history.append(
            pd.Series({'mid_price': self.getCurrentMidPrice()}, name=currentTime))
        self.mid_price_history.dropna(inplace=True)

    def computeMidPriceMovingAverages(self):
        """ Returns the short-window and long-window moving averages of mid price.
        :return:
        """
        try:
            short_moving_avg = self.mid_price_history.rolling(self.short_window).mean().iloc[-1]['mid_price']
            long_moving_avg = self.mid_price_history.rolling(self.long_window).mean().iloc[-1]['mid_price']
            return short_moving_avg, long_moving_avg
        except IndexError:
            return None, None

    def wakeup(self, currentTime):
        """ Action to be taken by agent at each wakeup.

            :param currentTime: pd.Timestamp for current simulation time
        """
        super().wakeup(currentTime)
        short_moving_avg, long_moving_avg = self.computeMidPriceMovingAverages()
        if short_moving_avg is not None and long_moving_avg is not None:
            if short_moving_avg > long_moving_avg:
                self.placeMarketOrder(self.order_size, 0)
            elif short_moving_avg < long_moving_avg:
                self.placeMarketOrder(self.order_size, 1)

    def getWakeFrequency(self):
        """ Set next wakeup time for agent. """
        return pd.Timedelta(self.wake_freq)




