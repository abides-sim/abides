from agent.TradingAgent import TradingAgent
import pandas as pd
from util.util import log_print
from collections import namedtuple, deque
from util.util import ignored


ANCHOR_TOP_STR = 'top'
ANCHOR_BOTTOM_STR = 'bottom'


class POVMarketMakerAgent(TradingAgent):
    """ This class implements a modification of the Chakraborty-Kearns `ladder` market-making strategy, wherein the
        the size of order placed at each level is set as a fraction of measured transacted volume in the previous time
        period.
    """

    def __init__(self, id, name, type, symbol, starting_cash, pov=0.05, min_order_size=20, window_size=5, anchor=ANCHOR_BOTTOM_STR,
                 num_ticks=20, wake_up_freq='1s', subscribe=False, subscribe_freq=10e9, subscribe_num_levels=1,
                 log_orders=False, random_state=None):

        super().__init__(id, name, type, starting_cash=starting_cash, log_orders=log_orders, random_state=random_state)
        self.symbol = symbol      # Symbol traded
        self.pov = pov  # fraction of transacted volume placed at each price level
        self.min_order_size = min_order_size  # minimum size order to place at each level, if pov <= min
        self.window_size = window_size  # Size in ticks (cents) of how wide the window around mid price is
        self.anchor = self.validateAnchor(anchor)  # anchor either top of window or bottom of window to mid-price
        self.num_ticks = num_ticks  # number of ticks on each side of window in which to place liquidity

        self.wake_up_freq = wake_up_freq  # Frequency of agent wake up
        self.subscribe = subscribe  # Flag to determine whether to subscribe to data or use polling mechanism
        self.subscribe_freq = subscribe_freq  # Frequency in nanoseconds^-1 at which to receive market updates
                                              # in subscribe mode
        self.subscribe_num_levels = subscribe_num_levels  # Number of orderbook levels in subscription mode
        self.log_orders = log_orders

        ## Internal variables

        self.subscription_requested = False
        self.state = self.initialiseState()
        self.order_size = self.min_order_size

        self.last_mid = None  # last observed mid price
        self.LIQUIDITY_DROPOUT_WARNING = f"Liquidity dropout for agent {self.name}."

    def initialiseState(self):
        """ Returns variables that keep track of whether spread and transacted volume have been observed. """
        if not self.subscribe:
            return {
                "AWAITING_SPREAD": True,
                "AWAITING_TRANSACTED_VOLUME": True
            }
        else:
            return {
                "AWAITING_MARKET_DATA": True,
                "AWAITING_TRANSACTED_VOLUME": True
            }

    def validateAnchor(self, anchor):
        """ Checks that input parameter anchor takes allowed value, raises ValueError if not.

        :param anchor: str
        :return:
        """
        if anchor not in [ANCHOR_TOP_STR, ANCHOR_BOTTOM_STR]:
            raise ValueError(f"Variable anchor must take the value `{ANCHOR_BOTTOM_STR}` or `{ANCHOR_TOP_STR}`")
        else:
            return anchor

    def kernelStarting(self, startTime):
        super().kernelStarting(startTime)

    def wakeup(self, currentTime):
        """ Agent wakeup is determined by self.wake_up_freq """
        can_trade = super().wakeup(currentTime)
        if self.subscribe and not self.subscription_requested:
            super().requestDataSubscription(self.symbol, levels=self.subscribe_num_levels,
                                            freq=pd.Timedelta(self.subscribe_freq, unit='ns'))
            self.subscription_requested = True
            self.get_transacted_volume(self.symbol, lookback_period=self.subscribe_freq)
            self.state = self.initialiseState()

        elif can_trade and not self.subscribe:
            self.getCurrentSpread(self.symbol, depth=self.subscribe_num_levels)
            self.get_transacted_volume(self.symbol, lookback_period=self.wake_up_freq)
            self.initialiseState()

    def receiveMessage(self, currentTime, msg):
        """ Processes message from exchange. Main function is to update orders in orderbook relative to mid-price.

        :param simulation current time
        :param message received by self from ExchangeAgent

        :type currentTime: pd.Timestamp
        :type msg: str

        :return:
        """

        super().receiveMessage(currentTime, msg)

        if self.last_mid is not None:
            mid = self.last_mid

        if msg.body['msg'] == 'QUERY_TRANSACTED_VOLUME' and self.state['AWAITING_TRANSACTED_VOLUME'] is True:
            self.updateOrderSize()
            self.state['AWAITING_TRANSACTED_VOLUME'] = False

        if not self.subscribe:
            if msg.body['msg'] == 'QUERY_SPREAD' and self.state['AWAITING_SPREAD'] is True:
                bid, _, ask, _ = self.getKnownBidAsk(self.symbol)
                if bid and ask:
                    mid = int((ask + bid) / 2)
                    self.last_mid = mid
                    self.state['AWAITING_SPREAD'] = False
                else:
                    log_print(f"SPREAD MISSING at time {currentTime}")

            if self.state['AWAITING_SPREAD'] is False and self.state['AWAITING_TRANSACTED_VOLUME'] is False:
                self.cancelAllOrders()
                self.placeOrders(mid)
                self.state = self.initialiseState()
                self.setWakeup(currentTime + self.getWakeFrequency())

        else:  # subscription mode
            if msg.body['msg'] == 'MARKET_DATA' and self.state['AWAITING_MARKET_DATA'] is True:
                bid = self.known_bids[self.symbol][0][0] if self.known_bids[self.symbol] else None
                ask = self.known_asks[self.symbol][0][0] if self.known_asks[self.symbol] else None
                if bid and ask:
                    mid = int((ask + bid) / 2)
                    self.last_mid = mid
                    self.state['AWAITING_MARKET_DATA'] = False
                else:
                    log_print(f"SPREAD MISSING at time {currentTime}")
                    self.state['AWAITING_MARKET_DATA'] = False

            if self.state['MARKET_DATA'] is False and self.state['AWAITING_TRANSACTED_VOLUME'] is False:
                self.placeOrders(mid)
                self.state = self.initialiseState()

    def updateOrderSize(self):
        """ Updates size of order to be placed. """
        qty = round(self.pov * self.transacted_volume[self.symbol])
        self.order_size = qty if qty >= self.min_order_size else self.min_order_size

    def computeOrdersToPlace(self, mid):
        """ Given a mid price, computes the orders that need to be removed from orderbook, and adds these orders to
            bid and ask deques.

        :param mid: mid-price
        :type mid: int

        :return:
        """

        if self.anchor == ANCHOR_BOTTOM_STR:
            highest_bid = int(mid - 1)
            lowest_ask = int(mid + self.window_size)
        elif self.anchor == ANCHOR_TOP_STR:
            highest_bid = int(mid - self.window_size)
            lowest_ask = int(mid + 1)

        lowest_bid = highest_bid - self.num_ticks
        highest_ask = lowest_ask + self.num_ticks

        bids_to_place = [price for price in range(lowest_bid, highest_bid + 1)]
        asks_to_place = [price for price in range(lowest_ask, highest_ask + 1)]

        return bids_to_place, asks_to_place

    def placeOrders(self, mid):
        """ Given a mid-price, compute new orders that need to be placed, then send the orders to the Exchange.

            :param mid: mid-price
            :type mid: int

        """

        bid_orders, ask_orders = self.computeOrdersToPlace(mid)
        for bid_price in bid_orders:
            log_print(f'{self.name}: Placing BUY limit order of size {self.order_size} @ price {bid_price}')
            self.placeLimitOrder(self.symbol, self.order_size, True, bid_price)

        for ask_price in ask_orders:
            log_print(f'{self.name}: Placing SELL limit order of size {self.order_size} @ price {ask_price}')
            self.placeLimitOrder(self.symbol, self.order_size, False, ask_price)

    def getWakeFrequency(self):
        """ Get time increment corresponding to wakeup period. """
        return pd.Timedelta(self.wake_up_freq)

    def cancelAllOrders(self):
        """ Cancels all resting limit orders placed by the market maker """
        for _, order in self.orders.items():
            self.cancelOrder(order)
