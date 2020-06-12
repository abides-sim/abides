from agent.TradingAgent import TradingAgent
import pandas as pd
from util.util import log_print
from collections import namedtuple, deque
from util.util import ignored, sigmoid
from math import floor, ceil


ANCHOR_TOP_STR = 'top'
ANCHOR_BOTTOM_STR = 'bottom'
ANCHOR_MIDDLE_STR = 'middle'

ADAPTIVE_SPREAD_STR = 'adaptive'
INITIAL_SPREAD_VALUE = 50


class AdaptiveMarketMakerAgent(TradingAgent):
    """ This class implements a modification of the Chakraborty-Kearns `ladder` market-making strategy, wherein the
        the size of order placed at each level is set as a fraction of measured transacted volume in the previous time
        period.

        Can skew orders to size of current inventory using beta parameter, whence beta == 0 represents inventory being
        ignored and beta == infinity represents all liquidity placed on one side of book.

        ADAPTIVE SPREAD: the market maker's spread can be set either as a fixed or value or can be adaptive,

    """

    def __init__(self, id, name, type, symbol, starting_cash, pov=0.05, min_order_size=20, window_size=5, anchor=ANCHOR_MIDDLE_STR,
                 num_ticks=20, level_spacing=0.5, wake_up_freq='1s', subscribe=False, subscribe_freq=10e9, subscribe_num_levels=1, cancel_limit_delay=50,
                 skew_beta=0, spread_alpha=0.85, backstop_quantity=None, log_orders=False, random_state=None):

        super().__init__(id, name, type, starting_cash=starting_cash, log_orders=log_orders, random_state=random_state)
        self.is_adaptive = False
        self.symbol = symbol      # Symbol traded
        self.pov = pov  # fraction of transacted volume placed at each price level
        self.min_order_size = min_order_size  # minimum size order to place at each level, if pov <= min
        self.anchor = self.validateAnchor(anchor)  # anchor either top of window or bottom of window to mid-price
        self.window_size = self.validateWindowSize(window_size)  # Size in ticks (cents) of how wide the window around mid price is. If equal to
                                                                # string 'adaptive' then ladder starts at best bid and ask
        self.num_ticks = num_ticks  # number of ticks on each side of window in which to place liquidity
        self.level_spacing = level_spacing  #  level spacing as a fraction of the spread
        self.wake_up_freq = wake_up_freq  # Frequency of agent wake up
        self.subscribe = subscribe  # Flag to determine whether to subscribe to data or use polling mechanism
        self.subscribe_freq = subscribe_freq  # Frequency in nanoseconds^-1 at which to receive market updates
                                              # in subscribe mode
        self.subscribe_num_levels = subscribe_num_levels  # Number of orderbook levels in subscription mode
        self.cancel_limit_delay  = cancel_limit_delay  # delay in nanoseconds between order cancellations and new limit order placements

        self.skew_beta = skew_beta  # parameter for determining order placement imbalance
        self.spread_alpha = spread_alpha  # parameter for exponentially weighted moving average of spread. 1 corresponds to ignoring old values, 0 corresponds to no updates
        self.backstop_quantity = backstop_quantity  # how many orders to place at outside order level, to prevent liquidity dropouts. If None then place same as at other levels.
        self.log_orders = log_orders

        ## Internal variables

        self.subscription_requested = False
        self.state = self.initialiseState()
        self.buy_order_size = self.min_order_size
        self.sell_order_size = self.min_order_size

        self.last_mid = None  # last observed mid price
        self.last_spread = INITIAL_SPREAD_VALUE  # last observed spread moving average
        self.tick_size = None if self.is_adaptive else ceil(self.last_spread * self.level_spacing)
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
        if anchor not in [ANCHOR_TOP_STR, ANCHOR_BOTTOM_STR, ANCHOR_MIDDLE_STR]:
            raise ValueError(f"Variable anchor must take the value `{ANCHOR_BOTTOM_STR}`, `{ANCHOR_MIDDLE_STR}` or "
                             f"`{ANCHOR_TOP_STR}`")
        else:
            return anchor

    def validateWindowSize(self, window_size):
        """ Checks that input parameter window_size takes allowed value, raises ValueError if not

        :param window_size:
        :return:
        """
        try:  # fixed window size specified
            return int(window_size)
        except:
            if window_size.lower() == 'adaptive':
                self.is_adaptive = True
                self.anchor = ANCHOR_MIDDLE_STR
                return None
            else:
                raise ValueError(f"Variable window_size must be of type int or string {ADAPTIVE_SPREAD_STR}.")

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
            self.cancelAllOrders()
            self.delay(self.cancel_limit_delay)
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

        if self.last_spread is not None and self.is_adaptive:
            self._adaptive_update_window_and_tick_size()

        if msg.body['msg'] == 'QUERY_TRANSACTED_VOLUME' and self.state['AWAITING_TRANSACTED_VOLUME'] is True:
            self.updateOrderSize()
            self.state['AWAITING_TRANSACTED_VOLUME'] = False

        if not self.subscribe:
            if msg.body['msg'] == 'QUERY_SPREAD' and self.state['AWAITING_SPREAD'] is True:
                bid, _, ask, _ = self.getKnownBidAsk(self.symbol)
                if bid and ask:
                    mid = int((ask + bid) / 2)
                    self.last_mid = mid
                    if self.is_adaptive:
                        spread = int(ask - bid)
                        self._adaptive_update_spread(spread)

                    self.state['AWAITING_SPREAD'] = False
                else:
                    log_print("SPREAD MISSING at time {}", currentTime)
                    self.state['AWAITING_SPREAD'] = False  # use last mid price and spread

            if self.state['AWAITING_SPREAD'] is False and self.state['AWAITING_TRANSACTED_VOLUME'] is False:
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
                    if self.is_adaptive:
                        spread = int(ask - bid)
                        self._adaptive_update_spread(spread)

                    self.state['AWAITING_MARKET_DATA'] = False
                else:
                    log_print("SPREAD MISSING at time {}", currentTime)
                    self.state['AWAITING_MARKET_DATA'] = False

            if self.state['MARKET_DATA'] is False and self.state['AWAITING_TRANSACTED_VOLUME'] is False:
                self.placeOrders(mid)
                self.state = self.initialiseState()

    def _adaptive_update_spread(self, spread):
        """ Update internal spread estimate with exponentially weighted moving average
        :param spread:
        :return:
        """
        spread_ewma = self.spread_alpha * spread + (1 - self.spread_alpha) * self.last_spread
        self.window_size = spread_ewma
        self.last_spread = spread_ewma

    def _adaptive_update_window_and_tick_size(self):
        """ Update window size and tick size relative to internal spread estimate.

        :return:
        """
        self.window_size = self.last_spread
        self.tick_size = round(self.level_spacing * self.window_size)
        if self.tick_size == 0:
            self.tick_size = 1

    def updateOrderSize(self):
        """ Updates size of order to be placed. """
        qty = round(self.pov * self.transacted_volume[self.symbol])
        if self.skew_beta == 0:  # ignore inventory
            self.buy_order_size = qty if qty >= self.min_order_size else self.min_order_size
            self.sell_order_size = qty if qty >= self.min_order_size else self.min_order_size
        else:
            holdings = self.getHoldings(self.symbol)
            proportion_sell = sigmoid(holdings, self.skew_beta)
            sell_size = ceil(proportion_sell * qty)
            buy_size = floor((1 - proportion_sell) * qty)

            self.buy_order_size = buy_size if buy_size >= self.min_order_size else self.min_order_size
            self.sell_order_size = sell_size if sell_size >= self.min_order_size else self.min_order_size

    def computeOrdersToPlace(self, mid):
        """ Given a mid price, computes the orders that need to be removed from orderbook, and adds these orders to
            bid and ask deques.

        :param mid: mid-price
        :type mid: int

        :return:
        """

        if self.anchor == ANCHOR_MIDDLE_STR:
            highest_bid = int(mid) - floor(0.5 * self.window_size)
            lowest_ask = int(mid) + ceil(0.5 * self.window_size)
        elif self.anchor == ANCHOR_BOTTOM_STR:
            highest_bid = int(mid - 1)
            lowest_ask = int(mid + self.window_size)
        elif self.anchor == ANCHOR_TOP_STR:
            highest_bid = int(mid - self.window_size)
            lowest_ask = int(mid + 1)

        lowest_bid = highest_bid - ((self.num_ticks - 1) * self.tick_size)
        highest_ask = lowest_ask + ((self.num_ticks - 1) * self.tick_size)

        bids_to_place = [price for price in range(lowest_bid, highest_bid + self.tick_size, self.tick_size)]
        asks_to_place = [price for price in range(lowest_ask, highest_ask + self.tick_size, self.tick_size)]

        return bids_to_place, asks_to_place

    def placeOrders(self, mid):
        """ Given a mid-price, compute new orders that need to be placed, then send the orders to the Exchange.

            :param mid: mid-price
            :type mid: int

        """

        bid_orders, ask_orders = self.computeOrdersToPlace(mid)

        if self.backstop_quantity is not None:
            bid_price = bid_orders[0]
            log_print('{}: Placing BUY limit order of size {} @ price {}', self.name, self.backstop_quantity, bid_price)
            self.placeLimitOrder(self.symbol, self.backstop_quantity, True, bid_price)
            bid_orders = bid_orders[1:]

            ask_price = ask_orders[-1]
            log_print('{}: Placing SELL limit order of size {} @ price {}', self.name, self.backstop_quantity, ask_price)
            self.placeLimitOrder(self.symbol, self.backstop_quantity, False, ask_price)
            ask_orders = ask_orders[:-1]

        for bid_price in bid_orders:
            log_print('{}: Placing BUY limit order of size {} @ price {}', self.name, self.buy_order_size, bid_price)
            self.placeLimitOrder(self.symbol, self.buy_order_size, True, bid_price)

        for ask_price in ask_orders:
            log_print('{}: Placing SELL limit order of size {} @ price {}', self.name, self.sell_order_size, ask_price)
            self.placeLimitOrder(self.symbol, self.sell_order_size, False, ask_price)

    def getWakeFrequency(self):
        """ Get time increment corresponding to wakeup period. """
        return pd.Timedelta(self.wake_up_freq)

    def cancelAllOrders(self):
        """ Cancels all resting limit orders placed by the market maker """
        for _, order in self.orders.items():
            self.cancelOrder(order)
