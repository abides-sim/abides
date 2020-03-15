from agent.TradingAgent import TradingAgent
import pandas as pd
from util.util import log_print
from collections import namedtuple, deque
from util.util import ignored


ANCHOR_TOP_STR = 'top'
ANCHOR_BOTTOM_STR = 'bottom'


class SpreadBasedMarketMakerAgent(TradingAgent):
    """ This class implements the Chakraborty-Kearns `ladder` market-making strategy. """

    _Order = namedtuple('_Order', ['price', 'id'])  # Internal data structure used to describe a placed order

    def __init__(self, id, name, type, symbol, starting_cash, order_size=1, window_size=5, anchor=ANCHOR_BOTTOM_STR,
                 num_ticks=20, wake_up_freq='1s', subscribe=True, subscribe_freq=10e9, subscribe_num_levels=1,
                 log_orders=False, random_state=None):

        super().__init__(id, name, type, starting_cash=starting_cash, log_orders=log_orders, random_state=random_state)
        self.symbol = symbol      # Symbol traded
        self.order_size = order_size  # order size per price level
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
        self.state = "AWAITING_WAKEUP"

        self.current_bids = None  # double-ended queue holding bid orders in the book
        self.current_asks = None  # double-ended queue holding ask orders in the book
        self.last_mid = None  # last observed mid price
        self.order_id_counter = 0  # counter for bookkeeping orders made by self
        self.LIQUIDITY_DROPOUT_WARNING = f"Liquidity dropout for agent {self.name}."

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
            super().requestDataSubscription(self.symbol, levels=self.subscribe_num_levels, freq=self.subscribe_freq)
            self.subscription_requested = True
            self.state = 'AWAITING_MARKET_DATA'
        elif can_trade and not self.subscribe:
            self.getCurrentSpread(self.symbol, depth=self.subscribe_num_levels)
            self.state = 'AWAITING_SPREAD'

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

        if not self.subscribe and self.state == 'AWAITING_SPREAD' and msg.body['msg'] == 'QUERY_SPREAD':

            bid, _, ask, _ = self.getKnownBidAsk(self.symbol)
            if bid and ask:
                mid = int((ask + bid) / 2)
            else:
                log_print(f"SPREAD MISSING at time {currentTime}")

            orders_to_cancel = self.computeOrdersToCancel(mid)
            self.cancelOrders(orders_to_cancel)
            self.placeOrders(mid)
            self.setWakeup(currentTime + self.getWakeFrequency())
            self.state = 'AWAITING_WAKEUP'
            self.last_mid = mid

        elif self.subscribe and self.state == 'AWAITING_MARKET_DATA' and msg.body['msg'] == 'MARKET_DATA':

            bid = self.known_bids[self.symbol][0][0] if self.known_bids[self.symbol] else None
            ask = self.known_asks[self.symbol][0][0] if self.known_asks[self.symbol] else None
            if bid and ask:
                mid = int((ask + bid) / 2)
            else:
                log_print(f"SPREAD MISSING at time {currentTime}")
                return

            orders_to_cancel = self.computeOrdersToCancel(mid)
            self.cancelOrders(orders_to_cancel)
            self.placeOrders(mid)
            self.state = 'AWAITING_MARKET_DATA'
            self.last_mid = mid

    def computeOrdersToCancel(self, mid):
        """ Given a mid price, computes the orders that need to be removed from orderbook, and pops these orders from
            bid and ask deques.

        :param mid: mid-price
        :type mid: int

        :return:
        """

        orders_to_cancel = []

        if (self.current_asks is None) or (self.current_bids is None):
            return orders_to_cancel

        num_ticks_to_increase = int(mid - self.last_mid)

        if num_ticks_to_increase > 0:
            for _ in range(num_ticks_to_increase):
                with ignored(self.LIQUIDITY_DROPOUT_WARNING, IndexError):
                    orders_to_cancel.append(self.current_bids.popleft())
                with ignored(self.LIQUIDITY_DROPOUT_WARNING, IndexError):
                    orders_to_cancel.append(self.current_asks.popleft())
        elif num_ticks_to_increase < 0:
            for _ in range(- num_ticks_to_increase):
                with ignored(self.LIQUIDITY_DROPOUT_WARNING, IndexError):
                    orders_to_cancel.append(self.current_bids.pop())
                with ignored(self.LIQUIDITY_DROPOUT_WARNING, IndexError):
                    orders_to_cancel.append(self.current_asks.pop())

        return orders_to_cancel

    def cancelOrders(self, orders_to_cancel):
        """ Given a list of _Order objects, remove the corresponding orders from ExchangeAgent's orderbook

        :param orders_to_cancel: orders to remove from orderbook
        :type orders_to_cancel: list(_Order)
        :return:
        """
        for order_tuple in orders_to_cancel:
            order_id = order_tuple.id
            try:
                order = self.orders[order_id]
                self.cancelOrder(order)
            except KeyError:
                continue

    def computeOrdersToPlace(self, mid):
        """ Given a mid price, computes the orders that need to be removed from orderbook, and adds these orders to
            bid and ask deques.

        :param mid: mid-price
        :type mid: int

        :return:
        """

        bids_to_place = []
        asks_to_place = []

        if (not self.current_asks) or (not self.current_bids):
            self.cancelAllOrders()
            self.initialiseBidsAsksDeques(mid)
            bids_to_place.extend([order for order in self.current_bids])
            asks_to_place.extend([order for order in self.current_asks])
            return bids_to_place, asks_to_place

        if self.last_mid is not None:
            num_ticks_to_increase = int(mid - self.last_mid)
        else:
            num_ticks_to_increase = 0

        if num_ticks_to_increase > 0:

            base_bid_price = self.current_bids[-1].price
            base_ask_price = self.current_asks[-1].price

            for price_increment in range(1, num_ticks_to_increase + 1):
                bid_price = base_bid_price + price_increment
                new_bid_order = self.generateNewOrderId(bid_price)
                bids_to_place.append(new_bid_order)
                self.current_bids.append(new_bid_order)

                ask_price = base_ask_price + price_increment
                new_ask_order = self.generateNewOrderId(ask_price)
                asks_to_place.append(new_ask_order)
                self.current_asks.append(new_ask_order)

        elif num_ticks_to_increase < 0:

            base_bid_price = self.current_bids[0].price
            base_ask_price = self.current_asks[0].price

            for price_increment in range(1, 1 - num_ticks_to_increase):
                bid_price = base_bid_price - price_increment
                new_bid_order = self.generateNewOrderId(bid_price)
                bids_to_place.append(new_bid_order)
                self.current_bids.appendleft(new_bid_order)

                ask_price = base_ask_price - price_increment
                new_ask_order = self.generateNewOrderId(ask_price)
                asks_to_place.append(new_ask_order)
                self.current_asks.appendleft(new_ask_order)

        return bids_to_place, asks_to_place

    def placeOrders(self, mid):
        """ Given a mid-price, compute new orders that need to be placed, then send the orders to the Exchange.

            :param mid: mid-price
            :type mid: int

        """

        bid_orders, ask_orders = self.computeOrdersToPlace(mid)
        for bid_order in bid_orders:
            log_print(f'{self.name}: Placing BUY limit order of size {self.order_size} @ price {bid_order.price}')
            self.placeLimitOrder(self.symbol, self.order_size, True, bid_order.price, order_id=bid_order.id)

        for ask_order in ask_orders:
            log_print(f'{self.name}: Placing SELL limit order of size {self.order_size} @ price {ask_order.price}')
            self.placeLimitOrder(self.symbol, self.order_size, False, ask_order.price, order_id=ask_order.id)

    def initialiseBidsAsksDeques(self, mid):
        """ Initialise the current_bids and current_asks object attributes, which internally keep track of the limit
            orders sent to the Exchange.

            :param mid: mid-price
            :type mid: int

        """

        if self.anchor == ANCHOR_BOTTOM_STR:
            highest_bid = int(mid - 1)
            lowest_ask = int(mid + self.window_size)
        elif self.anchor == ANCHOR_TOP_STR:
            highest_bid = int(mid - self.window_size)
            lowest_ask = int(mid + 1)

        lowest_bid = highest_bid - self.num_ticks
        highest_ask = lowest_ask + self.num_ticks

        self.current_bids = deque([self.generateNewOrderId(price) for price in range(lowest_bid, highest_bid + 1)])
        self.current_asks = deque([self.generateNewOrderId(price) for price in range(lowest_ask, highest_ask + 1)])

    def generateNewOrderId(self, price):
        """ Generate a _Order object for a particular price level

        :param price:
        :type price: int

        """
        self.order_id_counter += 1
        order_id = f"{self.name}_{self.id}_{self.order_id_counter}"
        return self._Order(price, order_id)

    def getWakeFrequency(self):
        """ Get time increment corresponding to wakeup period. """
        return pd.Timedelta(self.wake_up_freq)

    def cancelAllOrders(self):
        """ Cancels all resting limit orders placed by the market maker """
        for _, order in self.orders.items():
            self.cancelOrder(order)
