from agent.TradingAgent import TradingAgent
import pandas as pd
from util.util import log_print


DEFAULT_LEVELS_QUOTE_DICT = {1: [1, 0, 0, 0, 0],
                             2: [.5, .5, 0, 0, 0],
                             3: [.34, .33, .33, 0, 0],
                             4: [.25, .25, .25, .25, 0],
                             5: [.20, .20, .20, .20, .20]}


class MarketMakerAgent(TradingAgent):
    """
    Simple market maker agent that attempts to provide liquidity in the orderbook by placing orders on both sides every
    time it wakes up. The agent starts off by cancelling any existing orders, it then queries the current spread to
    determine the trades to be placed. The order size is chosen at random between min_size and max_size which are parameters
    of the agent.

    If subscribe == True
        The agent places orders in 1-5 price levels randomly with sizes determined by the levels_quote_dict.

    Else if subscribe == False:
        The agent places an order of size / 2 at both the best bid and best ask price levels

    """

    def __init__(self, id, name, type, symbol, starting_cash, min_size, max_size , wake_up_freq='1s',
                 subscribe=False, subscribe_freq=10e9, subscribe_num_levels=5, log_orders=False, random_state=None):

        super().__init__(id, name, type, starting_cash=starting_cash, log_orders=log_orders, random_state=random_state)
        self.symbol = symbol      # Symbol traded
        self.min_size = min_size  # Minimum order size
        self.max_size = max_size  # Maximum order size
        self.size = round(self.random_state.randint(self.min_size, self.max_size) / 2) # order size per LOB side
        self.wake_up_freq = wake_up_freq  # Frequency of agent wake up
        self.subscribe = subscribe  # Flag to determine whether to subscribe to data or use polling mechanism
        self.subscribe_freq = subscribe_freq  # Frequency in nanoseconds^-1 at which to receive market updates
                                              # in subscribe mode
        self.subscribe_num_levels = subscribe_num_levels  # Number of orderbook levels in subscription mode
        self.subscription_requested = False
        self.log_orders = log_orders
        self.state = "AWAITING_WAKEUP"
        # Percentage of the order size to be placed at different levels is determined by levels_quote_dict
        self.levels_quote_dict = DEFAULT_LEVELS_QUOTE_DICT
        self.num_levels = None
        self.size_split = None
        self.last_spread = 10

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
            self.cancelOrders()
            self.getCurrentSpread(self.symbol, depth=self.subscribe_num_levels)
            self.state = 'AWAITING_SPREAD'

    def receiveMessage(self, currentTime, msg):
        """ Market Maker actions are determined after obtaining the bids and asks in the LOB """
        super().receiveMessage(currentTime, msg)
        if not self.subscribe and self.state == 'AWAITING_SPREAD' and msg.body['msg'] == 'QUERY_SPREAD':
            self.cancelOrders()
            mid = self.last_trade[self.symbol]

            self.num_levels = 2 * self.subscribe_num_levels   # Number of price levels to place the trades in

            bid, bid_vol, ask, ask_vol = self.getKnownBidAsk(self.symbol)

            if bid and ask:
                mid = int((ask + bid) / 2)
                spread = int(abs(ask - bid)/2)
            else:
                log_print(f"SPREAD MISSING at time {currentTime}")
                spread = self.last_spread

            for i in range(self.num_levels):
                self.size = round(self.random_state.randint(self.min_size, self.max_size) / 2)
                #bids
                self.placeLimitOrder(self.symbol, self.size, True, mid - spread - i)
                #asks
                self.placeLimitOrder(self.symbol, self.size, False, mid + spread + i)

            self.setWakeup(currentTime + self.getWakeFrequency())
            self.state = 'AWAITING_WAKEUP'

        elif self.subscribe and self.state == 'AWAITING_MARKET_DATA' and msg.body['msg'] == 'MARKET_DATA':
            self.cancelOrders()
            num_levels_place = len(self.levels_quote_dict.keys())
            self.num_levels = self.random_state.randint(1, num_levels_place)  # Number of price levels to place the trades in
            self.size_split = self.levels_quote_dict.get(self.num_levels)  # % of the order size to be placed at different levels
            self.placeOrders(self.known_bids[self.symbol], self.known_asks[self.symbol])
            self.state = 'AWAITING_MARKET_DATA'

    def placeOrders(self, bids, asks):
        if bids and asks:
            buy_quotes, sell_quotes = {}, {}
            self.size = round(self.random_state.randint(self.min_size, self.max_size) / 2)
            for i in range(self.num_levels):
                vol = round(self.size_split[i] * self.size)
                try:
                    buy_quotes[bids[i][0]] = vol
                except IndexError:
                    # Orderbook price level i empty so create a new price level with bid price 1 CENT below
                    buy_quotes[bids[-1][0] - 1] = vol
                try:
                    sell_quotes[asks[i][0]] = vol
                except IndexError:
                    # Orderbook price level i empty so create a new price level with bid price 1 CENT above
                    sell_quotes[asks[-1][0] + 1] = vol

            for price, vol in buy_quotes.items():
                self.placeLimitOrder(self.symbol, vol, True, price)
            for price, vol in sell_quotes.items():
                self.placeLimitOrder(self.symbol, vol, False, price)


    def cancelOrders(self):
        """ cancels all resting limit orders placed by the market maker """
        for _, order in self.orders.items():
            self.cancelOrder(order)

    def getWakeFrequency(self):
        return pd.Timedelta(self.wake_up_freq)