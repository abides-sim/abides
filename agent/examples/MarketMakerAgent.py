from agent.TradingAgent import TradingAgent
import pandas as pd


class MarketMakerAgent(TradingAgent):
    """
    Simple market maker agent that attempts to provide liquidity in the orderbook by placing orders on both sides every
    time it wakes up. The agent starts off by cancelling any existing orders, it then queries the current spread to
    determine the trades to be placed. The order size is chosen at random between min_size and max_size which are parameters
    of the agent. The agent places orders in 1-5 price levels randomly with sizes determined by the levels_quote_dict.
    """

    def __init__(self, id, name, type, symbol, starting_cash, min_size, max_size , wake_up_freq='10s',
                 log_orders=False, random_state=None):

        super().__init__(id, name, type, starting_cash=starting_cash, log_orders=log_orders, random_state=random_state)
        self.symbol = symbol      # Symbol traded
        self.min_size = min_size  # Minimum order size
        self.max_size = max_size  # Maximum order size
        self.size = round(self.random_state.randint(self.min_size, self.max_size) / 2) # order size per LOB side
        self.wake_up_freq = wake_up_freq # Frequency of agent wake up
        self.log_orders = log_orders
        self.state = "AWAITING_WAKEUP"
        # Percentage of the order size to be placed at different levels is determined by levels_quote_dict
        self.levels_quote_dict = {1: [1, 0, 0, 0, 0],
                                  2: [.5, .5, 0, 0, 0],
                                  3: [.34, .33, .33, 0, 0],
                                  4: [.25, .25, .25, .25, 0],
                                  5: [.20, .20, .20, .20, .20]}

    def kernelStarting(self, startTime):
        super().kernelStarting(startTime)

    def wakeup(self, currentTime):
        """ Agent wakeup is determined by self.wake_up_freq """
        can_trade = super().wakeup(currentTime)
        if not can_trade: return
        self.cancelOrders()
        self.getCurrentSpread(self.symbol, depth=5)
        self.state = 'AWAITING_SPREAD'

    def receiveMessage(self, currentTime, msg):
        """ Market Maker actions are determined after obtaining the bids and asks in the LOB """
        super().receiveMessage(currentTime, msg)
        if self.state == 'AWAITING_SPREAD' and msg.body['msg'] == 'QUERY_SPREAD':
            bids, asks = self.getKnownBidAsk(self.symbol, best=False)
            num_levels = self.random_state.randint(1, 6)        # Number of price levels to place the trades in
            size_split = self.levels_quote_dict.get(num_levels) # % of the order size to be placed at different levels

            if bids and asks:
                buy_quotes, sell_quotes = {}, {}
                for i in range(num_levels):
                    vol = round(size_split[i] * self.size)
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
            self.setWakeup(currentTime + self.getWakeFrequency())
            self.state = 'AWAITING_WAKEUP'

    def cancelOrders(self):
        """ cancels all resting limit orders placed by the market maker """
        for _, order in self.orders.items():
            self.cancelOrder(order)

    def getWakeFrequency(self):
        return pd.Timedelta(self.wake_up_freq)