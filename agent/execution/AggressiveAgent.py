from agent.TradingAgent import TradingAgent


class AggressiveAgent(TradingAgent):
    """
    AggressiveAgent class representing an agent placing MARKET orders in the order book
    Attributes:
        symbol (str):           Name of the stock traded
        timestamp (datetime):   order placement time stamp
        direction (str):        order direction ('BUY' or 'SELL')
        quantity (int):         order quantity
        log_orders (bool):      log the order(s) placed
    """

    def __init__(self, id, name, type, symbol, starting_cash,
                 timestamp, direction, quantity,
                 log_orders=False, random_state=None):
        super().__init__(id, name, type, starting_cash=starting_cash, log_orders=log_orders, random_state=random_state)
        self.symbol = symbol
        self.timestamp = timestamp
        self.direction = direction
        self.quantity = quantity
        self.log_orders = log_orders
        self.state = 'AWAITING_WAKEUP'

    def wakeup(self, currentTime):
        can_trade = super().wakeup(currentTime)
        if not can_trade: return
        elif currentTime == self.timestamp:
            self.getCurrentSpread(self.symbol, depth=100)
            self.state = 'AWAITING_SPREAD'

    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)
        if self.state == 'AWAITING_SPREAD' and msg.body['msg'] == 'QUERY_SPREAD':
            self.placeMarketOrder(self.symbol, self.quantity, self.direction == 'BUY')

    def getWakeFrequency(self):
        return self.timestamp - self.mkt_open