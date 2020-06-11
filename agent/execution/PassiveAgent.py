from agent.TradingAgent import TradingAgent
from util.util import log_print


class PassiveAgent(TradingAgent):
    """
    PassiveAgent class representing an agent placing LIMIT orders in the order book
    Attributes:
        symbol (str):           Name of the stock traded
        timestamp (datetime):   order placement time stamp
        direction (str):        order direction ('BUY' or 'SELL')
        quantity (int):         order quantity
        limit_price (float):    order limit price
        log_orders (bool):      log the order(s) placed
    """

    def __init__(self, id, name, type, symbol, starting_cash,
                 timestamp, direction, quantity, limit_price=None,
                 log_orders=False, random_state=None):
        super().__init__(id, name, type, starting_cash=starting_cash, log_orders=log_orders, random_state=random_state)
        self.symbol = symbol
        self.timestamp = timestamp
        self.direction = direction
        self.quantity = quantity
        self.limit_price = limit_price
        self.log_orders = log_orders
        self.state = 'AWAITING_WAKEUP'

    def wakeup(self, currentTime):
        can_trade = super().wakeup(currentTime)
        if not can_trade: return
        elif currentTime == self.timestamp:
            if self.limit_price:
                self.placeLimitOrder(symbol=self.symbol, quantity=self.quantity,
                                     is_buy_order=self.direction == 'BUY', limit_price=self.limit_price)
                log_print('[---- {} - {} ----]: LIMIT ORDER PLACED - {} @ {}'.format(self.name, currentTime,
                                                                                     self.quantity, self.limit_price))
            else:
                self.getCurrentSpread(self.symbol)
                self.state = 'AWAITING_SPREAD'

    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)
        if self.state == 'AWAITING_SPREAD' and msg.body['msg'] == 'QUERY_SPREAD':
            bid, _, ask, _ = self.getKnownBidAsk(self.symbol)
            limit_price = bid if self.direction == 'BUY' else ask
            self.placeLimitOrder(symbol=self.symbol, quantity=self.quantity,
                                 is_buy_order=self.direction == 'BUY', limit_price=limit_price)
            log_print('[---- {} - {} ----]: LIMIT ORDER PLACED - {} @ {}'.format(self.name, currentTime,
                                                                                 self.quantity, limit_price))

    def getWakeFrequency(self):
        return self.timestamp - self.mkt_open