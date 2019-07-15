from agent.TradingAgent import TradingAgent


class ExperimentalAgent(TradingAgent):

    def __init__(self, id, name, symbol,
                 starting_cash, execution_timestamp, quantity, is_buy_order, limit_price,
                 log_orders = False, random_state = None):
        super().__init__(id, name, starting_cash, random_state)
        self.symbol              = symbol
        self.execution_timestamp = execution_timestamp
        self.quantity            = quantity
        self.is_buy_order        = is_buy_order
        self.limit_price         = limit_price
        self.log_orders          = log_orders

    def kernelStarting(self, startTime):
        super().kernelStarting(startTime)

    def wakeup(self, currentTime):
        super().wakeup(currentTime)
        self.last_trade[self.symbol] = 0
        if not self.mkt_open or not self.mkt_close:
            return
        elif (currentTime > self.mkt_open) and (currentTime < self.mkt_close):
            if currentTime == self.execution_timestamp:
                self.placeLimitOrder(self.symbol, self.quantity, self.is_buy_order, self.limit_price)
                if self.log_orders: self.logEvent('LIMIT_ORDER', {'agent_id': self.id, 'fill_price': None,
                                                                  'is_buy_order': self.is_buy_order, 'limit_price': self.limit_price,
                                                                  'order_id': 1, 'quantity': self.quantity, 'symbol': self.symbol,
                                                                  'time_placed': str(currentTime)})

    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)

    def getWakeFrequency(self):
        return self.execution_timestamp - self.mkt_open
