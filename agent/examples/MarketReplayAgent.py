from agent.TradingAgent import TradingAgent
from util.order.LimitOrder import LimitOrder
from util.util import log_print


class MarketReplayAgent(TradingAgent):

    def __init__(self, id, name, type, symbol, date, starting_cash, log_orders=False, random_state=None):
        super().__init__(id, name, type, starting_cash=starting_cash, log_orders=log_orders, random_state=random_state)
        self.symbol = symbol
        self.date = date
        self.log_orders = log_orders
        self.executed_trades = dict()
        self.state = 'AWAITING_WAKEUP'
        self.orders_dict = None
        self.wakeup_times = None

    def kernelStarting(self, startTime):
        super().kernelStarting(startTime)
        self.orders_dict = self.kernel.oracle.orders_dict
        self.wakeup_times = self.kernel.oracle.wakeup_times

    def kernelStopping(self):
        super().kernelStopping()

    def wakeup(self, currentTime):
        super().wakeup(currentTime)
        if not self.mkt_open or not self.mkt_close:
            return
        try:
            self.setWakeup(self.wakeup_times[0])
            self.wakeup_times.pop(0)
            self.placeOrder(currentTime, self.orders_dict[currentTime])
        except IndexError:
            log_print(f"Market Replay Agent submitted all orders - last order @ {currentTime}")

    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)
        if msg.body['msg'] == 'ORDER_EXECUTED':
            order = msg.body['order']
            self.executed_trades[currentTime] = [order.fill_price, order.quantity]

    def placeOrder(self, currentTime, order):
        if len(order) == 1:
            order = order[0]
            order_id = order['ORDER_ID']
            existing_order = self.orders.get(order_id)
            if not existing_order:
                self.placeLimitOrder(self.symbol, order['SIZE'], order['BUY_SELL_FLAG'] == 'BUY', order['PRICE'],
                                     order_id=order_id)
            elif existing_order and order['SIZE'] == 0:
                self.cancelOrder(existing_order)
            elif existing_order:
                self.modifyOrder(existing_order, LimitOrder(self.id, currentTime, self.symbol, order['SIZE'],
                                                            order['BUY_SELL_FLAG'] == 'BUY', order['PRICE'],
                                                            order_id=order_id))
        else:
            for ind_order in order:
                self.placeOrder(currentTime, order=[ind_order])

    def getWakeFrequency(self):
        log_print(f"Market Replay Agent first wake up: {self.kernel.oracle.first_wakeup}")
        return self.kernel.oracle.first_wakeup - self.mkt_open
