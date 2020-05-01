import pandas as pd
import datetime

from agent.TradingAgent import TradingAgent
from util.util import log_print


class ExecutionAgent(TradingAgent):

    def __init__(self, id, name, type, symbol, starting_cash,
                 direction, quantity, execution_time_horizon,
                 trade=True, log_orders=False, random_state=None):
        super().__init__(id, name, type, starting_cash=starting_cash, log_orders=log_orders, random_state=random_state)
        self.symbol = symbol
        self.direction = direction
        self.quantity = quantity
        self.execution_time_horizon = execution_time_horizon

        self.start_time = self.execution_time_horizon[0]
        self.end_time = self.execution_time_horizon[-1]
        self.schedule = None

        self.rem_quantity = quantity
        self.arrival_price = None

        self.accepted_orders = []
        self.trade = trade
        self.log_orders = log_orders

    def kernelStopping(self):
        super().kernelStopping()
        if self.trade:
            slippage = self.get_average_transaction_price() - self.arrival_price if self.direction == 'BUY' else \
                       self.arrival_price - self.get_average_transaction_price()
            self.logEvent('DIRECTION', self.direction, True)
            self.logEvent('TOTAL_QTY', self.quantity, True)
            self.logEvent('REM_QTY', self.rem_quantity, True)
            self.logEvent('ARRIVAL_MID', self.arrival_price, True)
            self.logEvent('AVG_TXN_PRICE', self.get_average_transaction_price(), True)
            self.logEvent('SLIPPAGE', slippage, True)

    def wakeup(self, currentTime):
        can_trade = super().wakeup(currentTime)
        if not can_trade: return
        if self.trade:
            try:
                self.setWakeup([time for time in self.execution_time_horizon if time > currentTime][0])
            except IndexError:
                pass

            self.getCurrentSpread(self.symbol, depth=1000)
            self.state = 'AWAITING_SPREAD'

    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)
        if msg.body['msg'] == 'ORDER_EXECUTED': self.handleOrderExecution(currentTime, msg)
        elif msg.body['msg'] == 'ORDER_ACCEPTED': self.handleOrderAcceptance(currentTime, msg)
        if self.rem_quantity > 0 and self.state == 'AWAITING_SPREAD' and msg.body['msg'] == 'QUERY_SPREAD':
            self.cancelOrders()
            self.placeOrders(currentTime)

    def handleOrderExecution(self, currentTime, msg):
        executed_order = msg.body['order']
        self.executed_orders.append(executed_order)
        executed_qty = sum(executed_order.quantity for executed_order in self.executed_orders)
        self.rem_quantity = self.quantity - executed_qty
        log_print(f'[---- {self.name} - {currentTime} ----]: LIMIT ORDER EXECUTED - {executed_order.quantity} @ {executed_order.fill_price}')
        log_print(f'[---- {self.name} - {currentTime} ----]: EXECUTED QUANTITY: {executed_qty}')
        log_print(f'[---- {self.name} - {currentTime} ----]: REMAINING QUANTITY (NOT EXECUTED): {self.rem_quantity}')
        log_print(f'[---- {self.name} - {currentTime} ----]: % EXECUTED: {round((1 - self.rem_quantity / self.quantity) * 100, 2)} \n')

    def handleOrderAcceptance(self, currentTime, msg):
        accepted_order = msg.body['order']
        self.accepted_orders.append(accepted_order)
        accepted_qty = sum(accepted_order.quantity for accepted_order in self.accepted_orders)
        log_print(f'[---- {self.name} - {currentTime} ----]: ACCEPTED QUANTITY : {accepted_qty}')

    def placeOrders(self, currentTime):
        if currentTime == self.execution_time_horizon[-2]:
            self.placeMarketOrder(symbol=self.symbol, quantity=self.rem_quantity, is_buy_order=self.direction == 'BUY')
        elif currentTime in self.execution_time_horizon[:-2]:
            bid, _, ask, _ = self.getKnownBidAsk(self.symbol)

            if currentTime == self.start_time:
                self.arrival_price = (bid + ask) / 2
                log_print(f"[---- {self.name}  - {currentTime} ----]: Arrival Mid Price {self.arrival_price}")

            qty = self.schedule[pd.Interval(currentTime, currentTime+datetime.timedelta(minutes=1))]
            price = ask if self.direction == 'BUY' else bid
            self.placeLimitOrder(symbol=self.symbol, quantity=qty,
                                 is_buy_order=self.direction == 'BUY', limit_price=price)
            log_print(f'[---- {self.name} - {currentTime} ----]: LIMIT ORDER PLACED - {qty} @ {price}')

    def cancelOrders(self):
        for _, order in self.orders.items():
            self.cancelOrder(order)

    def getWakeFrequency(self):
        return self.execution_time_horizon[0] - self.mkt_open