from util.order.Order import Order
from Kernel import Kernel
from agent.FinancialAgent import dollarize

import sys

silent_mode = False


class MarketOrder(Order):

    def __init__(self, agent_id, time_placed, symbol, quantity, is_buy_order, order_id=None):
        super().__init__(agent_id, time_placed, symbol, quantity, is_buy_order, order_id)

    def __str__(self):
        if silent_mode: return ''

        return "(Agent {} @ {}) : MKT Order {} {} {}".format(self.agent_id, Kernel.fmtTime(self.time_placed),
                                                             "BUY" if self.is_buy_order else "SELL",
                                                             self.quantity, self.symbol)

    def __repr__(self):
        if silent_mode: return ''
        return self.__str__()
