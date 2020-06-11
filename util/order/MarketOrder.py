from util.order.Order import Order
from Kernel import Kernel
from agent.FinancialAgent import dollarize
from copy import deepcopy

import sys

silent_mode = False


class MarketOrder(Order):

    def __init__(self, agent_id, time_placed, symbol, quantity, is_buy_order, order_id=None, tag=None):
        super().__init__(agent_id, time_placed, symbol, quantity, is_buy_order, order_id=order_id, tag=tag)

    def __str__(self):
        if silent_mode: return ''

        return "(Agent {} @ {}) : MKT Order {} {} {}".format(self.agent_id, Kernel.fmtTime(self.time_placed),
                                                             "BUY" if self.is_buy_order else "SELL",
                                                             self.quantity, self.symbol)

    def __repr__(self):
        if silent_mode: return ''
        return self.__str__()

    def __copy__(self):
        order = MarketOrder(self.agent_id, self.time_placed, self.symbol, self.quantity, self.is_buy_order,
                            order_id=self.order_id,
                            tag=self.tag)
        Order._order_ids.pop()  # remove duplicate agent ID
        order.fill_price = self.fill_price
        return order

    def __deepcopy__(self, memodict={}):
        # Deep copy instance attributes
        agent_id = deepcopy(self.agent_id, memodict)
        time_placed = deepcopy(self.time_placed, memodict)
        symbol = deepcopy(self.symbol, memodict)
        quantity = deepcopy(self.quantity, memodict)
        is_buy_order = deepcopy(self.is_buy_order, memodict)
        order_id = deepcopy(self.order_id, memodict)
        tag = deepcopy(self.tag, memodict)
        fill_price = deepcopy(self.fill_price, memodict)

        # Create new order object
        order = MarketOrder(agent_id, time_placed, symbol, quantity, is_buy_order, order_id=order_id, tag=tag)
        order.fill_price = fill_price

        return order
