# BasketOrder class, inherits from Order class.  These are the
# Orders that typically go in a Primary Exchange and immediately get filled.
# A buy order translates to a creation order for an ETF share
# A sell order translates to a redemption order for shares of the underlying.

from util.order.Order import Order
from Kernel import Kernel
from agent.FinancialAgent import dollarize

import sys

# Module level variable that can be changed by config files.
silent_mode = False

class BasketOrder (Order):

  def __init__ (self, agent_id, time_placed, symbol, quantity, is_buy_order, dollar=True, order_id=None):
    super().__init__(agent_id, time_placed, symbol, quantity, is_buy_order, order_id)
    self.dollar = dollar

  def __str__ (self):
    if silent_mode: return ''

    filled = ''
    if self.dollar:
        if self.fill_price: filled = " (filled @ {})".format(dollarize(self.fill_price))
    else:
        if self.fill_price: filled = " (filled @ {})".format(self.fill_price)

    # Until we make explicit market orders, we make a few assumptions that EXTREME prices on limit
    # orders are trying to represent a market order.  This only affects printing - they still hit
    # the order book like limit orders, which is wrong.
    return "(Order_ID: {} Agent {} @ {}) : {} {} {} @ {}{}".format(self.order_id, self.agent_id,
                                                                   Kernel.fmtTime(self.time_placed),
                                                                   "CREATE" if self.is_buy_order else "REDEEM",
                                                                   self.quantity, self.symbol, 
                                                                   filled, self.fill_price)

  def __repr__ (self):
    if silent_mode: return ''
    return self.__str__()
