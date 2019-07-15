# LimitOrder class, inherits from Order class, adds a limit price.  These are the
# Orders that typically go in an Exchange's OrderBook.

from util.order.Order import Order
from Kernel import Kernel
from agent.FinancialAgent import dollarize

import sys

# Module level variable that can be changed by config files.
silent_mode = False

class LimitOrder (Order):


  def __init__ (self, agent_id, time_placed, symbol, quantity, is_buy_order, limit_price, order_id=None):
    super().__init__(agent_id, time_placed, symbol, quantity, is_buy_order, order_id)

    # The limit price is the minimum price the agent will accept (for a sell order) or
    # the maximum price the agent will pay (for a buy order).
    self.limit_price = limit_price

  def __str__ (self):
    if silent_mode: return ''

    filled = ''
    if self.fill_price: filled = " (filled @ {})".format(dollarize(self.fill_price))

    # Until we make explicit market orders, we make a few assumptions that EXTREME prices on limit
    # orders are trying to represent a market order.  This only affects printing - they still hit
    # the order book like limit orders, which is wrong.
    return "(Agent {} @ {}) : {} {} {} @ {}{}".format(self.agent_id, Kernel.fmtTime(self.time_placed),
                                                      "BUY" if self.is_buy_order else "SELL", self.quantity, self.symbol,
                                                      dollarize(self.limit_price) if abs(self.limit_price) < sys.maxsize else 'MKT', filled)

  def __repr__ (self):
    if silent_mode: return ''
    return self.__str__()
