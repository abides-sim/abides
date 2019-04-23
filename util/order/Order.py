# A basic Order type used by an Exchange to conduct trades or maintain an order book.
# This should not be confused with order Messages agents send to request an Order.
# Specific order types will inherit from this (like LimitOrder).

class Order:

  order_id = 0

  def __init__(self, agent_id, time_placed, symbol, quantity, is_buy_order):
    self.agent_id = agent_id
    self.time_placed = time_placed
    self.symbol = symbol
    self.quantity = quantity
    self.is_buy_order = is_buy_order

    # Assign and increment the next unique order_id (simulation-wide).
    self.order_id = Order.order_id
    Order.order_id += 1

    # Create placeholder fields that don't get filled in until certain
    # events happen.  (We could instead subclass to a special FilledOrder
    # class that adds these later?)
    self.fill_price = None

