# A basic Order type used by an Exchange to conduct trades or maintain an order book.
# This should not be confused with order Messages agents send to request an Order.
# Specific order types will inherit from this (like LimitOrder).

class Order:

  order_id = 0
  order_ids = []

  def __init__(self, agent_id, time_placed, symbol, quantity, is_buy_order, order_id=None):
    self.agent_id = agent_id

    # Time at which the order was created by the agent.
    self.time_placed = time_placed

    # Equity symbol for the order.
    self.symbol = symbol

    # Number of equity units affected by the order.
    self.quantity = quantity

    # Boolean: True indicates a buy order; False indicates a sell order.
    self.is_buy_order = is_buy_order

    # Order ID: either self generated or assigned
    self.order_id = self.generateOrderId() if not order_id else order_id
    Order.order_ids.append(self.order_id)

    # Create placeholder fields that don't get filled in until certain
    # events happen.  (We could instead subclass to a special FilledOrder
    # class that adds these later?)
    self.fill_price = None

  def generateOrderId(self):
    # generates a unique order ID if the order ID is not specified
    if not Order.order_id in Order.order_ids:
      oid = Order.order_id
    else:
      Order.order_id += 1
      oid = self.generateOrderId()
    return oid