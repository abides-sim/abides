# A basic Order type used by an Exchange to conduct trades or maintain an order book.
# This should not be confused with order Messages agents send to request an Order.
# Specific order types will inherit from this (like LimitOrder).

from copy import deepcopy


class Order:
    order_id = 0
    _order_ids = set()

    def __init__(self, agent_id, time_placed, symbol, quantity, is_buy_order, order_id=None, tag=None):

        self.agent_id = agent_id

        # Time at which the order was created by the agent.
        self.time_placed: pd_Timestamp = time_placed

        # Equity symbol for the order.
        self.symbol = symbol

        # Number of equity units affected by the order.
        self.quantity = quantity

        # Boolean: True indicates a buy order; False indicates a sell order.
        self.is_buy_order = is_buy_order

        # Order ID: either self generated or assigned
        self.order_id = self.generateOrderId() if not order_id else order_id
        Order._order_ids.add(self.order_id)

        # Create placeholder fields that don't get filled in until certain
        # events happen.  (We could instead subclass to a special FilledOrder
        # class that adds these later?)
        self.fill_price = None

        # Tag: a free-form user-defined field that can contain any information relevant to the
        #      entity placing the order.  Recommend keeping it alphanumeric rather than
        #      shoving in objects, as it will be there taking memory for the lifetime of the
        #      order and in all logging mechanisms.  Intent: for strategy agents to set tags
        #      to help keep track of the intent of particular orders, to simplify their code.
        self.tag = tag

    def generateOrderId(self):
        # generates a unique order ID if the order ID is not specified
        if not Order.order_id in Order._order_ids:
            oid = Order.order_id
        else:
            Order.order_id += 1
            oid = self.generateOrderId()
        return oid

    def to_dict(self):
        as_dict = deepcopy(self).__dict__
        as_dict['time_placed'] = self.time_placed.isoformat()
        return as_dict

    def __copy__(self):
        raise NotImplementedError

    def __deepcopy__(self, memodict={}):
        raise NotImplementedError
