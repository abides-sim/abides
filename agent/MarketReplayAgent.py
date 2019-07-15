import pandas as pd

from agent.TradingAgent import TradingAgent
from util.order.LimitOrder import LimitOrder
from util.util import log_print


class MarketReplayAgent(TradingAgent):


  def __init__(self, id, name, type, symbol, date, starting_cash, log_orders = False, random_state = None):
    super().__init__(id, name, type, starting_cash=starting_cash, log_orders=log_orders, random_state = random_state)
    self.symbol      = symbol
    self.date        = date
    self.log_orders  = log_orders
    self.state       = 'AWAITING_WAKEUP'


  def kernelStarting(self, startTime):
    super().kernelStarting(startTime)
    self.oracle = self.kernel.oracle

  def kernelStopping (self):
    super().kernelStopping()

  def wakeup (self, currentTime):
    self.state = 'INACTIVE'
    try:
        super().wakeup(currentTime)
        self.last_trade[self.symbol] = self.oracle.getDailyOpenPrice(self.symbol, self.mkt_open)
        if not self.mkt_open or not self.mkt_close:
          return
        order = self.oracle.trades_df.loc[self.oracle.trades_df.timestamp == currentTime]
        wake_up_time = self.oracle.trades_df.loc[self.oracle.trades_df.timestamp > currentTime].iloc[0].timestamp
        if (currentTime > self.mkt_open) and (currentTime < self.mkt_close):
          self.state = 'ACTIVE'
          try:
            self.placeOrder(currentTime, order)
          except Exception as e:
            log_print(e)
        self.setWakeup(wake_up_time)
    except Exception as e:
        log_print(str(e))


  def receiveMessage (self, currentTime, msg):
    super().receiveMessage(currentTime, msg)


  def placeOrder(self, currentTime, order):
    if len(order) == 1:
        type        = order.type.item()
        id          = order.order_id.item()
        direction   = order.direction.item()
        price       = order.price.item()
        vol         = order.vol.item()

        existing_order = self.orders.get(id)

        if type == 'NEW':
            self.placeLimitOrder(self.symbol, vol, direction == 'BUY', int(price), order_id=id)
        elif type in ['CANCELLATION', 'PARTIAL_CANCELLATION']:
            if existing_order:
                if type == 'CANCELLATION':
                    self.cancelOrder(existing_order)
                elif type == 'PARTIAL_CANCELLATION':
                    new_order = LimitOrder(self.id, currentTime, self.symbol, vol, direction == 'BUY', int(price), order_id=id)
                    self.modifyOrder(existing_order, new_order)
        elif type in ['EXECUTE_VISIBLE', 'EXECUTE_HIDDEN']:
            if existing_order:
                if existing_order.quantity == vol:
                    self.cancelOrder(existing_order)
                else:
                    new_vol = existing_order.quantity - vol
                    if new_vol == 0:
                        self.cancelOrder(existing_order)
                    else:
                        executed_order = LimitOrder(self.id, currentTime, self.symbol, new_vol, direction == 'BUY', int(price), order_id=id)
                        self.modifyOrder(existing_order, executed_order)
                        self.orders.get(id).quantity = new_vol
    else:
        orders = self.oracle.trades_df.loc[self.oracle.trades_df.timestamp == currentTime]
        for index, order in orders.iterrows():
            self.placeOrder(currentTime, order = pd.DataFrame(order).T)


  def getWakeFrequency(self):
    return self.oracle.trades_df.iloc[0].timestamp - self.mkt_open