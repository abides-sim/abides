import pandas as pd

from agent.TradingAgent import TradingAgent
from util.util import log_print
from util.order.LimitOrder import LimitOrder
from message.Message import Message

class MarketReplayAgent(TradingAgent):


  def __init__(self, id, name, symbol, date, startingCash, log_orders = False, random_state = None):
    super().__init__(id, name, startingCash, random_state)
    self.symbol      = symbol
    self.date        = date
    self.log_orders = log_orders


  def kernelStarting(self, startTime):
    super().kernelStarting(startTime)
    self.oracle = self.kernel.oracle


  def wakeup (self, currentTime):
    super().wakeup(currentTime)
    self.last_trade[self.symbol] = self.oracle.getDailyOpenPrice(self.symbol, self.mkt_open)
    if not self.mkt_open or not self.mkt_close:
      return
    elif currentTime == self.oracle.orderbook_df.iloc[0].name:
      order = self.oracle.trades_df.loc[self.oracle.trades_df.timestamp == currentTime]
      self.placeMktOpenOrders(order, t=currentTime)
      self.setWakeup(self.oracle.trades_df.loc[self.oracle.trades_df.timestamp > currentTime].iloc[0].timestamp)
    elif (currentTime > self.mkt_open) and (currentTime < self.mkt_close):
      try:
        order = self.oracle.trades_df.loc[self.oracle.trades_df.timestamp == currentTime]
        self.placeOrder(currentTime, order)
        self.setWakeup(self.oracle.trades_df.loc[self.oracle.trades_df.timestamp > currentTime].iloc[0].timestamp)
      except Exception as e:
        log_print(e)


  def receiveMessage (self, currentTime, msg):
    super().receiveMessage(currentTime, msg)


  def placeMktOpenOrders(self, snapshot_order, t=0):
      orders_snapshot = self.oracle.orderbook_df.loc[self.oracle.orderbook_df.index == t].T
      for i in range(0, len(orders_snapshot) - 1, 4):
          ask_price = orders_snapshot.iloc[i][0]
          ask_vol = orders_snapshot.iloc[i + 1][0]
          bid_price = orders_snapshot.iloc[i + 2][0]
          bid_vol = orders_snapshot.iloc[i + 3][0]

          if snapshot_order.direction.item() == 'BUY' and bid_price == snapshot_order.price.item():
              bid_vol -= snapshot_order.vol.item()
          elif snapshot_order.direction.item() == 'SELL' and ask_price == snapshot_order.price.item():
              ask_vol -= snapshot_order.vol.item()

          self.placeLimitOrder(self.symbol, bid_vol, True, float(bid_price), dollar=False)
          self.placeLimitOrder(self.symbol, ask_vol, False, float(ask_price), dollar=False)
      self.placeOrder(snapshot_order.timestamp.item(), snapshot_order)


  def placeOrder(self, currentTime, order):
    if len(order) == 1:
        type        = order.type.item()
        id          = order.order_id.item()
        direction   = order.direction.item()
        price       = order.price.item()
        vol         = order.vol.item()
        if type == 'NEW':
            self.placeLimitOrder(self.symbol, vol, direction == 'BUY', float(price), dollar=False, order_id=id)
        elif type in ['CANCELLATION', 'PARTIAL_CANCELLATION']:
            existing_order = self.orders.get(id)
            if existing_order:
                if type == 'CANCELLATION':
                    self.cancelOrder(existing_order)
                elif type == 'PARTIAL_CANCELLATION':
                    new_order = LimitOrder(self.id, currentTime, self.symbol, vol, direction == 'BUY', float(price),
                                           dollar=False, order_id=id)
                    self.modifyOrder(existing_order, new_order)
            else:
                self.replicateOrderbookSnapshot(currentTime)
        elif type in ['EXECUTE_VISIBLE', 'EXECUTE_HIDDEN']:
            existing_order = self.orders.get(id)
            if existing_order:
                if existing_order.quantity == vol:
                    self.cancelOrder(existing_order)
                else:
                    new_vol = existing_order.quantity - vol
                    if new_vol == 0:
                        self.cancelOrder(existing_order)
                    else:
                        executed_order = LimitOrder(self.id, currentTime, self.symbol, new_vol, direction == 'BUY', float(price),
                                       dollar=False, order_id=id)
                        self.modifyOrder(existing_order, executed_order)
                        self.orders.get(id).quantity = new_vol
            else:
                self.replicateOrderbookSnapshot(currentTime)
    else:
        orders = self.oracle.trades_df.loc[self.oracle.trades_df.timestamp == currentTime]
        for index, order in orders.iterrows():
            self.placeOrder(currentTime, order = pd.DataFrame(order).T)


  def replicateOrderbookSnapshot(self, currentTime):
       log_print("Received notification of orderbook snapshot replication at: {}".format(currentTime))
       self.sendMessage(self.exchangeID, Message({"msg": "REPLICATE_ORDERBOOK_SNAPSHOT", "sender": self.id,
                                                  "symbol": self.symbol, "timestamp": str(currentTime)}))
       if self.log_orders: self.logEvent('REPLICATE_ORDERBOOK_SNAPSHOT', currentTime)


  def getWakeFrequency(self):
    return self.oracle.trades_df.iloc[0].timestamp - self.mkt_open