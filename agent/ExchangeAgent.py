# The ExchangeAgent expects an agent id, printable name, timestamp to open and close trading,
# and a list of equity symbols for which it should create order books.
from agent.FinancialAgent import FinancialAgent
from message.Message import Message
from util.OrderBook import OrderBook
from util.util import print

import sys

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)

from copy import deepcopy

class ExchangeAgent(FinancialAgent):

  def __init__(self, id, name, mkt_open, mkt_close, symbols, book_freq='S', pipeline_delay = 40000,
               computation_delay = 1, stream_history = 0):

    super().__init__(id, name)

    # Do not request repeated wakeup calls.
    self.reschedule = False

    # Store this exchange's open and close times.
    self.mkt_open = mkt_open
    self.mkt_close = mkt_close

    # Right now, only the exchange agent has a parallel processing pipeline delay.
    self.pipeline_delay = pipeline_delay
    self.computation_delay = computation_delay

    # The exchange maintains an order stream of all orders leading to the last L trades
    # to support certain agents from the auction literature (GD, HBL, etc).
    self.stream_history = stream_history

    # Create an order book for each symbol.
    self.order_books = {}

    for symbol in symbols:
      self.order_books[symbol] = OrderBook(self, symbol)
      
    # At what frequency will we archive the order books for visualization and analysis?
    self.book_freq = book_freq


  # The exchange agent overrides this to obtain a reference to a DataOracle.
  # This is needed to establish a "last trade price" at open (i.e. an opening
  # price) in case agents query last trade before any simulated trades are made.
  # This can probably go away once we code the opening cross auction.
  def kernelInitializing (self, kernel):
    super().kernelInitializing(kernel)

    self.oracle = self.kernel.oracle

    # Obtain opening prices (in integer cents).  These are not noisy right now.
    for symbol in self.order_books:
      self.order_books[symbol].last_trade = self.oracle.getDailyOpenPrice(symbol, self.mkt_open)
      print ("Opening price for {} is {}".format(symbol, self.order_books[symbol].last_trade))


  # The exchange agent overrides this to additionally log the full depth of its
  # order books for the entire day.
  def kernelTerminating (self):
    super().kernelTerminating()

    # Skip order book dump if requested.
    if self.book_freq is None: return

    for symbol in self.order_books:
      book = self.order_books[symbol]

      # Log full depth quotes (price, volume) from this order book at some pre-determined frequency.
      if book.book_log:

        ### THE FAST WAY

        # This must already be sorted by time because it was a list of order book snapshots and time
        # only increases in our simulation.  BUT it can have duplicates if multiple orders happen
        # in the same nanosecond.  (This particularly happens if using nanoseconds as the discrete
        # but fine-grained unit for more theoretic studies.)
        dfLog = pd.DataFrame(book.book_log)
        dfLog.set_index('QuoteTime', inplace=True)

        if True:
          dfLog = dfLog[~dfLog.index.duplicated(keep='last')]
          dfLog.sort_index(inplace=True)
          dfLog = dfLog.resample(self.book_freq).ffill()
          dfLog.sort_index(inplace=True)
  
          time_idx = pd.date_range(self.mkt_open, self.mkt_close, freq=self.book_freq, closed='right')
          dfLog = dfLog.reindex(time_idx, method='ffill')
          dfLog.sort_index(inplace=True)

          dfLog = dfLog.stack()
          dfLog.sort_index(inplace=True)
  
          quotes = sorted(dfLog.index.get_level_values(1).unique())
          min_quote = quotes[0]
          max_quote = quotes[-1]
          quotes = range(min_quote, max_quote+1)
  
          filledIndex = pd.MultiIndex.from_product([time_idx, quotes], names=['time','quote'])
          dfLog = dfLog.reindex(filledIndex)
          dfLog.fillna(0, inplace=True)
  
          dfLog.rename('Volume')
  
          df = pd.DataFrame(index=dfLog.index)
          df['Volume'] = dfLog


        ### THE SLOW WAY

        if False:

          # Make a MultiIndex dataframe of (Seconds, QuotePrice) -> Volume, giving the quote prices and volumes
          # at the end of each second the market was open.
          seconds = pd.date_range(self.mkt_open, self.mkt_close, freq=self.book_freq, closed='right')
          quotes = dfLog.columns
  
          df = pd.DataFrame(index=pd.MultiIndex.from_product([seconds, quotes], names=['time','quote']))
          df['Volume'] = 0
  
          df.sort_index(inplace=True)
  
          logWriteStart = pd.Timestamp('now')
          i = 0
  
          for idx, row in df.iterrows():
            if i % 1000 == 0:
              print ("Exchange writing order book log, interval {}, wallclock elapsed {}".format(idx[0], pd.Timestamp('now') - logWriteStart), override=True)
  
            best = dfLog.index.asof(idx[0])
            if pd.isnull(best): continue
            df.loc[idx,'Volume'] = dfLog.loc[best,idx[1]]
  
            i += 1
  
          print ("Exchange sorting order book index.", override=True)
          df.sort_index(inplace=True)
  
          # Create a filled version of the index without gaps from min to max quote price.
          min_quote = df.index.get_level_values(1)[0]
          max_quote = df.index.get_level_values(1)[-1]
          quotes = range(min_quote, max_quote+1)
  
          # Create the new index and move the data over.
          print ("Exchange reindexing order book.", override=True)
          filledIndex = pd.MultiIndex.from_product([seconds, quotes], names=['time','quote'])
          df = df.reindex(filledIndex)
  
          # NaNs represent that there is NO volume at this quoted price at this time, so they should become zero.
          df.fillna(0, inplace=True)
  
          print ("Exchange archiving order book.", override=True)

        self.writeLog(df, filename='orderbook_{}'.format(symbol))

        print ("Order book archival complete.", override=True)
   

  def receiveMessage (self, currentTime, msg):
    super().receiveMessage(currentTime, msg)

    # Exchanges currently get a very fast (but not instant) computation delay of 1 ns for handling
    # all order types.  Note that computation delay MUST be updated before any calls to sendMessage.
    self.setComputationDelay(self.computation_delay)

    # We're closed.
    if currentTime > self.mkt_close:
      # Most messages after close will receive a 'MKT_CLOSED' message in response.  A few things
      # might still be processed, like requests for final trade prices or such.
      if msg.body['msg'] in ['LIMIT_ORDER', 'CANCEL_ORDER']:
        print ("{} received {}: {}".format(self.name, msg.body['msg'], msg.body['order']))
        self.sendMessage(msg.body['sender'], Message({ "msg": "MKT_CLOSED" }))

        # Don't do any further processing on these messages!
        return
      elif 'QUERY' in msg.body['msg']:
        # Specifically do allow querying after market close, so agents can get the
        # final trade of the day as their "daily close" price for a symbol.
        pass
      else:
        print ("{} received {}, discarded: market is closed.".format(self.name, msg.body['msg']))
        self.sendMessage(msg.body['sender'], Message({ "msg": "MKT_CLOSED" }))

        # Don't do any further processing on these messages!
        return

    # Log all received messages.
    if msg.body['msg'] in ['LIMIT_ORDER', 'CANCEL_ORDER']:
      self.logEvent(msg.body['msg'], msg.body['order'])
    else:
      self.logEvent(msg.body['msg'], msg.body['sender'])

    # Handle message types understood by this exchange.
    if msg.body['msg'] == "WHEN_MKT_OPEN":
      print ("{} received WHEN_MKT_OPEN request from agent {}".format(self.name, msg.body['sender']))

      # The exchange is permitted to respond to requests for simple immutable data (like "what are your
      # hours?") instantly.  This does NOT include anything that queries mutable data, like equity
      # quotes or trades.
      self.setComputationDelay(0)

      self.sendMessage(msg.body['sender'], Message({ "msg": "WHEN_MKT_OPEN", "data": self.mkt_open }))
    elif msg.body['msg'] == "WHEN_MKT_CLOSE":
      print ("{} received WHEN_MKT_CLOSE request from agent {}".format(self.name, msg.body['sender']))

      # The exchange is permitted to respond to requests for simple immutable data (like "what are your
      # hours?") instantly.  This does NOT include anything that queries mutable data, like equity
      # quotes or trades.
      self.setComputationDelay(0)

      self.sendMessage(msg.body['sender'], Message({ "msg": "WHEN_MKT_CLOSE", "data": self.mkt_close }))
    elif msg.body['msg'] == "QUERY_LAST_TRADE":
      symbol = msg.body['symbol']
      if symbol not in self.order_books:
        print ("Last trade request discarded.  Unknown symbol: {}".format(symbol))
      else:
        print ("{} received QUERY_LAST_TRADE ({}) request from agent {}".format(self.name, symbol, msg.body['sender']))

        self.sendMessage(msg.body['sender'], Message({ "msg": "QUERY_LAST_TRADE", "symbol": symbol,
             "data": self.order_books[symbol].last_trade, "mkt_closed": True if currentTime > self.mkt_close else False }))
    elif msg.body['msg'] == "QUERY_SPREAD":
      symbol = msg.body['symbol']
      depth = msg.body['depth']
      if symbol not in self.order_books:
        print ("Bid-ask spread request discarded.  Unknown symbol: {}".format(symbol))
      else:
        print ("{} received QUERY_SPREAD ({}:{}) request from agent {}".format(self.name, symbol, depth, msg.body['sender']))
        self.sendMessage(msg.body['sender'], Message({ "msg": "QUERY_SPREAD", "symbol": symbol, "depth": depth,
             "bids": self.order_books[symbol].getInsideBids(depth), "asks": self.order_books[symbol].getInsideAsks(depth),
             "data": self.order_books[symbol].last_trade, "mkt_closed": True if currentTime > self.mkt_close else False,
             "book": self.order_books[symbol].prettyPrint(silent=True) }))
    elif msg.body['msg'] == "QUERY_ORDER_STREAM":
      symbol = msg.body['symbol']
      length = msg.body['length']

      if symbol not in self.order_books:
        print ("Order stream request discarded.  Unknown symbol: {}".format(symbol))
      else:
        print ("{} received QUERY_ORDER_STREAM ({}:{}) request from agent {}".format(self.name, symbol, length, msg.body['sender']))
      
      # We return indices [1:length] inclusive because the agent will want "orders leading up to the last
      # L trades", and the items under index 0 are more recent than the last trade.
      self.sendMessage(msg.body['sender'], Message({ "msg": "QUERY_ORDER_STREAM", "symbol": symbol, "length": length,
           "mkt_closed": True if currentTime > self.mkt_close else False,
           "orders": self.order_books[symbol].history[1:length+1]
           }))
    elif msg.body['msg'] == "LIMIT_ORDER":
      order = msg.body['order']
      print ("{} received LIMIT_ORDER: {}".format(self.name, order))
      if order.symbol not in self.order_books:
        print ("Order discarded.  Unknown symbol: {}".format(order.symbol))
      else:
        self.order_books[order.symbol].handleLimitOrder(deepcopy(order))
    elif msg.body['msg'] == "CANCEL_ORDER":
      # Note: this is somewhat open to abuse, as in theory agents could cancel other agents' orders.
      # An agent could also become confused if they receive a (partial) execution on an order they
      # then successfully cancel, but receive the cancel confirmation first.  Things to think about
      # for later...
      order = msg.body['order']
      print ("{} received CANCEL_ORDER: {}".format(self.name, order))
      if order.symbol not in self.order_books:
        print ("Cancellation request discarded.  Unknown symbol: {}".format(order.symbol))
      else:
        self.order_books[order.symbol].cancelOrder(deepcopy(order))
      

  def sendMessage (self, recipientID, msg):
    # The ExchangeAgent automatically applies appropriate parallel processing pipeline delay
    # to exactly those message types which require it.
    # TODO: probably organize the order types into constant categories once there are more.
    if msg.body['msg'] in ['ORDER_ACCEPTED', 'ORDER_CANCELLED', 'ORDER_EXECUTED']:
      super().sendMessage(recipientID, msg, delay = self.pipeline_delay)
      self.logEvent(msg.body['msg'], msg.body['order'])
    else:
      super().sendMessage(recipientID, msg)


  def getMarketOpen(self):
    return self.__mkt_open

  def getMarketClose(self):
    return self.__mkt_close

