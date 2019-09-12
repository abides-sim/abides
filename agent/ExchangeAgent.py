# The ExchangeAgent expects a numeric agent id, printable name, agent type, timestamp to open and close trading,
# a list of equity symbols for which it should create order books, a frequency at which to archive snapshots
# of its order books, a pipeline delay (in ns) for order activity, the exchange computation delay (in ns),
# the levels of order stream history to maintain per symbol (maintains all orders that led to the last N trades),
# whether to log all order activity to the agent log, and a random state object (already seeded) to use
# for stochasticity.
from agent.FinancialAgent import FinancialAgent
from message.Message import Message
from util.OrderBook import OrderBook
from util.util import log_print, delist

import jsons as js
import pandas as pd
pd.set_option('display.max_rows', 500)

from copy import deepcopy


class ExchangeAgent(FinancialAgent):

  def __init__(self, id, name, type, mkt_open, mkt_close, symbols, book_freq='S', pipeline_delay = 40000,
               computation_delay = 1, stream_history = 0, log_orders = False, random_state = None):

    super().__init__(id, name, type, random_state)

    # Do not request repeated wakeup calls.
    self.reschedule = False

    # Store this exchange's open and close times.
    self.mkt_open = mkt_open
    self.mkt_close = mkt_close

    # Right now, only the exchange agent has a parallel processing pipeline delay.  This is an additional
    # delay added only to order activity (placing orders, etc) and not simple inquiries (market operating
    # hours, etc).
    self.pipeline_delay = pipeline_delay

    # Computation delay is applied on every wakeup call or message received.
    self.computation_delay = computation_delay

    # The exchange maintains an order stream of all orders leading to the last L trades
    # to support certain agents from the auction literature (GD, HBL, etc).
    self.stream_history = stream_history

    # Log all order activity?
    self.log_orders = log_orders

    # Create an order book for each symbol.
    self.order_books = {}

    for symbol in symbols:
      self.order_books[symbol] = OrderBook(self, symbol)

    # At what frequency will we archive the order books for visualization and analysis?
    self.book_freq = book_freq


  # The exchange agent overrides this to obtain a reference to an oracle.
  # This is needed to establish a "last trade price" at open (i.e. an opening
  # price) in case agents query last trade before any simulated trades are made.
  # This can probably go away once we code the opening cross auction.
  def kernelInitializing (self, kernel):
    super().kernelInitializing(kernel)

    self.oracle = self.kernel.oracle

    # Obtain opening prices (in integer cents).  These are not noisy right now.
    for symbol in self.order_books:
      try:
        self.order_books[symbol].last_trade = self.oracle.getDailyOpenPrice(symbol, self.mkt_open)
        log_print ("Opening price for {} is {}", symbol, self.order_books[symbol].last_trade)
      except AttributeError as e:
        log_print(str(e))


  # The exchange agent overrides this to additionally log the full depth of its
  # order books for the entire day.
  def kernelTerminating (self):
    super().kernelTerminating()

    # If the oracle supports writing the fundamental value series for its
    # symbols, write them to disk.
    if hasattr(self.oracle, 'f_log'):
      for symbol in self.oracle.f_log:
        dfFund = pd.DataFrame(self.oracle.f_log[symbol])
        dfFund.set_index('FundamentalTime', inplace=True)
        self.writeLog(dfFund, filename='fundamental_{}'.format(symbol))
      print("Fundamental archival complete.")

    if self.book_freq is None:
      return
    elif self.book_freq == 'all': # log all orderbook updates
      self.logOrderBook()
    else:
      # Iterate over the order books controlled by this exchange.
      for symbol in self.order_books:
        book = self.order_books[symbol]

        # Log full depth quotes (price, volume) from this order book at some pre-determined frequency.
        # Here we are looking at the actual log for this order book (i.e. are there snapshots to export,
        # independent of the requested frequency).
        if book.book_log:

          # This must already be sorted by time because it was a list of order book snapshots and time
          # only increases in our simulation.  BUT it can have duplicates if multiple orders happen
          # in the same nanosecond.  (This particularly happens if using nanoseconds as the discrete
          # but fine-grained unit for more theoretic studies.)
          dfLog = pd.DataFrame(book.book_log)
          dfLog.set_index('QuoteTime', inplace=True)

          # With multiple quotes in a nanosecond, use the last one, then resample to the requested freq.
          dfLog = dfLog[~dfLog.index.duplicated(keep='last')]
          dfLog.sort_index(inplace=True)
          dfLog = dfLog.resample(self.book_freq).ffill()
          dfLog.sort_index(inplace=True)

          # Create a fully populated index at the desired frequency from market open to close.
          # Then project the logged data into this complete index.
          time_idx = pd.date_range(self.mkt_open, self.mkt_close, freq=self.book_freq, closed='right')
          dfLog = dfLog.reindex(time_idx, method='ffill')
          dfLog.sort_index(inplace=True)

          dfLog = dfLog.stack()
          dfLog.sort_index(inplace=True)

          # Get the full range of quotes at the finest possible resolution.
          quotes = sorted(dfLog.index.get_level_values(1).unique())
          min_quote = quotes[0]
          max_quote = quotes[-1]
          quotes = range(min_quote, max_quote+1)

          # Restructure the log to have multi-level rows of all possible pairs of time and quote
          # with volume as the only column.
          filledIndex = pd.MultiIndex.from_product([time_idx, quotes], names=['time','quote'])
          dfLog = dfLog.reindex(filledIndex)
          dfLog.fillna(0, inplace=True)

          dfLog.rename('Volume')

          df = pd.DataFrame(index=dfLog.index)
          df['Volume'] = dfLog


          # Archive the order book snapshots directly to a file named with the symbol, rather than
          # to the exchange agent log.
          self.writeLog(df, filename='orderbook_{}'.format(symbol))
      print ("Order book archival complete.")

  def receiveMessage(self, currentTime, msg):
    super().receiveMessage(currentTime, msg)

    # Unless the intent of an experiment is to examine computational issues within an Exchange,
    # it will typically have either 1 ns delay (near instant but cannot process multiple orders
    # in the same atomic time unit) or 0 ns delay (can process any number of orders, always in
    # the atomic time unit in which they are received).  This is separate from, and additional
    # to, any parallel pipeline delay imposed for order book activity.

    # Note that computation delay MUST be updated before any calls to sendMessage.
    self.setComputationDelay(self.computation_delay)

    # Is the exchange closed?  (This block only affects post-close, not pre-open.)
    if currentTime > self.mkt_close:
      # Most messages after close will receive a 'MKT_CLOSED' message in response.  A few things
      # might still be processed, like requests for final trade prices or such.
      if msg.body['msg'] in ['LIMIT_ORDER', 'CANCEL_ORDER']:
        log_print("{} received {}: {}", self.name, msg.body['msg'], msg.body['order'])
        self.sendMessage(msg.body['sender'], Message({"msg": "MKT_CLOSED"}))

        # Don't do any further processing on these messages!
        return
      elif 'QUERY' in msg.body['msg']:
        # Specifically do allow querying after market close, so agents can get the
        # final trade of the day as their "daily close" price for a symbol.
        pass
      else:
        log_print("{} received {}, discarded: market is closed.", self.name, msg.body['msg'])
        self.sendMessage(msg.body['sender'], Message({"msg": "MKT_CLOSED"}))

        # Don't do any further processing on these messages!
        return

    # Log order messages only if that option is configured.  Log all other messages.
    if msg.body['msg'] in ['LIMIT_ORDER', 'CANCEL_ORDER']:
      if self.log_orders: self.logEvent(msg.body['msg'], js.dump(msg.body['order']))
    else:
      self.logEvent(msg.body['msg'], msg.body['sender'])

    # Handle all message types understood by this exchange.
    if msg.body['msg'] == "WHEN_MKT_OPEN":
      log_print("{} received WHEN_MKT_OPEN request from agent {}", self.name, msg.body['sender'])

      # The exchange is permitted to respond to requests for simple immutable data (like "what are your
      # hours?") instantly.  This does NOT include anything that queries mutable data, like equity
      # quotes or trades.
      self.setComputationDelay(0)

      self.sendMessage(msg.body['sender'], Message({"msg": "WHEN_MKT_OPEN", "data": self.mkt_open}))
    elif msg.body['msg'] == "WHEN_MKT_CLOSE":
      log_print("{} received WHEN_MKT_CLOSE request from agent {}", self.name, msg.body['sender'])

      # The exchange is permitted to respond to requests for simple immutable data (like "what are your
      # hours?") instantly.  This does NOT include anything that queries mutable data, like equity
      # quotes or trades.
      self.setComputationDelay(0)

      self.sendMessage(msg.body['sender'], Message({"msg": "WHEN_MKT_CLOSE", "data": self.mkt_close}))
    elif msg.body['msg'] == "QUERY_LAST_TRADE":
      symbol = msg.body['symbol']
      if symbol not in self.order_books:
        log_print("Last trade request discarded.  Unknown symbol: {}", symbol)
      else:
        log_print("{} received QUERY_LAST_TRADE ({}) request from agent {}", self.name, symbol, msg.body['sender'])

        # Return the single last executed trade price (currently not volume) for the requested symbol.
        # This will return the average share price if multiple executions resulted from a single order.
        self.sendMessage(msg.body['sender'], Message({"msg": "QUERY_LAST_TRADE", "symbol": symbol,
                                                      "data": self.order_books[symbol].last_trade,
                                                      "mkt_closed": True if currentTime > self.mkt_close else False}))
    elif msg.body['msg'] == "QUERY_SPREAD":
      symbol = msg.body['symbol']
      depth = msg.body['depth']
      if symbol not in self.order_books:
        log_print("Bid-ask spread request discarded.  Unknown symbol: {}", symbol)
      else:
        log_print("{} received QUERY_SPREAD ({}:{}) request from agent {}", self.name, symbol, depth,
                  msg.body['sender'])

        # Return the requested depth on both sides of the order book for the requested symbol.
        # Returns price levels and aggregated volume at each level (not individual orders).
        self.sendMessage(msg.body['sender'], Message({"msg": "QUERY_SPREAD", "symbol": symbol, "depth": depth,
                                                      "bids": self.order_books[symbol].getInsideBids(depth),
                                                      "asks": self.order_books[symbol].getInsideAsks(depth),
                                                      "data": self.order_books[symbol].last_trade,
                                                      "mkt_closed": True if currentTime > self.mkt_close else False,
                                                      "book": ''}))

        # It is possible to also send the pretty-printed order book to the agent for logging, but forcing pretty-printing
        # of a large order book is very slow, so we should only do it with good reason.  We don't currently
        # have a configurable option for it.
        # "book": self.order_books[symbol].prettyPrint(silent=True) }))
    elif msg.body['msg'] == "QUERY_ORDER_STREAM":
      symbol = msg.body['symbol']
      length = msg.body['length']

      if symbol not in self.order_books:
        log_print("Order stream request discarded.  Unknown symbol: {}", symbol)
      else:
        log_print("{} received QUERY_ORDER_STREAM ({}:{}) request from agent {}", self.name, symbol, length,
                  msg.body['sender'])

      # We return indices [1:length] inclusive because the agent will want "orders leading up to the last
      # L trades", and the items under index 0 are more recent than the last trade.
      self.sendMessage(msg.body['sender'], Message({"msg": "QUERY_ORDER_STREAM", "symbol": symbol, "length": length,
                                                    "mkt_closed": True if currentTime > self.mkt_close else False,
                                                    "orders": self.order_books[symbol].history[1:length + 1]
                                                    }))
    elif msg.body['msg'] == "LIMIT_ORDER":
      order = msg.body['order']
      log_print("{} received LIMIT_ORDER: {}", self.name, order)
      if order.symbol not in self.order_books:
        log_print("Order discarded.  Unknown symbol: {}", order.symbol)
      else:
        # Hand the order to the order book for processing.
        self.order_books[order.symbol].handleLimitOrder(deepcopy(order))
    elif msg.body['msg'] == "CANCEL_ORDER":
      # Note: this is somewhat open to abuse, as in theory agents could cancel other agents' orders.
      # An agent could also become confused if they receive a (partial) execution on an order they
      # then successfully cancel, but receive the cancel confirmation first.  Things to think about
      # for later...
      order = msg.body['order']
      log_print("{} received CANCEL_ORDER: {}", self.name, order)
      if order.symbol not in self.order_books:
        log_print("Cancellation request discarded.  Unknown symbol: {}", order.symbol)
      else:
        # Hand the order to the order book for processing.
        self.order_books[order.symbol].cancelOrder(deepcopy(order))
    elif msg.body['msg'] == 'MODIFY_ORDER':
      # Replace an existing order with a modified order.  There could be some timing issues
      # here.  What if an order is partially executed, but the submitting agent has not
      # yet received the norification, and submits a modification to the quantity of the
      # (already partially executed) order?  I guess it is okay if we just think of this
      # as "delete and then add new" and make it the agent's problem if anything weird
      # happens.
      order = msg.body['order']
      new_order = msg.body['new_order']
      log_print("{} received MODIFY_ORDER: {}, new order: {}".format(self.name, order, new_order))
      if order.symbol not in self.order_books:
        log_print("Modification request discarded.  Unknown symbol: {}".format(order.symbol))
      else:
        self.order_books[order.symbol].modifyOrder(deepcopy(order), deepcopy(new_order))

  def sendMessage (self, recipientID, msg):
    # The ExchangeAgent automatically applies appropriate parallel processing pipeline delay
    # to those message types which require it.
    # TODO: probably organize the order types into categories once there are more, so we can
    # take action by category (e.g. ORDER-related messages) instead of enumerating all message
    # types to be affected.
    if msg.body['msg'] in ['ORDER_ACCEPTED', 'ORDER_CANCELLED', 'ORDER_EXECUTED']:
      # Messages that require order book modification (not simple queries) incur the additional
      # parallel processing delay as configured.
      super().sendMessage(recipientID, msg, delay = self.pipeline_delay)
      if self.log_orders: self.logEvent(msg.body['msg'], js.dump(msg.body['order']))
    else:
      # Other message types incur only the currently-configured computation delay for this agent.
      super().sendMessage(recipientID, msg)




  # Logs the orderbook as a pandas dataframe. This method produces the orderbook with timestamps equal to the timestamps
  # the orderbook was updated (tick by tick)
  def logOrderBook(self):
    for symbol in self.order_books:
      depth = 10
      book = self.order_books[symbol]
      dfLog = pd.DataFrame([book.bid_levels_price_dict, book.bid_levels_size_dict,
                            book.ask_levels_price_dict, book.ask_levels_size_dict]).T
      dfLog.columns = ['bid_level_prices', 'bid_level_sizes', 'ask_level_prices', 'ask_level_sizes']
      cols = delist([["ask_price_{}".format(level), "ask_size_{}".format(level), "bid_price_{}".format(level),
                      "bid_size_{}".format(level)] for level in range(1, depth+1)])
      orderbook_df = pd.DataFrame(columns=cols, index=dfLog.index)
      for index, row in orderbook_df.iterrows():
        ask_level_prices_dict = dfLog.loc[index].ask_level_prices
        ask_level_sizes_dict = dfLog.loc[index].ask_level_sizes
        bid_level_prices_dict = dfLog.loc[index].bid_level_prices
        bid_level_sizes_dict = dfLog.loc[index].bid_level_sizes
        for i in range(1, depth+1):
          try:
            row['ask_price_{}'.format(i)], row['ask_size_{}'.format(i)] = ask_level_prices_dict[i], \
                                                                          ask_level_sizes_dict[i]
          except KeyError:
            log_print("One Orderbook side empty")
          try:
            row['bid_price_{}'.format(i)], row['bid_size_{}'.format(i)] = bid_level_prices_dict[i], \
                                                                          bid_level_sizes_dict[i]
          except KeyError:
            log_print("One Orderbook side empty")
      self.writeLog(orderbook_df, filename='orderbook_{}'.format(symbol))

  # Simple accessor methods for the market open and close times.
  def getMarketOpen(self):
    return self.__mkt_open

  def getMarketClose(self):
    return self.__mkt_close

