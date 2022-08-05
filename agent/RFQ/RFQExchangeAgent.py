from agent.FinancialAgent import FinancialAgent
from message.Message import Message
from util.OrderBook import OrderBook
from util.util import log_print

import datetime as dt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import pandas as pd
pd.set_option('display.max_rows', 500)

from copy import deepcopy


class ExchangeAgent(FinancialAgent):

  def __init__(self, id, name, type, mkt_open, mkt_close, symbols, book_freq='S', wide_book=False, pipeline_delay = 40000,
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

    #TODO: continue

    # don't need to maintain order books but agents will need to track prices for symbols (oracle?)

    def kernelInitializing (self, kernel):
      super().kernelInitializing(kernel)

      self.oracle = self.kernel.oracle

    def kernelTerminating (self):
      super().kernelTerminating()

      # If the oracle supports writing the fundamental value series for its
      # symbols, write them to disk.
      if hasattr(self.oracle, 'f_log'):
        for symbol in self.oracle.f_log:
          dfFund = pd.DataFrame(self.oracle.f_log[symbol])
          if not dfFund.empty:
            dfFund.set_index('FundamentalTime', inplace=True)
            self.writeLog(dfFund, filename='fundamental_{}'.format(symbol))
            log_print("Fundamental archival complete.")
      if self.book_freq is None: return
      else:
        # TODO: log trades? or just last trade?
        # previously logged order book snapshot at end
        pass
    
  def receiveMessage(self, currentTime, msg):
    super().receiveMessage(currentTime, msg)

    # Note that computation delay MUST be updated before any calls to sendMessage.
    self.setComputationDelay(self.computation_delay)

    # Is the exchange closed?  (This block only affects post-close, not pre-open.)
    if currentTime > self.mkt_close:
      # Most messages after close will receive a 'MKT_CLOSED' message in response.  A few things
      # might still be processed, like requests for final trade prices or such.
      if msg.body['msg'] in ['RFQ']:
        log_print("{} received {}: {}", self.name, msg.body['msg'], msg.body['order'])
        self.sendMessage(msg.body['sender'], Message({"msg": "MKT_CLOSED"}))

        # Don't do any further processing on these messages!
        return
      elif 'QUERY' in msg.body['msg']: # TODO: can agents see trades in RFQ
        # Specifically do allow querying after market close, so agents can get the
        # final trade of the day as their "daily close" price for a symbol.
        pass
      else:
        log_print("{} received {}, discarded: market is closed.", self.name, msg.body['msg'])
        self.sendMessage(msg.body['sender'], Message({"msg": "MKT_CLOSED"}))

        # Don't do any further processing on these messages!
        return

    # Log order messages only if that option is configured.  Log all other messages.
    if msg.body['msg'] in ['RFQ']:
      if self.log_orders: self.logEvent(msg.body['msg'], msg.body['order'].to_dict())
    else:
      self.logEvent(msg.body['msg'], msg.body['sender'])

    # Handle the DATA SUBSCRIPTION request and cancellation messages from the agents.
    if msg.body['msg'] in ["MARKET_DATA_SUBSCRIPTION_REQUEST", "MARKET_DATA_SUBSCRIPTION_CANCELLATION"]:
      log_print("{} received {} request from agent {}", self.name, msg.body['msg'], msg.body['sender'])
      self.updateSubscriptionDict(msg, currentTime)

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
    

    elif msg.body['msg'] == "RFQ":

      order = msg.body['order'] # TODO: replace with RFQ class?
      log_print("{} received RFQ: {}", self.name, order)
      if order.symbol not in self.symbols:
        log_print("RFQ discarded.  Unknown symbol: {}", order.symbol)
      else:
        # Hand the order to the order book for processing.
        # self.publishOrderBookData()
        #TODO: handle RFQ
        pass

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
        self.publishOrderBookData()
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
        self.publishOrderBookData()    