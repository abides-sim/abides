from agent.FinancialAgent import FinancialAgent
from agent.ExchangeAgent import ExchangeAgent
from message.Message import Message
from util.order.LimitOrder import LimitOrder
from util.order.MarketOrder import MarketOrder
from util.util import log_print

from copy import deepcopy
import sys

# The TradingAgent class (via FinancialAgent, via Agent) is intended as the
# base class for all trading agents (i.e. not things like exchanges) in a
# market simulation.  It handles a lot of messaging (inbound and outbound)
# and state maintenance automatically, so subclasses can focus just on
# implementing a strategy without too much bookkeeping.
class TradingAgent(FinancialAgent):

  def __init__(self, id, name, type, random_state=None, starting_cash=100000, log_orders=False, log_to_file=True):
    # Base class init.
    super().__init__(id, name, type, random_state, log_to_file)

    # We don't yet know when the exchange opens or closes.
    self.mkt_open = None
    self.mkt_close = None

    # Log order activity?
    self.log_orders = log_orders

    # Log all activity to file?
    if log_orders is None:
      self.log_orders = False
      self.log_to_file = False

    # Store starting_cash in case we want to refer to it for performance stats.
    # It should NOT be modified.  Use the 'CASH' key in self.holdings.
    # 'CASH' is always in cents!  Note that agents are limited by their starting
    # cash, currently without leverage.  Taking short positions is permitted,
    # but does NOT increase the amount of at-risk capital allowed.
    self.starting_cash = starting_cash

    # TradingAgent has constants to support simulated market orders.
    self.MKT_BUY = sys.maxsize
    self.MKT_SELL = 0

    # The base TradingAgent will track its holdings and outstanding orders.
    # Holdings is a dictionary of symbol -> shares.  CASH is a special symbol
    # worth one cent per share.  Orders is a dictionary of active, open orders
    # (not cancelled, not fully executed) keyed by order_id.
    self.holdings = { 'CASH' : starting_cash }
    self.orders = {}

    # The base TradingAgent also tracks last known prices for every symbol
    # for which it has received as QUERY_LAST_TRADE message.  Subclass
    # agents may use or ignore this as they wish.  Note that the subclass
    # agent must request pricing when it wants it.  This agent does NOT
    # automatically generate such requests, though it has a helper function
    # that can be used to make it happen.
    self.last_trade = {}

    # used in subscription mode to record the timestamp for which the data was current in the ExchangeAgent
    self.exchange_ts = {}

    # When a last trade price comes in after market close, the trading agent
    # automatically records it as the daily close price for a symbol.
    self.daily_close_price = {}
    
    self.nav_diff = 0
    self.basket_size = 0

    # The agent remembers the last known bids and asks (with variable depth,
    # showing only aggregate volume at each price level) when it receives
    # a response to QUERY_SPREAD.
    self.known_bids = {}
    self.known_asks = {}

    # The agent remembers the order history communicated by the exchange
    # when such is requested by an agent (for example, a heuristic belief
    # learning agent).
    self.stream_history = {}

    # The agent records the total transacted volume in the exchange for a given symbol and lookback period
    self.transacted_volume = {}

    # Each agent can choose to log the orders executed
    self.executed_orders = []

    # For special logging at the first moment the simulator kernel begins
    # running (which is well after agent init), it is useful to keep a simple
    # boolean flag.
    self.first_wake = True

    # Remember whether we have already passed the exchange close time, as far
    # as we know.
    self.mkt_closed = False

    # This is probably a transient feature, but for now we permit the exchange
    # to return the entire order book sometimes, for development and debugging.
    # It is very expensive to pass this around, and breaks "simulation physics",
    # but can really help understand why agents are making certain decisions.
    # Subclasses should NOT rely on this feature as part of their strategy,
    # as it will go away.
    self.book = ''


  # Simulation lifecycle messages.

  def kernelStarting(self, startTime):
    # self.kernel is set in Agent.kernelInitializing()
    self.logEvent('STARTING_CASH', self.starting_cash, True)

    # Find an exchange with which we can place orders.  It is guaranteed
    # to exist by now (if there is one).
    self.exchangeID = self.kernel.findAgentByType(ExchangeAgent)

    log_print ("Agent {} requested agent of type Agent.ExchangeAgent.  Given Agent ID: {}",
               self.id, self.exchangeID)

    # Request a wake-up call as in the base Agent.
    super().kernelStarting(startTime)


  def kernelStopping (self):
    # Always call parent method to be safe.
    super().kernelStopping()

    # Print end of day holdings.
    self.logEvent('FINAL_HOLDINGS', self.fmtHoldings(self.holdings))
    self.logEvent('FINAL_CASH_POSITION', self.holdings['CASH'], True)

    # Mark to market.
    cash = self.markToMarket(self.holdings)

    self.logEvent('ENDING_CASH', cash, True)
    print ("Final holdings for {}: {}.  Marked to market: {}".format(self.name, self.fmtHoldings(self.holdings),
                                                                     cash))
    
    # Record final results for presentation/debugging.  This is an ugly way
    # to do this, but it is useful for now.
    mytype = self.type
    gain = cash - self.starting_cash

    if mytype in self.kernel.meanResultByAgentType:
      self.kernel.meanResultByAgentType[mytype] += gain
      self.kernel.agentCountByType[mytype] += 1
    else:
      self.kernel.meanResultByAgentType[mytype] = gain
      self.kernel.agentCountByType[mytype] = 1


  # Simulation participation messages.

  def wakeup (self, currentTime):
    super().wakeup(currentTime)

    if self.first_wake:
      # Log initial holdings.
      self.logEvent('HOLDINGS_UPDATED', self.holdings)
      self.first_wake = False

    if self.mkt_open is None:
      # Ask our exchange when it opens and closes.
      self.sendMessage(self.exchangeID, Message({ "msg" : "WHEN_MKT_OPEN", "sender": self.id }))
      self.sendMessage(self.exchangeID, Message({ "msg" : "WHEN_MKT_CLOSE", "sender": self.id }))

    # For the sake of subclasses, TradingAgent now returns a boolean
    # indicating whether the agent is "ready to trade" -- has it received
    # the market open and closed times, and is the market not already closed.
    return (self.mkt_open and self.mkt_close) and not self.mkt_closed

  def requestDataSubscription(self, symbol, levels, freq):
      self.sendMessage(recipientID = self.exchangeID,
                       msg = Message({"msg": "MARKET_DATA_SUBSCRIPTION_REQUEST",
                                      "sender": self.id, "symbol": symbol, "levels": levels, "freq": freq}))

  # Used by any Trading Agent subclass to cancel subscription to market data from the Exchange Agent
  def cancelDataSubscription(self, symbol):
    self.sendMessage(recipientID=self.exchangeID,
                     msg=Message({"msg": "MARKET_DATA_SUBSCRIPTION_CANCELLATION",
                                  "sender": self.id, "symbol": symbol}))


  def receiveMessage (self, currentTime, msg):
    super().receiveMessage(currentTime, msg)

    # Do we know the market hours?
    had_mkt_hours = self.mkt_open is not None and self.mkt_close is not None

    # Record market open or close times.
    if msg.body['msg'] == "WHEN_MKT_OPEN":
      self.mkt_open = msg.body['data']

      log_print ("Recorded market open: {}", self.kernel.fmtTime(self.mkt_open))

    elif msg.body['msg'] == "WHEN_MKT_CLOSE":
      self.mkt_close = msg.body['data']

      log_print ("Recorded market close: {}", self.kernel.fmtTime(self.mkt_close))

    elif msg.body['msg'] == "ORDER_EXECUTED":
      # Call the orderExecuted method, which subclasses should extend.  This parent
      # class could implement default "portfolio tracking" or "returns tracking"
      # behavior.
      order = msg.body['order']

      self.orderExecuted(order)

    elif msg.body['msg'] == "ORDER_ACCEPTED":
      # Call the orderAccepted method, which subclasses should extend.
      order = msg.body['order']

      self.orderAccepted(order)

    elif msg.body['msg'] == "ORDER_CANCELLED":
      # Call the orderCancelled method, which subclasses should extend.
      order = msg.body['order']

      self.orderCancelled(order)

    elif msg.body['msg'] == "MKT_CLOSED":
      # We've tried to ask the exchange for something after it closed.  Remember this
      # so we stop asking for things that can't happen.

      self.marketClosed()

    elif msg.body['msg'] == 'QUERY_LAST_TRADE':
      # Call the queryLastTrade method, which subclasses may extend.
      # Also note if the market is closed.
      if msg.body['mkt_closed']: self.mkt_closed = True

      self.queryLastTrade(msg.body['symbol'], msg.body['data'])

    elif msg.body['msg'] == 'QUERY_SPREAD':
      # Call the querySpread method, which subclasses may extend.
      # Also note if the market is closed.
      if msg.body['mkt_closed']: self.mkt_closed = True

      self.querySpread(msg.body['symbol'], msg.body['data'], msg.body['bids'], msg.body['asks'], msg.body['book'])

    elif msg.body['msg'] == 'QUERY_ORDER_STREAM':
      # Call the queryOrderStream method, which subclasses may extend.
      # Also note if the market is closed.
      if msg.body['mkt_closed']: self.mkt_closed = True

      self.queryOrderStream(msg.body['symbol'], msg.body['orders'])

    elif msg.body['msg'] == 'QUERY_TRANSACTED_VOLUME':
      if msg.body['mkt_closed']: self.mkt_closed = True
      self.query_transacted_volume(msg.body['symbol'], msg.body['transacted_volume'])

    elif msg.body['msg'] == 'MARKET_DATA':
      self.handleMarketData(msg)

    # Now do we know the market hours?
    have_mkt_hours = self.mkt_open is not None and self.mkt_close is not None

    # Once we know the market open and close times, schedule a wakeup call for market open.
    # Only do this once, when we first have both items.
    if have_mkt_hours and not had_mkt_hours:
      # Agents are asked to generate a wake offset from the market open time.  We structure
      # this as a subclass request so each agent can supply an appropriate offset relative
      # to its trading frequency.
      ns_offset = self.getWakeFrequency()

      self.setWakeup(self.mkt_open + ns_offset)


  # Used by any Trading Agent subclass to query the last trade price for a symbol.
  # This activity is not logged.
  def getLastTrade (self, symbol):
    self.sendMessage(self.exchangeID, Message({ "msg" : "QUERY_LAST_TRADE", "sender": self.id,
                                                "symbol" : symbol })) 


  # Used by any Trading Agent subclass to query the current spread for a symbol.
  # This activity is not logged.
  def getCurrentSpread (self, symbol, depth=1):
    self.sendMessage(self.exchangeID, Message({ "msg" : "QUERY_SPREAD", "sender": self.id,
                                                "symbol" : symbol, "depth" : depth }))


  # Used by any Trading Agent subclass to query the recent order stream for a symbol.
  def getOrderStream (self, symbol, length=1):
    self.sendMessage(self.exchangeID, Message({ "msg" : "QUERY_ORDER_STREAM", "sender": self.id,
                                                "symbol" : symbol, "length" : length }))

  def get_transacted_volume(self, symbol, lookback_period='10min'):
    """ Used by any trading agent subclass to query the total transacted volume in a given lookback period """
    self.sendMessage(self.exchangeID, Message({ "msg": "QUERY_TRANSACTED_VOLUME", "sender": self.id,
                                                "symbol": symbol, "lookback_period": lookback_period}))

  # Used by any Trading Agent subclass to place a limit order.  Parameters expect:
  # string (valid symbol), int (positive share quantity), bool (True == BUY), int (price in cents).
  # The call may optionally specify an order_id (otherwise global autoincrement is used) and
  # whether cash or risk limits should be enforced or ignored for the order.
  def placeLimitOrder (self, symbol, quantity, is_buy_order, limit_price, order_id=None, ignore_risk = True, tag = None):
    order = LimitOrder(self.id, self.currentTime, symbol, quantity, is_buy_order, limit_price, order_id, tag)

    if quantity > 0:
      # Test if this order can be permitted given our at-risk limits.
      new_holdings = self.holdings.copy()

      q = order.quantity if order.is_buy_order else -order.quantity

      if order.symbol in new_holdings: new_holdings[order.symbol] += q
      else: new_holdings[order.symbol] = q

      # If at_risk is lower, always allow.  Otherwise, new_at_risk must be below starting cash.
      if not ignore_risk:
        # Compute before and after at-risk capital.
        at_risk = self.markToMarket(self.holdings) - self.holdings['CASH']
        new_at_risk = self.markToMarket(new_holdings) - new_holdings['CASH']

        if (new_at_risk > at_risk) and (new_at_risk > self.starting_cash):
          log_print ("TradingAgent ignored limit order due to at-risk constraints: {}\n{}", order, self.fmtHoldings(self.holdings))
          return

      # Copy the intended order for logging, so any changes made to it elsewhere
      # don't retroactively alter our "as placed" log of the order.  Eventually
      # it might be nice to make the whole history of the order into transaction
      # objects inside the order (we're halfway there) so there CAN be just a single
      # object per order, that never alters its original state, and eliminate all these copies.
      self.orders[order.order_id] = deepcopy(order)
      self.sendMessage(self.exchangeID, Message({ "msg" : "LIMIT_ORDER", "sender": self.id,
                                                  "order" : order })) 

      # Log this activity.
      if self.log_orders: self.logEvent('ORDER_SUBMITTED', order.to_dict())

    else:
      log_print ("TradingAgent ignored limit order of quantity zero: {}", order)

  def placeMarketOrder(self, symbol, quantity, is_buy_order, order_id=None, ignore_risk = True, tag=None):
    """
      Used by any Trading Agent subclass to place a market order. The market order is created as multiple limit orders
      crossing the spread walking the book until all the quantities are matched.
      :param symbol (str):        name of the stock traded
      :param quantity (int):      order quantity
      :param is_buy_order (bool): True if Buy else False
      :param order_id:            Order ID for market replay
      :param ignore_risk (bool):  Determines whether cash or risk limits should be enforced or ignored for the order
      :return:
    """
    order = MarketOrder(self.id, self.currentTime, symbol, quantity, is_buy_order, order_id)
    if quantity > 0:
      # compute new holdings
      new_holdings = self.holdings.copy()
      q = order.quantity if order.is_buy_order else -order.quantity
      if order.symbol in new_holdings: new_holdings[order.symbol] += q
      else: new_holdings[order.symbol] = q

      if not ignore_risk:
        # Compute before and after at-risk capital.
        at_risk = self.markToMarket(self.holdings) - self.holdings['CASH']
        new_at_risk = self.markToMarket(new_holdings) - new_holdings['CASH']

        if (new_at_risk > at_risk) and (new_at_risk > self.starting_cash):
          log_print("TradingAgent ignored market order due to at-risk constraints: {}\n{}",
                    order, self.fmtHoldings(self.holdings))
          return
      self.orders[order.order_id] = deepcopy(order)
      self.sendMessage(self.exchangeID, Message({"msg" : "MARKET_ORDER", "sender": self.id, "order": order}))
      if self.log_orders: self.logEvent('ORDER_SUBMITTED', order.to_dict())
    else:
      log_print("TradingAgent ignored market order of quantity zero: {}", order)

  def cancelOrder(self, order):
    """Used by any Trading Agent subclass to cancel any order.  The order must currently
    appear in the agent's open orders list."""
    if isinstance(order, LimitOrder):
      self.sendMessage(self.exchangeID, Message({"msg": "CANCEL_ORDER", "sender": self.id,
                                                 "order": order}))
      # Log this activity.
      if self.log_orders: self.logEvent('CANCEL_SUBMITTED', order.to_dict())
    else:
      log_print("order {} of type, {} cannot be cancelled", order, type(order))

  def modifyOrder (self, order, newOrder):
    """ Used by any Trading Agent subclass to modify any existing limit order.  The order must currently
        appear in the agent's open orders list.  Some additional tests might be useful here
        to ensure the old and new orders are the same in some way."""
    self.sendMessage(self.exchangeID, Message({ "msg" : "MODIFY_ORDER", "sender": self.id,
                                                "order" : order, "new_order" : newOrder}))

    # Log this activity.
    if self.log_orders: self.logEvent('MODIFY_ORDER', order.to_dict())


  # Handles ORDER_EXECUTED messages from an exchange agent.  Subclasses may wish to extend,
  # but should still call parent method for basic portfolio/returns tracking.
  def orderExecuted (self, order):
    log_print ("Received notification of execution for: {}", order)

    # Log this activity.
    if self.log_orders: self.logEvent('ORDER_EXECUTED', order.to_dict())

    # At the very least, we must update CASH and holdings at execution time.
    qty = order.quantity if order.is_buy_order else -1 * order.quantity
    sym = order.symbol

    if sym in self.holdings:
      self.holdings[sym] += qty
    else:
      self.holdings[sym] = qty

    if self.holdings[sym] == 0: del self.holdings[sym]

    # As with everything else, CASH holdings are in CENTS.
    self.holdings['CASH'] -= (qty * order.fill_price)
    
    # If this original order is now fully executed, remove it from the open orders list.
    # Otherwise, decrement by the quantity filled just now.  It is _possible_ that due
    # to timing issues, it might not be in the order list (i.e. we issued a cancellation
    # but it was executed first, or something).
    if order.order_id in self.orders:
      o = self.orders[order.order_id]

      if order.quantity >= o.quantity: del self.orders[order.order_id]
      else: o.quantity -= order.quantity

    else:
      log_print ("Execution received for order not in orders list: {}", order)

    log_print ("After execution, agent open orders: {}", self.orders)

    # After execution, log holdings.
    self.logEvent('HOLDINGS_UPDATED', self.holdings)


  # Handles ORDER_ACCEPTED messages from an exchange agent.  Subclasses may wish to extend.
  def orderAccepted (self, order):
    log_print ("Received notification of acceptance for: {}", order)

    # Log this activity.
    if self.log_orders: self.logEvent('ORDER_ACCEPTED', order.to_dict())


    # We may later wish to add a status to the open orders so an agent can tell whether
    # a given order has been accepted or not (instead of needing to override this method).


  # Handles ORDER_CANCELLED messages from an exchange agent.  Subclasses may wish to extend.
  def orderCancelled (self, order):
    log_print ("Received notification of cancellation for: {}", order)

    # Log this activity.
    if self.log_orders: self.logEvent('ORDER_CANCELLED', order.to_dict())

    # Remove the cancelled order from the open orders list.  We may of course wish to have
    # additional logic here later, so agents can easily "look for" cancelled orders.  Of
    # course they can just override this method.
    if order.order_id in self.orders:
      del self.orders[order.order_id]
    else:
      log_print ("Cancellation received for order not in orders list: {}", order)


  # Handles MKT_CLOSED messages from an exchange agent.  Subclasses may wish to extend.
  def marketClosed (self):
    log_print ("Received notification of market closure.")

    # Log this activity.
    self.logEvent('MKT_CLOSED')

    # Remember that this has happened.
    self.mkt_closed = True


  # Handles QUERY_LAST_TRADE messages from an exchange agent.
  def queryLastTrade (self, symbol, price):
    self.last_trade[symbol] = price

    log_print ("Received last trade price of {} for {}.", self.last_trade[symbol], symbol)

    if self.mkt_closed:
      # Note this as the final price of the day.
      self.daily_close_price[symbol] = self.last_trade[symbol]

      log_print ("Received daily close price of {} for {}.", self.last_trade[symbol], symbol)


  # Handles QUERY_SPREAD messages from an exchange agent.
  def querySpread (self, symbol, price, bids, asks, book):
    # The spread message now also includes last price for free.
    self.queryLastTrade(symbol, price)

    self.known_bids[symbol] = bids
    self.known_asks[symbol] = asks

    if bids: best_bid, best_bid_qty = (bids[0][0], bids[0][1])
    else: best_bid, best_bid_qty = ('No bids', 0)

    if asks: best_ask, best_ask_qty = (asks[0][0], asks[0][1])
    else: best_ask, best_ask_qty = ('No asks', 0)

    log_print ("Received spread of {} @ {} / {} @ {} for {}", best_bid_qty, best_bid, best_ask_qty, best_ask, symbol)

    self.logEvent("BID_DEPTH", bids)
    self.logEvent("ASK_DEPTH", asks)
    self.logEvent("IMBALANCE", [sum([x[1] for x in bids]), sum([x[1] for x in asks])])

    self.book = book

  def handleMarketData(self, msg):
    '''
    Handles Market Data messages for agents using subscription mechanism
    '''
    symbol = msg.body['symbol']
    self.known_asks[symbol] = msg.body['asks']
    self.known_bids[symbol] = msg.body['bids']
    self.last_trade[symbol] = msg.body['last_transaction']
    self.exchange_ts[symbol] = msg.body['exchange_ts']


  # Handles QUERY_ORDER_STREAM messages from an exchange agent.
  def queryOrderStream (self, symbol, orders):
    # It is up to the requesting agent to do something with the data, which is a list of dictionaries keyed
    # by order id.  The list index is 0 for orders since the most recent trade, 1 for orders that led up to
    # the most recent trade, and so on.  Agents are not given index 0 (orders more recent than the last
    # trade).
    self.stream_history[self.symbol] = orders

  def query_transacted_volume(self, symbol, transacted_volume):
    """ Handles the QUERY_TRANSACTED_VOLUME messages from the exchange agent"""
    self.transacted_volume[symbol] = transacted_volume

  # Utility functions that perform calculations from available knowledge, but implement no
  # particular strategy.


  # Extract the current known bid and asks. This does NOT request new information.
  def getKnownBidAsk (self, symbol, best=True):
    if best:
      bid = self.known_bids[symbol][0][0] if self.known_bids[symbol] else None
      ask = self.known_asks[symbol][0][0] if self.known_asks[symbol] else None
      bid_vol = self.known_bids[symbol][0][1] if self.known_bids[symbol] else 0
      ask_vol = self.known_asks[symbol][0][1] if self.known_asks[symbol] else 0
      return bid, bid_vol, ask, ask_vol
    else:
      bids = self.known_bids[symbol] if self.known_bids[symbol] else None
      asks = self.known_asks[symbol] if self.known_asks[symbol] else None
      return bids, asks


  # Extract the current bid and ask liquidity within a certain proportion of the
  # inside bid and ask.  (i.e. within=0.01 means to report total BID shares
  # within 1% of the best bid price, and total ASK shares within 1% of the best
  # ask price)
  #
  # Returns bid_liquidity, ask_liquidity.  Note that this is from the order book
  # perspective, not the agent perspective.  (The agent would be selling into the
  # bid liquidity, etc.)
  def getKnownLiquidity (self, symbol, within=0.00):
    bid_liq = self.getBookLiquidity(self.known_bids[symbol], within)
    ask_liq = self.getBookLiquidity(self.known_asks[symbol], within)

    log_print ("Bid/ask liq: {}, {}", bid_liq, ask_liq)
    log_print ("Known bids: {}", self.known_bids[self.symbol])
    log_print ("Known asks: {}", self.known_asks[self.symbol])

    return bid_liq, ask_liq


  # Helper function for the above.  Checks one side of the known order book.
  def getBookLiquidity (self, book, within):
    liq = 0
    for i, (price, shares) in enumerate(book):
      if i == 0:
        best = price

      # Is this price within "within" proportion of the best price?
      if abs(best - price) <= int(round(best * within)):
        log_print ("Within {} of {}: {} with {} shares", within, best, price, shares)
        liq += shares

    return liq


  # Marks holdings to market (including cash).
  def markToMarket (self, holdings, use_midpoint=False):
    cash = holdings['CASH']
    
    cash += self.basket_size * self.nav_diff

    for symbol, shares in holdings.items():
      if symbol == 'CASH': continue

      if use_midpoint:
        bid, ask, midpoint = self.getKnownBidAskMidpoint(symbol)
        if bid is None or ask is None or midpoint is None:
          value = self.last_trade[symbol] * shares
        else:
          value = midpoint * shares
      else:
        value = self.last_trade[symbol] * shares

      cash += value

      self.logEvent('MARK_TO_MARKET', "{} {} @ {} == {}".format(shares, symbol,
                    self.last_trade[symbol], value))

    self.logEvent('MARKED_TO_MARKET', cash)

    return cash


  # Gets holdings.  Returns zero for any symbol not held.
  def getHoldings (self, symbol):
    if symbol in self.holdings: return self.holdings[symbol]
    return 0


  # Get the known best bid, ask, and bid/ask midpoint from cached data.  No volume.
  def getKnownBidAskMidpoint (self, symbol) :
    bid = self.known_bids[symbol][0][0] if self.known_bids[symbol] else None
    ask = self.known_asks[symbol][0][0] if self.known_asks[symbol] else None

    midpoint = int(round((bid + ask) / 2)) if bid is not None and ask is not None else None

    return bid, ask, midpoint

  def get_average_transaction_price(self):
    """ Calculates the average price paid (weighted by the order size) """
    return round(sum(executed_order.quantity * executed_order.fill_price for executed_order in self.executed_orders) / \
                 sum(executed_order.quantity for executed_order in self.executed_orders), 2)

  # Prints holdings.  Standard dictionary->string representation is almost fine, but it is
  # less confusing to see the CASH holdings in dollars and cents, instead of just integer
  # cents.  We could change to a Holdings object that knows to print CASH "special".
  def fmtHoldings (self, holdings):
    h = ''
    for k,v in sorted(holdings.items()):
      if k == 'CASH': continue
      h += "{}: {}, ".format(k,v)

    # There must always be a CASH entry.
    h += "{}: {}".format('CASH', holdings['CASH'])
    h = '{ ' + h + ' }'
    return h


  pass
