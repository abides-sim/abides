# The ExchangeAgent expects a numeric agent id, printable name, agent type, timestamp to open and close trading,
# a list of equity symbols for which it should create order books, a frequency at which to archive snapshots
# of its order books, a pipeline delay (in ns) for order activity, the exchange computation delay (in ns),
# the levels of order stream history to maintain per symbol (maintains all orders that led to the last N trades),
# whether to log all order activity to the agent log, and a random state object (already seeded) to use
# for stochasticity.
from agent.FinancialAgent import FinancialAgent
from agent.ExchangeAgent import ExchangeAgent
from message.Message import Message
from util.util import log_print

import pandas as pd
pd.set_option('display.max_rows', 500)


class EtfPrimaryAgent(FinancialAgent):

  def __init__(self, id, name, type, prime_open, prime_close, symbol, pipeline_delay = 40000,
               computation_delay = 1, random_state = None):

    super().__init__(id, name, type, random_state)

    # Do not request repeated wakeup calls.
    self.reschedule = False

    # Store this exchange's open and close times.
    self.prime_open = prime_open
    self.prime_close = prime_close
    
    self.mkt_close = None
    
    self.nav = 0
    self.create = 0
    self.redeem = 0
    
    self.symbol = symbol

    # Right now, only the exchange agent has a parallel processing pipeline delay.  This is an additional
    # delay added only to order activity (placing orders, etc) and not simple inquiries (market operating
    # hours, etc).
    self.pipeline_delay = pipeline_delay

    # Computation delay is applied on every wakeup call or message received.
    self.computation_delay = computation_delay
   
  def kernelStarting(self, startTime):
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

    print ("Final C/R baskets for {}: {} creation baskets. {} redemption baskets".format(self.name,
                                                                                         self.create, self.redeem))


  # Simulation participation messages.

  def wakeup (self, currentTime):
    super().wakeup(currentTime)

    if self.mkt_close is None:
      # Ask our exchange when it opens and closes.
      self.sendMessage(self.exchangeID, Message({ "msg" : "WHEN_MKT_CLOSE", "sender": self.id }))
        
    else:
      # Get close price of ETF/nav
      self.getLastTrade(self.symbol)

  def receiveMessage (self, currentTime, msg):
    super().receiveMessage(currentTime, msg)

    # Unless the intent of an experiment is to examine computational issues within an Exchange,
    # it will typically have either 1 ns delay (near instant but cannot process multiple orders
    # in the same atomic time unit) or 0 ns delay (can process any number of orders, always in
    # the atomic time unit in which they are received).  This is separate from, and additional
    # to, any parallel pipeline delay imposed for order book activity.

    # Note that computation delay MUST be updated before any calls to sendMessage.
    self.setComputationDelay(self.computation_delay)

    # Is the exchange closed?  (This block only affects post-close, not pre-open.)
    if currentTime > self.prime_close:
      # Most messages after close will receive a 'PRIME_CLOSED' message in response.
      log_print ("{} received {}, discarded: prime is closed.", self.name, msg.body['msg'])
      self.sendMessage(msg.body['sender'], Message({ "msg": "PRIME_CLOSED" }))
      # Don't do any further processing on these messages!
      return

    
    if msg.body['msg'] == "WHEN_MKT_CLOSE":
      self.mkt_close = msg.body['data']
      log_print ("Recorded market close: {}", self.kernel.fmtTime(self.mkt_close))
      self.setWakeup(self.mkt_close)
      return
    
    elif msg.body['msg'] == 'QUERY_LAST_TRADE':
      # Call the queryLastTrade method.
      self.queryLastTrade(msg.body['symbol'], msg.body['data'])
      return

    self.logEvent(msg.body['msg'], msg.body['sender'])

    # Handle all message types understood by this exchange.
    if msg.body['msg'] == "WHEN_PRIME_OPEN":
      log_print ("{} received WHEN_PRIME_OPEN request from agent {}", self.name, msg.body['sender'])

      # The exchange is permitted to respond to requests for simple immutable data (like "what are your
      # hours?") instantly.  This does NOT include anything that queries mutable data, like equity
      # quotes or trades.
      self.setComputationDelay(0)

      self.sendMessage(msg.body['sender'], Message({ "msg": "WHEN_PRIME_OPEN", "data": self.prime_open }))
        
    elif msg.body['msg'] == "WHEN_PRIME_CLOSE":
      log_print ("{} received WHEN_PRIME_CLOSE request from agent {}", self.name, msg.body['sender'])

      # The exchange is permitted to respond to requests for simple immutable data (like "what are your
      # hours?") instantly.  This does NOT include anything that queries mutable data, like equity
      # quotes or trades.
      self.setComputationDelay(0)

      self.sendMessage(msg.body['sender'], Message({ "msg": "WHEN_PRIME_CLOSE", "data": self.prime_close }))
        
    elif msg.body['msg'] == "QUERY_NAV":
      log_print ("{} received QUERY_NAV ({}) request from agent {}", self.name, msg.body['sender'])

      # Return the NAV for the requested symbol.
      self.sendMessage(msg.body['sender'], Message({ "msg": "QUERY_NAV",
           "nav": self.nav, "prime_closed": True if currentTime > self.prime_close else False }))
        
    elif msg.body['msg'] == "BASKET_ORDER":
      order = msg.body['order']
      log_print ("{} received BASKET_ORDER: {}", self.name, order)
      if order.is_buy_order: self.create += 1
      else: self.redeem += 1
      order.fill_price = self.nav
      self.sendMessage(msg.body['sender'], Message({ "msg": "BASKET_EXECUTED", "order": order}))
  

  # Handles QUERY_LAST_TRADE messages from an exchange agent.
  def queryLastTrade (self, symbol, price):
    self.nav = price
    log_print ("Received daily close price or nav of {} for {}.", price, symbol)
                       
  # Used by any Trading Agent subclass to query the last trade price for a symbol.
  # This activity is not logged.
  def getLastTrade (self, symbol):
    self.sendMessage(self.exchangeID, Message({ "msg" : "QUERY_LAST_TRADE", "sender": self.id,
                                                "symbol" : symbol })) 

  # Simple accessor methods for the market open and close times.
  def getPrimeOpen(self):
    return self.__prime_open

  def getPrimeClose(self):
    return self.__prime_close