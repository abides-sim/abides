from agent.etf.EtfArbAgent import EtfArbAgent
from agent.etf.EtfPrimaryAgent import EtfPrimaryAgent
from message.Message import Message
from util.order.etf.BasketOrder import BasketOrder
from util.order.BasketOrder import BasketOrder
from util.util import log_print

import pandas as pd
import sys

class EtfMarketMakerAgent(EtfArbAgent):

  def __init__(self, id, name, type, portfolio = {}, gamma = 0, starting_cash=100000, lambda_a = 0.005,
                                                    log_orders = False, random_state = None):

    # Base class init.
    super().__init__(id, name, type, portfolio = portfolio, gamma = gamma, starting_cash=starting_cash,
                     lambda_a = lambda_a, log_orders=log_orders, random_state = random_state)
    
    # NEED TO TEST IF THERE ARE SYMBOLS IN PORTFOLIO
    
    # Store important parameters particular to the ETF arbitrage agent.
    self.inPrime = True                          # Determines if the agent also participates in the Primary ETF market
    
    # We don't yet know when the primary opens or closes.
    self.prime_open = None
    self.prime_close = None
    
    # Remember whether we have already passed the primary close time, as far
    # as we know.
    self.prime_closed = False
    self.switched_mkt = False
    
  def kernelStarting(self, startTime):
    # self.kernel is set in Agent.kernelInitializing()
    # self.exchangeID is set in TradingAgent.kernelStarting()
    
    self.primeID = self.kernel.findAgentByType(EtfPrimaryAgent)

    log_print ("Agent {} requested agent of type Agent.EtfPrimaryAgent.  Given Agent ID: {}",
               self.id, self.primeID)

    super().kernelStarting(startTime)
    

  def wakeup (self, currentTime):
    # Parent class handles discovery of exchange times and market_open wakeup call.
    super().wakeup(currentTime)
    
    # Only if the superclass leaves the state as ACTIVE should we proceed with our
    # trading strategy.
    if self.state != 'ACTIVE': return
    
    if not self.prime_open:
      # Ask our primary when it opens and closes, exchange is handled in TradingAgent
      self.sendMessage(self.primeID, Message({ "msg" : "WHEN_PRIME_OPEN", "sender": self.id }))
      self.sendMessage(self.primeID, Message({ "msg" : "WHEN_PRIME_CLOSE", "sender": self.id }))


    # Steady state wakeup behavior starts here.
    if not self.mkt_closed and self.prime_closed:
      print('The prime closed before the exchange')
      sys.exit()
    
    elif self.mkt_closed and self.prime_closed:
      return

    # If we've been told the market has closed for the day, we will only request
    # final price information, then stop.
    # If the market has closed and we haven't obtained the daily close price yet,
    # do that before we cease activity for the day.  Don't do any other behavior
    # after market close.
    elif self.mkt_closed and not self.prime_closed:
      if self.switched_mkt and self.currentTime >= self.prime_open:
        self.getEtfNav()
        self.state = 'AWAITING_NAV'
      elif not self.switched_mkt:
        for i,s in enumerate(self.portfolio):
          if s not in self.daily_close_price:
            self.getLastTrade(s)
            self.state = 'AWAITING_LAST_TRADE'
        if 'ETF' not in self.daily_close_price:
          self.getLastTrade('ETF')
          self.state = 'AWAITING_LAST_TRADE'
            
        print('holdings before primary: ' + str(self.holdings))
      
        self.setWakeup(self.prime_open)
        self.switched_mkt = True
      else:
        self.setWakeup(self.prime_open)
      return

    # Schedule a wakeup for the next time this agent should arrive at the market
    # (following the conclusion of its current activity cycle).
    # We do this early in case some of our expected message responses don't arrive.

    # Agents should arrive according to a Poisson process.  This is equivalent to
    # each agent independently sampling its next arrival time from an exponential
    # distribution in alternate Beta formation with Beta = 1 / lambda, where lambda
    # is the mean arrival rate of the Poisson process.
    else:
      delta_time = self.random_state.exponential(scale = 1.0 / self.lambda_a)
      self.setWakeup(currentTime + pd.Timedelta('{}ns'.format(int(round(delta_time)))))

      # Issue cancel requests for any open orders.  Don't wait for confirmation, as presently
      # the only reason it could fail is that the order already executed.  (But requests won't
      # be generated for those, anyway, unless something strange has happened.)
      self.cancelOrders()


      # The ETF arb agent DOES try to maintain a zero position, so there IS need to exit positions
      # as some "active trading" agents might.  It might exit a position based on its order logic,
      # but this will be as a natural consequence of its beliefs... but it submits marketable orders so...


      # If the calling agent is a subclass, don't initiate the strategy section of wakeup(), as it
      # may want to do something different.
      # FIGURE OUT WHAT TO DO WITH MULTIPLE SPREADS...
      for i,s in enumerate(self.portfolio):
        self.getCurrentSpread(s)
      self.getCurrentSpread('ETF')
      self.state = 'AWAITING_SPREAD'

  def placeOrder(self):
    etf_mid, index_mid, etf_p, index_p, empty_mid  = self.getPriceEstimates()
    if empty_mid:
      #print('no move because index or ETF was missing part of NBBO')
      pass
    elif (index_mid - etf_mid) > self.gamma:
      #print('buy ETF')
      for i,s in enumerate(self.portfolio):
        self.placeLimitOrder(s, 1, False, index_p[s]['bid'])
      self.placeLimitOrder('ETF', 1, True, etf_p['ask'])
    elif (etf_mid - index_mid) > self.gamma:
      #print('sell ETF')
      for i,s in enumerate(self.portfolio):
        self.placeLimitOrder(s, 1, True, index_p[s]['ask'])
      self.placeLimitOrder('ETF', 1, False, etf_p['bid'])
    else:
      pass
      #print('no move because abs(index - ETF mid) < gamma') 
    
  def decideBasket(self):
    print(self.portfolio)
    index_est = 0
    for i,s in enumerate(self.portfolio):
      index_est += self.daily_close_price[s]
    
    H = {}
    for i,s in enumerate(self.portfolio):
      H[s] = self.getHoldings(s)
    etf_h = self.getHoldings('ETF')
    
    self.nav_diff = self.nav - index_est
    if self.nav_diff > 0:
      if min(H.values()) > 0 and etf_h < 0:
        print("send creation basket")
        self.placeBasketOrder(min(H.values()), True)
      else: print('wrong side for basket')
    elif self.nav_diff < 0:
      if etf_h > 0 and max(H.values()) < 0:
        print("submit redemption basket")
        self.placeBasketOrder(etf_h, False)
      else: print('wrong side for basket')
    else:
      if min(H.values()) > 0 and etf_h < 0:
        print("send creation basket")
        self.placeBasketOrder(min(H.values()), True)
      elif etf_h > 0 and max(H.values()) < 0:
        print("submit redemption basket")
        self.placeBasketOrder(etf_h, False)
      
    
  def receiveMessage(self, currentTime, msg):
    # Parent class schedules market open wakeup call once market open/close times are known.
    super().receiveMessage(currentTime, msg)

    # We have been awakened by something other than our scheduled wakeup.
    # If our internal state indicates we were waiting for a particular event,
    # check if we can transition to a new state.
    
    # Record market open or close times.
    if msg.body['msg'] == "WHEN_PRIME_OPEN":
      self.prime_open = msg.body['data']

      log_print ("Recorded primary open: {}", self.kernel.fmtTime(self.prime_open))

    elif msg.body['msg'] == "WHEN_PRIME_CLOSE":
      self.prime_close = msg.body['data']

      log_print ("Recorded primary close: {}", self.kernel.fmtTime(self.prime_close))

    if self.state == 'AWAITING_NAV':
      if msg.body['msg'] == 'QUERY_NAV':
        if msg.body['prime_closed']: self.prime_closed = True
        self.queryEtfNav(msg.body['nav'])

        # But if the market is now closed, don't advance to placing orders.
        if self.prime_closed: return

        # We now have the information needed to place a C/R basket.
        self.decideBasket()
    
    elif self.state == 'AWAITING_BASKET':
      if msg.body['msg'] == 'BASKET_EXECUTED':
        order = msg.body['order']
        # We now have the information needed to place a C/R basket.
        for i,s in enumerate(self.portfolio):
          if order.is_buy_order: self.holdings[s] -= order.quantity
          else: self.holdings[s] += order.quantity
        if order.is_buy_order:
          self.holdings['ETF'] += order.quantity
          self.basket_size = order.quantity
        else:
          self.holdings['ETF'] -= order.quantity
          self.basket_size = -1 * order.quantity
      
        self.state = 'INACTIVE'


  # Internal state and logic specific to this agent subclass.

  # Used by any ETF Arb Agent subclass to query the Net Assest Value (NAV) of the ETF.
  # This activity is not logged.
  def getEtfNav (self):
    self.sendMessage(self.primeID, Message({ "msg" : "QUERY_NAV", "sender": self.id })) 
    
  # Used by ETF Arb Agent subclass to place a basket order.
  # This activity is not logged.
  def placeBasketOrder (self, quantity, is_create_order):
    order = BasketOrder(self.id, self.currentTime, 'ETF', quantity, is_create_order)
    print('BASKET ORDER PLACED: ' + str(order))
    self.sendMessage(self.primeID, Message({ "msg" : "BASKET_ORDER", "sender": self.id,
                                                  "order" : order })) 
    self.state = 'AWAITING_BASKET'
  
  # Handles QUERY NAV messages from primary
  def queryEtfNav(self, nav):
    self.nav = nav
    log_print ("Received NAV of ETF.")