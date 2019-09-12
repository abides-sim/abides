from agent.TradingAgent import TradingAgent
from util.util import log_print

import numpy as np
import pandas as pd


class EtfArbAgent(TradingAgent):

  def __init__(self, id, name, type, portfolio = {}, gamma = 0, starting_cash=100000, lambda_a = 0.005,
                                                    log_orders = False, random_state = None):

    # Base class init.
    super().__init__(id, name, type, starting_cash=starting_cash, log_orders=log_orders, random_state = random_state)

    # Store important parameters particular to the ETF arbitrage agent.
    self.inPrime = False                         # Determines if the agent also participates in the Primary ETF market
    self.portfolio = portfolio                   # ETF portfolio
    self.gamma = gamma                           # Threshold for difference between ETF and index to trade
    self.messageCount = len(self.portfolio) + 1  # Tracks the number of messages sent so all limit orders are sent after
                                                 # are sent after all mid prices are calculated
    #self.q_max = q_max                          # max unit holdings
    self.lambda_a = lambda_a                     # mean arrival rate of ETF arb agents (eventually change to a subscription)
    
    # NEED TO TEST IF THERE ARE SYMBOLS IN PORTFOLIO

    # The agent uses this to track whether it has begun its strategy or is still
    # handling pre-market tasks.
    self.trading = False

    # The agent begins in its "complete" state, not waiting for
    # any special event or condition.
    self.state = 'AWAITING_WAKEUP'
    
  def kernelStarting(self, startTime):
    # self.kernel is set in Agent.kernelInitializing()
    # self.exchangeID is set in TradingAgent.kernelStarting()

    super().kernelStarting(startTime)

    self.oracle = self.kernel.oracle


  def kernelStopping (self):
    # Always call parent method to be safe.
    super().kernelStopping()

    # Print end of day valuation.
    H = {}
    H['ETF'] = self.getHoldings('ETF')
    for i,s in enumerate(self.portfolio):
      H[s] = self.getHoldings(s)
    print(H)
    print(self.daily_close_price)


  def wakeup (self, currentTime):
    # Parent class handles discovery of exchange times and market_open wakeup call.
    super().wakeup(currentTime)

    self.state = 'INACTIVE'
    
    if not self.mkt_open or not self.mkt_close:
      return
    else:
      if not self.trading:
        self.trading = True
        # Time to start trading!
        log_print ("{} is ready to start trading now.", self.name)


    # Steady state wakeup behavior starts here.
    # If we've been told the market has closed for the day, we will only request
    # final price information, then stop.
    # If the market has closed and we haven't obtained the daily close price yet,
    # do that before we cease activity for the day.  Don't do any other behavior
    # after market close.
    # If the calling agent is a subclass, don't initiate the strategy section of wakeup(), as it
    # may want to do something different.
    if self.mkt_closed and not self.inPrime:
      for i,s in enumerate(self.portfolio):
        if s not in self.daily_close_price:
          self.getLastTrade(s)
          self.state = 'AWAITING_LAST_TRADE'
      if 'ETF' not in self.daily_close_price:
        self.getLastTrade('ETF')
        self.state = 'AWAITING_LAST_TRADE'
      return

    # Schedule a wakeup for the next time this agent should arrive at the market
    # (following the conclusion of its current activity cycle).
    # We do this early in case some of our expected message responses don't arrive.

    # Agents should arrive according to a Poisson process.  This is equivalent to
    # each agent independently sampling its next arrival time from an exponential
    # distribution in alternate Beta formation with Beta = 1 / lambda, where lambda
    # is the mean arrival rate of the Poisson process.
    elif not self.inPrime:
      delta_time = self.random_state.exponential(scale = 1.0 / self.lambda_a)
      self.setWakeup(currentTime + pd.Timedelta('{}ns'.format(int(round(delta_time)))))

      # Issue cancel requests for any open orders.  Don't wait for confirmation, as presently
      # the only reason it could fail is that the order already executed.  (But requests won't
      # be generated for those, anyway, unless something strange has happened.)
      self.cancelOrders()


      # The ETF arb agent DOES try to maintain a zero position, so there IS need to exit positions
      # as some "active trading" agents might.  It might exit a position based on its order logic,
      # but this will be as a natural consequence of its beliefs... but it submits marketable orders so...
      for i,s in enumerate(self.portfolio):
        self.getCurrentSpread(s)
      self.getCurrentSpread('ETF')
      self.state = 'AWAITING_SPREAD'

    else:
      self.state = 'ACTIVE'
  
  def getPriceEstimates(self):
    index_mids = np.empty(len(self.portfolio))
    index_p = {}
    empty_mid = False
    for i,s in enumerate(self.portfolio):
      bid, bid_vol, ask, ask_vol = self.getKnownBidAsk(s)
      if bid != None and ask != None:
        index_p[s] = {'bid': bid, 'ask': ask}
        mid = 0.5 * (int(bid) + int(ask))
      else:
        mid = float()
        index_p[s] = {'bid': float(), 'ask': float()}
        empty_mid = True
      index_mids[i] = mid
    bid, bid_vol, ask, ask_vol = self.getKnownBidAsk('ETF')
    etf_p = {'bid': bid, 'ask': ask}
    if bid != None and ask != None:
      etf_mid = 0.5 * (int(bid) + int(ask))
    else:
      etf_mid = float()
      empty_mid = True
    index_mid = np.sum(index_mids)
    return etf_mid, index_mid, etf_p, index_p, empty_mid

  def placeOrder(self):
    etf_mid, index_mid, etf_p, index_p, empty_mid  = self.getPriceEstimates()
    if empty_mid:
      #print('no move because index or ETF was missing part of NBBO')
      pass
    elif (index_mid - etf_mid) > self.gamma:
      self.placeLimitOrder('ETF', 1, True, etf_p['ask'])
    elif (etf_mid - index_mid) > self.gamma:
      self.placeLimitOrder('ETF', 1, False, etf_p['bid'])
    else:
      pass
      #print('no move because abs(index - ETF mid) < gamma')   
    
  def receiveMessage(self, currentTime, msg):
    # Parent class schedules market open wakeup call once market open/close times are known.
    super().receiveMessage(currentTime, msg)

    # We have been awakened by something other than our scheduled wakeup.
    # If our internal state indicates we were waiting for a particular event,
    # check if we can transition to a new state.
    
    if self.state == 'AWAITING_SPREAD':
      # We were waiting to receive the current spread/book.  Since we don't currently
      # track timestamps on retained information, we rely on actually seeing a
      # QUERY_SPREAD response message.

      if msg.body['msg'] == 'QUERY_SPREAD':
        # This is what we were waiting for.
        self.messageCount -= 1

        # But if the market is now closed, don't advance to placing orders.
        if self.mkt_closed: return

        # We now have the information needed to place a limit order with the eta
        # strategic threshold parameter.
        if self.messageCount == 0:
          self.placeOrder()
          self.messageCount = len(self.portfolio) + 1
          self.state = 'AWAITING_WAKEUP'


  # Internal state and logic specific to this agent subclass.

  # Cancel all open orders.
  # Return value: did we issue any cancellation requests?
  def cancelOrders (self):
    if not self.orders: return False

    for id, order in self.orders.items():
      self.cancelOrder(order)

    return True
    
  def getWakeFrequency (self):
    return pd.Timedelta(self.random_state.randint(low = 0, high = 100), unit='ns')