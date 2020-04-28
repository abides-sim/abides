from agent.TradingAgent import TradingAgent
from message.Message import Message
from util.util import log_print

import numpy as np
import pandas as pd
import sys

class QLearningAgent(TradingAgent):

  def __init__(self, id, name, type, symbol='IBM', starting_cash=100000,
                     qtable = None, log_orders = False, random_state = None):

    # Base class init.
    super().__init__(id, name, type, starting_cash=starting_cash, log_orders=log_orders, random_state = random_state)

    # Store important parameters particular to the QLearning agent.
    self.symbol = symbol
    self.qtable = qtable

    # The agent uses this to track whether it has begun its strategy or is still
    # handling pre-market tasks.
    self.trading = False

    # The agent begins in its "complete" state, not waiting for
    # any special event or condition.
    self.state = 'AWAITING_WAKEUP'

    # The agent tracks an experience history to sample and learn from over time.
    # Tuples of (s,a,s',r).  This is not currently used (nor part of the saved state).
    self.experience = []

    # Remember prior state, action, and portfolio value (marked to market).
    self.s = None
    self.a = None
    self.v = None



  # During kernelStopping, give the Kernel our saved state for potential
  # subsequent simulations during the same experiment.
  def kernelStopping (self):
    super().kernelStopping()
    self.updateAgentState(self.qtable)


  def wakeup (self, currentTime):
    # Parent class handles discovery of exchange times and market_open wakeup call.
    super().wakeup(currentTime)

    self.state = 'INACTIVE'

    if not self.mkt_open or not self.mkt_close:
      # TradingAgent handles discovery of exchange times.
      return
    else:
      if not self.trading:
        self.trading = True

        # Time to start trading!
        log_print ("{} is ready to start trading now.", self.name)


    # Steady state wakeup behavior starts here.

    # If we've been told the market has closed for the day, we will only request
    # final price information, then stop.
    if self.mkt_closed and (self.symbol in self.daily_close_price):
      # Market is closed and we already got the daily close price.
      return


    # Schedule a wakeup for the next time this agent should arrive at the market
    # (following the conclusion of its current activity cycle).
    # We do this early in case some of our expected message responses don't arrive.

    # The QLearning agent is not a background agent, so it should select strategically
    # appropriate times.  (Maybe we should LEARN the best times or frequencies at which
    # to trade, someday.)  Presently it just trades once a minute.
    self.setWakeup(currentTime + pd.Timedelta('1min'))
 
    # If the market has closed and we haven't obtained the daily close price yet,
    # do that before we cease activity for the day.  Don't do any other behavior
    # after market close.
    if self.mkt_closed and (not self.symbol in self.daily_close_price):
      self.getCurrentSpread(self.symbol)
      self.state = 'AWAITING_SPREAD'
      return

    # Cancel unfilled orders (but don't exit positions).
    self.cancelOrders()

    # Get the order book or whatever else we need for the state.
    self.getCurrentSpread(self.symbol, depth=1000)
    self.state = 'AWAITING_SPREAD'



  def placeOrder (self):
    # Called when it is time for the agent to determine a limit price and place an order.

    # Compute the order imbalance feature.
    bid_vol = sum([ v[1] for v in self.known_bids[self.symbol] ])
    ask_vol = sum([ v[1] for v in self.known_asks[self.symbol] ])
    imba = bid_vol - ask_vol

    # A unit of stock is now 100 shares instead of one.
    imba = int(imba / 100)

    # Get our current holdings in the stock of interest.
    h = self.getHoldings(self.symbol)
    
    # The new state will be called s_prime.  This agent simply uses current
    # holdings (limit: one share long or short) and offer volume imbalance.
    # State: 1000s digit is 0 (short), 1 (neutral), 2 (long).  Remaining digits
    #        are 000 (-100 imba) to 200 (+100 imba).
    s_prime = ((h + 1) * 1000) + (imba + 100)

    log_print ("h: {}, imba: {}, s_prime: {}", h, imba, s_prime)

    # Compute our reward from last time.  We estimate the change in the value
    # of our portfolio by marking it to market and comparing against the last
    # time we were contemplating an action.
    v = self.markToMarket(self.holdings, use_midpoint=True)
    r = v - self.v if self.v is not None else 0

    # Store our experience tuple.
    self.experience.append((self.s, self.a, s_prime, r))

    # Update our q table.
    old_q = self.qtable.q[self.s, self.a]
    old_weighted = (1 - self.qtable.alpha) * old_q

    a_prime = np.argmax(self.qtable.q[s_prime,:])
    new_q = r + (self.qtable.gamma * self.qtable.q[s_prime, a_prime])
    new_weighted = self.qtable.alpha * new_q

    self.qtable.q[self.s, self.a] = old_weighted + new_weighted


    # Decay alpha.
    self.qtable.alpha *= self.qtable.alpha_decay
    self.qtable.alpha = max(self.qtable.alpha, self.qtable.alpha_min)


    # Compute our next action.  0 = sell one, 1 == do nothing, 2 == buy one.
    if self.random_state.rand() < self.qtable.epsilon:
      # Random action, and decay epsilon.
      a = self.random_state.randint(0,3)
      self.qtable.epsilon *= self.qtable.epsilon_decay
      self.qtable.epsilon = max(self.qtable.epsilon, self.qtable.epsilon_min)
    else:
      # Expected best action.
      a = a_prime

    # Respect holding limit.
    if a == 0 and h == -1:  a = 1
    elif a == 2 and h == 1: a = 1


    # Remember s, a, and v for next time.
    self.s = s_prime
    self.a = a
    self.v = v


    # Place the order.  We probably want this to be a market order, once supported,
    # or use a "compute required price for guaranteed execution" function like the
    # impact agent, but that requires fetching quite a bit of book depth.
    if a == 0:   self.placeLimitOrder(self.symbol, 1, False, 50000)
    elif a == 2: self.placeLimitOrder(self.symbol, 1, True, 200000)


  def receiveMessage (self, currentTime, msg):
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

        # But if the market is now closed, don't advance to placing orders.
        if self.mkt_closed: return

        # We now have the information needed to place a limit order with the eta
        # strategic threshold parameter.
        self.placeOrder()
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

