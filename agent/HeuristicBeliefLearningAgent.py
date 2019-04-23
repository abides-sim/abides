from agent.ZeroIntelligenceAgent import ZeroIntelligenceAgent
from message.Message import Message
from util.util import print

from math import sqrt
import numpy as np
import pandas as pd
import sys

class HeuristicBeliefLearningAgent(ZeroIntelligenceAgent):

  def __init__(self, id, name, symbol, startingCash=100000, sigma_n=1000, 
                     r_bar=100000, kappa=0.05, sigma_s=100000, q_max=10,
                     sigma_pv=5000000, R_min = 0, R_max = 250, eta = 1.0,
                     lambda_a = 0.005, L = 8):

    # Base class init.
    super().__init__(id, name, symbol, startingCash=startingCash, sigma_n=sigma_n,
                     r_bar=r_bar, kappa=kappa, sigma_s=sigma_s, q_max=q_max,
                     sigma_pv=sigma_pv, R_min=R_min, R_max=R_max, eta=eta,
                     lambda_a = lambda_a)

    # Store important parameters particular to the HBL agent.
    self.L = L


  def wakeup (self, currentTime):
    # Parent class handles discovery of exchange times and market_open wakeup call.
    # Also handles SRG-type "background agent" needs that are not specific to HBL.
    super().wakeup(currentTime)

    # Only if the superclass leaves the state as ACTIVE should we proceed with our
    # trading strategy.
    if self.state != 'ACTIVE': return

    # To make trade decisions, the HBL agent requires recent order stream information.
    self.getOrderStream(self.symbol, length=self.L)
    self.state = 'AWAITING_STREAM'



  def placeOrder (self):
    # Called when it is time for the agent to determine a limit price and place an order.
    # This method implements the HBL strategy and falls back to the ZI (superclass)
    # strategy if there is not enough information for the HBL strategy.

    # See if there is enough history for HBL.  If not, we will _exactly_ perform the
    # ZI placeOrder().  If so, we will use parts of ZI but compute our limit price
    # differently.  Note that we are not given orders more recent than the most recent
    # trade.

    if len(self.stream_history[self.symbol]) < self.L:
      # Not enough history for HBL.
      print ("Insufficient history for HBL: length {}, L {}".format(len(self.stream_history[self.symbol]), self.L))
      super().placeOrder()
      return


    # There is enough history for HBL.

    # Use the superclass (ZI) method to obtain an observation, update internal estimate
    # parameters, decide to buy or sell, and calculate the total unit valuation, because
    # all of this logic is identical to ZI.
    v, buy = self.updateEstimates()


    # Walk through the visible order history and accumulate values needed for HBL's
    # estimation of successful transaction by limit price.
    sa = {}
    sb = {}
    ua = {}
    ub = {}

    low_p = sys.maxsize
    high_p = 0

    for h in self.stream_history[self.symbol]:
      # h follows increasing "transactions into the past", with index zero being orders
      # after the most recent transaction.
      for id, order in h.items():
        p = order['limit_price']
        if p < low_p: low_p = p
        if p > high_p: high_p = p

        # For now if there are any transactions, consider the order successful.  For single
        # unit orders as used in SRG configs, this is sufficient.  For multi-unit orders,
        # we may wish to switch to a proportion of shares executed.
        if order['is_buy_order']:
          if order['transactions']: sb[p] = 1 if not p in sb else sb[p] + 1
          else:
            ub[p] = 1 if not p in ub else ub[p] + 1
        else:
          if order['transactions']: sa[p] = 1 if not p in sa else sa[p] + 1
          else:
            ua[p] = 1 if not p in ua else ua[p] + 1


    # For each limit price between the lowest and highest observed price in history,
    # compute the estimated probability of a successful transaction.  Remember the
    # price that produces the greatest expected surplus.
    best_p = None
    best_Pr = None
    best_Es = -sys.maxsize

    for p in range(low_p, high_p+1):
      if buy:
        o = sum( [sa[x] for x in sa if x <= p] + [ua[x] for x in ua if x <= p] )
        s = sum( [sb[x] for x in sb if x <= p] )
        u = sum( [ub[x] for x in ub if x >= p] )
      else:
        o = sum( [sb[x] for x in sb if x >= p] + [ub[x] for x in ub if x >= p] )
        s = sum( [sa[x] for x in sa if x >= p] )
        u = sum( [ua[x] for x in ua if x <= p] )

      #print ("p {}, o {}, s {}, u {}".format(p, o, s, u))

      if o + s + u <= 0: Pr = 0
      else: Pr = (o + s) / (o + s + u)

      Es = Pr * (v - p) if buy else Pr * (p - v)

      if Es > best_Es:
        best_Es = Es
        best_Pr = Pr
        best_p = p

    # best_p should now contain the limit price that produces maximum expected surplus best_Es
    if best_Es > 0:
      print ("{} selects limit price {} with expected surplus {} (Pr = {:0.4f})".format(self.name, best_p, int(round(best_Es)), best_Pr))

      # Place the constructed order.
      self.placeLimitOrder(self.symbol, 1, buy, best_p)
    else:
      print ("{} elects not to place an order (best expected surplus <= 0)".format(self.name))



  def receiveMessage (self, currentTime, msg):

    # We have been awakened by something other than our scheduled wakeup.
    # If our internal state indicates we were waiting for a particular event,
    # check if we can transition to a new state.

    # Allow parent class to handle state + message combinations it understands.
    super().receiveMessage(currentTime, msg)

    # Do our special stuff.
    if self.state == 'AWAITING_STREAM':
      # We were waiting to receive the recent order stream.
      if msg.body['msg'] == 'QUERY_ORDER_STREAM':
        # This is what we were waiting for.

        # But if the market is now closed, don't advance.
        if self.mkt_closed: return

        self.getCurrentSpread(self.symbol)
        self.state = 'AWAITING_SPREAD'


