from agent.TradingAgent import TradingAgent
from message.Message import Message
from util.util import print

import numpy as np
import pandas as pd
import sys

class BackgroundAgent(TradingAgent):

  def __init__(self, id, name, symbol, startingCash, sigma_n, arb_last_trade, freq, trade_vol, offset_unit):
    # Base class init.
    super().__init__(id, name, startingCash)

    self.sigma_n = sigma_n

    self.symbol = symbol
    self.trading = False

    self.LOW_CUSHION  = 0.0015
    self.HIGH_CUSHION = 0.0025

    self.TRADE_THRESHOLD = 5
    #self.LIMIT_STD_CENTS = 0.001
    self.LIMIT_STD_CENTS = 0.01

    # Used by this agent to control how long to safely wait
    # for orders to have reached the exchange before proceeding (ns).
    self.message_delay = 1000000000

    # The agent begins in its "complete" state, not waiting for
    # any special event or condition.
    self.state = 'AWAITING_WAKEUP'

    # To provide some consistency, the agent maintains knowledge of its prior value belief.
    self.value_belief = None
    #self.learning_rate = 0.001
    #self.learning_rate = 0.50
    self.learning_rate = 1.0

    # This (for now) constant controls whether the agent arbs to the last trade or to the
    # bid-ask midpoint.
    self.ARB_LAST_TRADE = arb_last_trade

    # This controls the wakeup frequency of the trader.
    self.freq = freq

    # This should be the average trade volume of this trader.
    self.trade_vol = trade_vol

    # The unit of measurement for the -100 to +100 offset of wakeup time.
    self.offset_unit = offset_unit


  def kernelStarting(self, startTime):
    # self.kernel is set in Agent.kernelInitializing()
    # self.exchangeID is set in TradingAgent.kernelStarting()

    super().kernelStarting(startTime)

    self.oracle = self.kernel.oracle


  def wakeup (self, currentTime):
    # Parent class handles discovery of exchange times and market_open wakeup call.
    super().wakeup(currentTime)

    if not self.mkt_open or not self.mkt_close:
      # TradingAgent handles discovery of exchange times.
      return
    else:
      if not self.trading:
        self.trading = True

        # Time to start trading!
        print ("{} is ready to start trading now.".format(self.name))


    # Steady state wakeup behavior starts here.

    # First, see if we have received a MKT_CLOSED message for the day.  If so,
    # there's nothing to do except clean-up.  In the future, we could also do
    # some activity that is not order based (because the exchange and simulation
    # will still be running, just not accepting orders) like final price quotes
    # or final trade information.
    if self.mkt_closed and (self.symbol in self.daily_close_price):
      # Market is closed and we already got the daily close price.
        return


    # Next, schedule a wakeup for about five minutes, plus or minus ten seconds.
    # We do this early in case some of our expected message responses don't arrive.

    offset = np.random.randint(-100,100)
    self.setWakeup(currentTime + (pd.Timedelta(self.freq) + pd.Timedelta('{}{}'.format(offset, self.offset_unit))))

    # If the market is closed and we haven't obtained the daily close price yet,
    # do that before we cease activity for the day.  Don't do any other behavior
    # after market close.
    if self.mkt_closed and (not self.symbol in self.daily_close_price):
      self.getCurrentSpread(self.symbol)
      self.state = 'AWAITING_SPREAD'
      return


    # The agent's behavior has changed to cancel orders, wait for confirmation,
    # exit all positions, wait for confirmation, then enter new positions.
    # It does yield (return) in between these, so it can react to events that
    # occur in between.  This adds a few messages, but greatly improves the logical
    # flow of the simulation and solves several important problems.

    # On a true "wakeup", the agent is at one of its scheduled intervals.
    # We should first check for open orders we would like to cancel.
    # There should be no harm in issuing all the cancel orders simultaneously.

    if self.cancelOrders():
      self.state = 'AWAITING_CANCEL_CONFIRMATION'
      return

    # If we needed to cancel orders, the logic below will not execute, because
    # our ORDER_CANCELLED messages will come through receiveMessage().  If we
    # did not need to, we may as well carry on to exiting our positions.

    if self.exitPositions():
      self.state = 'AWAITING_EXIT_CONFIRMATION'
      return

    # The below logic is only reached if we neither needed to cancel orders
    # nor exit positions, in which case we may as well find out the most recent
    # trade prices and get ready to place new orders.

    self.getCurrentSpread(self.symbol)
    self.state = 'AWAITING_SPREAD'


  def receiveMessage (self, currentTime, msg):
    # Parent class schedules market open wakeup call once market open/close times are known.
    super().receiveMessage(currentTime, msg)

    # We have been awakened by something other than our scheduled wakeup.
    # If our internal state indicates we were waiting for a particular event,
    # check if we can transition to a new state.

    if self.state == 'AWAITING_CANCEL_CONFIRMATION':
      # We were waiting for all open orders to be cancelled.  See if that has happened.
      if not self.orders:
        # Ready to exit positions.
        if self.exitPositions():
          self.state = 'AWAITING_EXIT_CONFIRMATION'
          return

        # If we did not need to exit positions, go ahead and query the most recent trade.
        self.getCurrentSpread(self.symbol)
        self.state = 'AWAITING_SPREAD'

    elif self.state == 'AWAITING_EXIT_CONFIRMATION':
      # We were waiting for all open positions to be exited.  See if that has happened.
      if not self.havePositions():
        # Query the most recent trade and prepare to proceed.
        self.getCurrentSpread(self.symbol)
        self.state = 'AWAITING_SPREAD'

    elif self.state == 'AWAITING_SPREAD':
      # We were waiting to learn the most recent trade price of our symbol, so we would
      # know what direction of order we'd like to place.

      # Right now we can't tell from internal state whether the last_trade has been
      # updated recently, so we rely on actually seeing a last_trade response message.
      if msg.body['msg'] == 'QUERY_SPREAD':
        # This is what we were waiting for.

        # But if the market is now closed, don't advance to placing orders.
        if self.mkt_closed: return

        # Now we can obtain a new price belief and place our next set of orders.
        self.placeOrders()
        self.state = 'AWAITING_WAKEUP'


  # Internal state and logic specific to this Background (Oracle) Agent.

  # Cancel all open orders.
  # Return value: did we issue any cancellation requests?
  def cancelOrders (self):
    if not self.orders: return False

    for id, order in self.orders.items():
      self.cancelOrder(order)

    return True


  # Exit all open positions.
  # Return value: did we issue any orders to exit positions?
  def exitPositions (self):
    if not self.havePositions(): return False

    for sym, qty in self.holdings.items():
      if sym == 'CASH': continue

      # Place an exit order for this position.  Instead of (pseudo-) market
      # orders, we now place a limit likely to immediately execute.
      #last_trade = self.last_trade[self.symbol]

      #offset = int(round(np.random.uniform(low=self.LOW_CUSHION, high=self.HIGH_CUSHION) * last_trade))
      bid, bid_vol, ask, ask_vol = self.getKnownBidAsk(self.symbol)

      if not bid or not ask:
        # No spread yet.  Use last trade (or open price) for bid and ask.
        arb_target = self.last_trade[self.symbol]
        bid = arb_target
        ask = arb_target

      # Don't force exit with market orders.  Just sell to the bid or buy from the ask.
      # This should keep the order book from collapsing.
      if qty > 0: self.placeLimitOrder(sym, qty, False, bid)
      elif qty < 0: self.placeLimitOrder(sym, -qty, True, ask)
      else: del self.holdings[sym]

    return True


  # Do we have non-CASH positions?
  def havePositions (self):
    return len(self.holdings) > 1 or \
           (len(self.holdings) == 1 and 'CASH' not in self.holdings)


  # Request the last trade price for our symbol.
  def getLastTrade (self):
    super().getLastTrade(self.symbol)

    
  # Obtain new beliefs and place new orders for position entry.
  def placeOrders (self):

    # The background agents use the DataOracle to obtain noisy observations of the
    # actual historical intraday price on a particular date.  They use this to
    # produce a realistic "background" market of agents who trade based on a belief
    # that follows history (i.e. beliefs do not change based on other agent trading
    # activity) but whose behavior does react to market conditions -- because they
    # will try to arbitrage between their beliefs and the current market state.

    # Get current value belief for relevant stock (observation is noisy).  Beliefs
    # can change even when (unknown) real historical stock price has not changed.
    # sigma_n is the variance of gaussian observation noise as a proportion of the
    # current stock price.  (e.g. if stock trades at 100, sigma_n=0.01 will
    # select from a normal(mean=100,std=1) distribution.
    value_observation = self.oracle.observePrice(self.symbol, self.currentTime, sigma_n=self.sigma_n)

    # TESTING: value_belief is only allowed to change at a certain rate from the prior
    # belief, to keep some kind of consistency and make "beliefs" mean something.

    if self.value_belief: self.logEvent("OLD_BELIEF", self.value_belief)
    self.logEvent("BELIEF_OBSERVATION", value_observation)

    # If there was a prior belief, update it.
    if self.value_belief:
      delta = value_observation - self.value_belief
      print ("observation {}, old belief {}, delta {}".format(value_observation, self.value_belief, delta))
      self.value_belief = int(round(self.value_belief + (delta * self.learning_rate)))
    else:
      # Otherwise use the observation as the belief.
      self.value_belief = value_observation

    print ("New belief {}".format(self.value_belief))
    self.logEvent("NEW_BELIEF", self.value_belief)
    
    if self.ARB_LAST_TRADE:
      arb_target = self.last_trade[self.symbol]
    else:
      bid, bid_vol, ask, ask_vol = self.getKnownBidAsk(self.symbol)

      if bid and ask:
        arb_target = int(round((bid + ask) / 2))
      else:
        # No spread yet.  Use last trade (or open price) for arb_target, bid, and ask.
        arb_target = self.last_trade[self.symbol]
        bid = arb_target
        ask = arb_target

    print ("{} believes {} is worth {} at {}, arb target: {}.".format(self.name, self.symbol, 
           self.dollarize(self.value_belief), self.kernel.fmtTime(self.currentTime),
           self.dollarize(arb_target)))

    # The agents now have their desired behavior.  Instead of placing bracketing limit orders, they
    # arbitrage between the last trade price and their value belief.  This means one-sided orders.
    # Note that value_belief, like all prices, is in integer CENTS.
    #
    # The agent places limit orders designed to immediately execute.  They will pay slightly more (buy)
    # than the last trade price, or accept slightly less than the last current trade price (sell).
    # Offset must be adjusted to round cents.
    offset = int(round(np.random.uniform(low=self.LOW_CUSHION, high=self.HIGH_CUSHION) * arb_target))
    #shares = np.random.randint(100,400)
    if self.trade_vol < 200:
      print ("ERROR: BackgroundAgents don't work right with less than 200 average trade volume (shares)", override=True)
      sys.exit()
    else:
      shares = np.random.randint(200, int(round(self.trade_vol * 2)) - 200)

    # Pick an exit offset (to take profit) if the trade goes in the agent's favor by a percentage of the
    # current price.
    exit_offset = int(round(0.01 * arb_target))

    # If the last traded price is too close to the value belief, don't trade.
    if abs(arb_target - self.value_belief) < self.TRADE_THRESHOLD:
      # No trade.
      pass
    elif self.value_belief > arb_target:
      # The agent believes the price should be higher.  Go long.

      # Place 1/2 the shares for immediate execution.  This will be at least 100.
      #mkt_shares = int(round(shares/2))
      mkt_shares = 100

      # Use base limit.
      #base_limit = self.value_belief - self.TRADE_THRESHOLD
      #base_limit = int(round(((self.value_belief - arb_target) / 2) + arb_target))
      #self.placeLimitOrder(self.symbol, mkt_shares, True, ask)
      base_limit = ask
      self.placeLimitOrder(self.symbol, mkt_shares, True, base_limit)

      rem_shares = shares - mkt_shares

      while rem_shares > 0:
        trade_shares = 100 if rem_shares >= 100 else rem_shares
        rem_shares -= trade_shares

        # Each 100 share lot's limit price is drawn from a one-sided gaussian with the peak
        # at the (possibly previous after our trade above) best ask.
        #rand = np.random.normal(0, ask * self.LIMIT_STD_CENTS)
        #TMP? Peak a little shy of the belief.
        rand = np.random.normal(0, base_limit * self.LIMIT_STD_CENTS)
        limit_price = int(round(base_limit - abs(rand)))

        self.placeLimitOrder(self.symbol, trade_shares, True, limit_price)

      # Also place a profit-taking exit, not designed to be immediately executed.
      #self.placeLimitOrder(self.symbol, shares, False, arb_target + exit_offset)

    else:
      # The agent believes the price should be lower.  Go short.

      # Place 1/2 the shares for immediate execution.  This will be at least 100.
      #mkt_shares = int(round(shares/2))
      mkt_shares = 100

      # Use base limit.
      #base_limit = self.value_belief + self.TRADE_THRESHOLD
      #base_limit = int(round(((arb_target - self.value_belief) / 2) + self.value_belief))
      #self.placeLimitOrder(self.symbol, mkt_shares, False, bid)
      base_limit = bid
      self.placeLimitOrder(self.symbol, mkt_shares, False, base_limit)

      rem_shares = shares - mkt_shares

      while rem_shares > 0:
        trade_shares = 100 if rem_shares >= 100 else rem_shares
        rem_shares -= trade_shares

        # Each 100 share lot's limit price is drawn from a one-sided gaussian with the peak
        # at the (possibly previous after our trade above) best bid.
        #rand = np.random.normal(0, bid * self.LIMIT_STD_CENTS)
        #TMP? Peak a little shy of the belief.
        rand = np.random.normal(0, base_limit * self.LIMIT_STD_CENTS)
        limit_price = int(round(base_limit + abs(rand)))

        self.placeLimitOrder(self.symbol, trade_shares, False, limit_price)

      # Also place a profit-taking exit, not designed to be immediately executed.
      #self.placeLimitOrder(self.symbol, shares, True, arb_target - exit_offset)


  def getWakeFrequency (self):
    return pd.Timedelta(np.random.randint(low = 0, high = pd.Timedelta(self.freq) / np.timedelta64(1, 'ns')), unit='ns')

  # Parent class defines:
  #def placeLimitOrder (self, symbol, quantity, is_buy_order, limit_price):

