from agent.TradingAgent import TradingAgent

import pandas as pd


# Extends Impact agent to fire large order evenly over a predetermined time period
# Need to add: (1) duration, (2) number of wakeups, (3) desired execution size
class ImpactAgent(TradingAgent):

  def __init__(self, id, name, type, symbol = None, starting_cash = None, within = 0.01,
               impact = True, impact_time = None, impact_duration = 0, impact_trades = 0,
               impact_vol = None, random_state = None):
    # Base class init.
    super().__init__(id, name, type, starting_cash = starting_cash, random_state = random_state)

    self.symbol = symbol    # symbol to trade
    self.trading = False    # ready to trade
    self.traded = False     # has made its t trade

    # The amount of available "nearby" liquidity to consume when placing its order.
    self.within = within    # within this range of the inside price

    self.impact_time = impact_time          # When should we make the impact trade?
    self.impact_duration = impact_duration  # How long the agent should wait to submit the next trade
    self.impact_trades = impact_trades      # The number of trades to execute across
    self.impact_vol = impact_vol            # The total volume to execute across all trades

    # The agent begins in its "complete" state, not waiting for
    # any special event or condition.
    self.state = 'AWAITING_WAKEUP'

    # Controls whether the impact trade is actually placed.
    self.impact = impact


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
    # there's nothing to do except clean-up.
    if self.mkt_closed and (self.symbol in self.daily_close_price):
      # Market is closed and we already got the daily close price.
      return


    ### Impact agent operates at a specific time.
    if currentTime < self.impact_time:
      print ("Impact agent waiting for impact_time {}".format(self.impact_time))
      self.setWakeup(self.impact_time)
      return


    ### The impact agent only trades once, but we will monitor prices for
    ### the sake of performance.
    self.setWakeup(currentTime + pd.Timedelta('30m'))


    # If the market is closed and we haven't obtained the daily close price yet,
    # do that before we cease activity for the day.  Don't do any other behavior
    # after market close.
    #
    # Also, if we already made our one trade, do nothing except monitor prices.
    #if self.traded >= self.impact_trades or (self.mkt_closed and (not self.symbol in self.daily_close_price)):
    if self.traded or (self.mkt_closed and (not self.symbol in self.daily_close_price)):
      self.getLastTrade()
      self.state = 'AWAITING_LAST_TRADE'
      return

    #if self.traded < self.impact_trades:
      #self.setWakeup(currentTime + impact_duration)

    # The impact agent will place one order based on the current spread.
    self.getCurrentSpread()
    self.state = 'AWAITING_SPREAD'


  def receiveMessage (self, currentTime, msg):
    # Parent class schedules market open wakeup call once market open/close times are known.
    super().receiveMessage(currentTime, msg)

    # We have been awakened by something other than our scheduled wakeup.
    # If our internal state indicates we were waiting for a particular event,
    # check if we can transition to a new state.

    if self.state == 'AWAITING_SPREAD':
      # We were waiting for current spread information to make our trade.
      # If the message we just received is QUERY_SPREAD, that means we just got it.
      if msg.body['msg'] == 'QUERY_SPREAD':
        # Place our one trade.
        bid, bid_vol, ask, ask_vol = self.getKnownBidAsk(self.symbol)
        #bid_liq, ask_liq = self.getKnownLiquidity(self.symbol, within=self.within)
        print('within: ' + str(self.within))
        bid_liq, ask_liq = self.getKnownLiquidity(self.symbol, within=0.75)

        # Buy order.
        #direction, shares, price = True, int(round(ask_liq * self.greed)), ask

        # Sell order.  This should be a parameter, but isn't yet.
        #direction, shares = False, int(round(bid_liq * self.greed))
        direction, shares = False, int(round(bid_liq * 0.5))

        # Compute the limit price we must offer to ensure our order executes immediately.
        # This is essentially a workaround for the lack of true market orders in our
        # current simulation.
        price = self.computeRequiredPrice(direction, shares)

        # Actually place the order only if self.impact is true.
        if self.impact: 
          print ("Impact agent firing: {} {} @ {} @ {}".format('BUY' if direction else 'SELL', shares, self.dollarize(price), currentTime))
          self.placeLimitOrder (self.symbol, shares, direction, price)
        else:
          print ("Impact agent would fire: {} {} @ {} (but self.impact = False)".format('BUY' if direction else 'SELL', shares, self.dollarize(price)))

        self.traded = True
        self.state = 'AWAITING_WAKEUP'


  # Internal state and logic specific to this agent.

  def placeLimitOrder (self, symbol, quantity, is_buy_order, limit_price):
    super().placeLimitOrder(symbol, quantity, is_buy_order, limit_price, ignore_risk = True)

  # Computes required limit price to immediately execute a trade for the specified quantity
  # of shares.
  def computeRequiredPrice (self, direction, shares):
    book = self.known_asks[self.symbol] if direction else self.known_bids[self.symbol]

    # Start at the inside and add up the shares.
    t = 0

    for i in range(len(book)):
      p,v = book[i]
      t += v

      # If we have accumulated enough shares, return this price.
      # Need to also return if greater than the number of desired shares
      if t >= shares: return p

    # Not enough shares.  Just return worst price (highest ask, lowest bid).
    return book[-1][0]


  # Request the last trade price for our symbol.
  def getLastTrade (self):
    super().getLastTrade(self.symbol)


  # Request the spread for our symbol.
  def getCurrentSpread (self):
    # Impact agent gets depth 10000 on each side (probably everything).
    super().getCurrentSpread(self.symbol, 10000)


  def getWakeFrequency (self):
    return (pd.Timedelta('1ns'))


