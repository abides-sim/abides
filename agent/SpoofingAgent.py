from agent.TradingAgent import TradingAgent
import pandas as pd
from util.util import log_print

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class SpoofingAgent(TradingAgent):

    # The SpoofingAgent is a simple example of an agent that attempts to manipulate the price
    # of a stock for profit.  It takes a position in a particular stock, then maintains a significant
    # unexecuted order volume just beneath the spread.  The goal is to trick order book aware agents
    # into predicting a short-term price movement and positioning themselves accordingly, thus
    # driving stock prices in the direction that will be profitable for the spoofing agent.
    #
    # Parameters unique to this agent:
    # position_target: what real position should the spoofer try to profit from (sign indicates long/short)
    # lurk_ticks: how far from the inside should the spoofer lurk?
    # lurk_size: how much unexecuted volume should the spoofer maintain?
    # levels: how many order book levels does the agent need to consider?
    # exit_profit: how much profit (percent of entry level) should trigger our exit?
    # cooldown: how long should we wait after exit before re-entry?
    # freq: what should the order book subscription refresh frequency be?


    def __init__(self, id, name, type, symbol=None, position_target=100, lurk_ticks=1, lurk_size=1000, levels=10, freq=1000000000, starting_cash=1000000, log_orders=True, random_state=None):
        super().__init__(id, name, type, starting_cash=starting_cash, log_orders=log_orders, random_state=random_state)
        self.symbol = symbol
        self.position_target = position_target
        self.lurk_ticks = lurk_ticks
        self.lurk_size = lurk_size
        self.levels = levels
        self.freq = freq
        self.last_market_data_update = None

        self.delay_after_open = pd.Timedelta('30m')

        self.plotme = []


    def kernelStarting(self, startTime):
        self.strategy_start_time = startTime + self.delay_after_open
        super().kernelStarting(startTime)

    def wakeup(self, currentTime):
        super().wakeup(currentTime)
        if currentTime < self.strategy_start_time:
          self.setWakeup(self.strategy_start_time)
          return
        super().requestDataSubscription(self.symbol, levels=self.levels, freq=self.freq)
        self.setComputationDelay(1)

    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)
        if msg.body['msg'] == 'MARKET_DATA':
            # Note last time market data was received.
            self.last_market_data_update = currentTime

            # Get a numeric signed position from the holdings stucture.
            position = self.holdings[self.symbol] if self.symbol in self.holdings else 0
            log_print("Spoofer current holdings: {}", position)
            print(f"Spoofer current holdings: {position}")

            if position != self.position_target:
                # Adjust holdings to target for the agent's real position.
                adjustment = self.position_target - position

                direction = 'BUY' if adjustment > 0 else 'SELL'
                self.placeMarketOrder(self.symbol, direction, abs(adjustment))
                log_print("Adjusting holdings by {}", adjustment)
                print(f"Adjusting holdings by {adjustment}")
            else:
                log_print("Spoofer needs no position adjustment")
                print("Spoofer needs no position adjustment")

            # See where our unexecuted orders are relative to the inside spread.    
            bids, asks = msg.body['bids'], msg.body['asks']

            log_print("bids: {}, asks: {}", bids, asks)
            print(f"bids: {bids}, asks: {asks}")
            print(f"open orders: {self.orders}")


            ### HERE: consider changing spoof level to "how many shares should I keep between
            ###       me and the inside of the book" ?  We don't know when another agent
            ###       might place a huge order that eats into a place we don't want executed.
            ###       (Because a lot of our agents are dumb and don't mind causing impact.)

            return

            # OBI strategy.
            target = 0

            if bid_liq == 0 or ask_liq == 0:
                log_print("OBI agent inactive: zero bid or ask liquidity")
                return
            else:
                # bid_pct encapsulates both sides of the question, as a normalized expression
                # representing what fraction of total visible volume is on the buy side.
                bid_pct = bid_liq / (bid_liq + ask_liq)

                # If we are short, we need to decide if we should hold or exit.
                if self.is_short:
                    # Update trailing stop.
                    if bid_pct - self.trail_dist > self.trailing_stop:
                        log_print("Trailing stop updated: new > old ({:2f} > {:2f})", bid_pct - self.trail_dist, self.trailing_stop)
                        self.trailing_stop = bid_pct - self.trail_dist
                    else:
                        log_print("Trailing stop remains: potential < old ({:2f} < {:2f})", bid_pct - self.trail_dist, self.trailing_stop)

                    # Check the trailing stop.
                    if bid_pct < self.trailing_stop: 
                        log_print("OBI agent exiting short position: bid_pct < trailing_stop ({:2f} < {:2f})", bid_pct, self.trailing_stop)
                        target = 0
                        self.is_short = False
                        self.trailing_stop = None
                    else:
                        log_print("OBI agent holding short position: bid_pct > trailing_stop ({:2f} > {:2f})", bid_pct, self.trailing_stop)
                        target = -100
                # If we are long, we need to decide if we should hold or exit.
                elif self.is_long:
                    if bid_pct + self.trail_dist < self.trailing_stop:
                        log_print("Trailing stop updated: new < old ({:2f} < {:2f})", bid_pct + self.trail_dist, self.trailing_stop)
                        self.trailing_stop = bid_pct + self.trail_dist
                    else:
                        log_print("Trailing stop remains: potential > old ({:2f} > {:2f})", bid_pct + self.trail_dist, self.trailing_stop)

                    # Check the trailing stop.
                    if bid_pct > self.trailing_stop: 
                        log_print("OBI agent exiting long position: bid_pct > trailing_stop ({:2f} > {:2f})", bid_pct, self.trailing_stop)
                        target = 0
                        self.is_long = False
                        self.trailing_stop = None
                    else:
                        log_print("OBI agent holding long position: bid_pct < trailing_stop ({:2f} < {:2f})", bid_pct, self.trailing_stop)
                        target = 100
                # If we are flat, we need to decide if we should enter (long or short).
                else:
                  if bid_pct < (0.5 - self.entry_threshold):
                      log_print("OBI agent entering long position: bid_pct < entry_threshold ({:2f} < {:2f})", bid_pct, 0.5 - self.entry_threshold)
                      target = 100
                      self.is_long = True
                      self.trailing_stop = bid_pct + self.trail_dist
                      log_print("Initial trailing stop: {:2f}", self.trailing_stop)
                  elif bid_pct > (0.5 + self.entry_threshold):
                      log_print("OBI agent entering short position: bid_pct > entry_threshold ({:2f} > {:2f})", bid_pct, 0.5 + self.entry_threshold)
                      target = -100
                      self.is_short = True
                      self.trailing_stop = bid_pct - self.trail_dist
                      log_print("Initial trailing stop: {:2f}", self.trailing_stop)
                  else:
                      log_print("OBI agent staying flat: long_entry < bid_pct < short_entry ({:2f} < {:2f} < {:2f})", 0.5 - self.entry_threshold, bid_pct, 0.5 + self.entry_threshold)
                      target = 0


                self.plotme.append( { 'currentTime' : self.currentTime, 'midpoint' : (asks[0][0] + bids[0][0]) / 2, 'bid_pct' : bid_pct } )


    def getWakeFrequency(self):
        return pd.Timedelta('1s')


    # Cancel all open orders.
    # Return value: did we issue any cancellation requests?
    def cancelOrders(self):
        if not self.orders: return False

        for id, order in self.orders.items():
            self.cancelOrder(order)

        return True


    # Lifecycle.
    def kernelTerminating(self):
      # Plotting code is probably not needed here long term, but helps during development.

      #df = pd.DataFrame(self.plotme)
      #df.set_index('currentTime', inplace=True)
      #df.rolling(30).mean().plot(secondary_y=['bid_pct'], figsize=(12,9))
      #plt.show()
      super().kernelTerminating()


