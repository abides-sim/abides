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


    def __init__(self, id, name, type, symbol=None, position_target=100, lurk_ticks=4, lurk_size=1000, lurk_cushion=200, levels=10, freq=1000000000, starting_cash=1000000, strat_start=None, log_orders=True, random_state=None):
        super().__init__(id, name, type, starting_cash=starting_cash, log_orders=log_orders, random_state=random_state)
        self.symbol = symbol
        self.position_target = position_target
        self.lurk_ticks = lurk_ticks
        self.lurk_size = lurk_size
        self.lurk_cushion = lurk_cushion
        self.levels = levels
        self.freq = freq
        self.last_market_data_update = None

        self.strat_start = strat_start
        self.strat_started = False

        self.plotme = []


    def wakeup(self, currentTime):
        super().wakeup(currentTime)
        super().requestDataSubscription(self.symbol, levels=self.levels, freq=self.freq)
        self.setComputationDelay(1)

    def receiveMessage(self, currentTime, msg):
        super().receiveMessage(currentTime, msg)
        if msg.body['msg'] == 'MARKET_DATA':
            if self.strat_start is not None:
              if currentTime < self.strat_start:
                return

            if not self.strat_started:
              print (f"SPOOFING STRAT STARTS NOW")
              self.strat_started = True

            # Note last time market data was received.
            self.last_market_data_update = currentTime

            # Get a numeric signed position from the holdings stucture.
            position = self.holdings[self.symbol] if self.symbol in self.holdings else 0

            # TODO: Make a function for this?
            request = 0
            for i, o in self.orders.items():
              if o.tag == "position":
                request += o.quantity

            log_print("Spoofer current holdings plus positioning orders: {} + {} == {}", position, request, position + request)
            print(f"Spoofer current holdings plus positioning orders: {position} + {request} == {position + request}")

            position += request

            if position != self.position_target:
                # This is potentially bad, so if we already have spoofing orders in place,
                # cancel them until our position is back where we want it.
                self.cancelSpoofOrders()

                # Adjust holdings to target for the agent's real position.
                adjustment = self.position_target - position

                direction = 'BUY' if adjustment > 0 else 'SELL'
                self.placeMarketOrder(self.symbol, direction, abs(adjustment), tag = "position")
                log_print("Adjusting holdings by {}", adjustment)
                print(f"Adjusting holdings by {adjustment}")

                # No spoofing unless and until we are in position.
                return
            else:
                log_print("Spoofer needs no position adjustment")
                print("Spoofer needs no position adjustment")

            # Spoof?  Don't spoof?
            #return

            # See where our unexecuted orders are relative to the inside spread.    
            bids, asks = msg.body['bids'], msg.body['asks']

            log_print("bids: {}, asks: {}", bids, asks)
            print(f"bids: {bids}, asks: {asks}")
            print(f"open orders: {self.orders}")


            ### HERE: consider changing spoof level to "how many shares should I keep between
            ###       me and the inside of the book" ?  We don't know when another agent
            ###       might place a huge order that eats into a place we don't want executed.
            ###       (Because a lot of our agents are dumb and don't mind causing impact.)

            # Potential features to consider (manual or learning).  How far from midpoint is 50%
            # of total liquidity in that direction?  What percentage of volume is within 1% of the midpoint?
            # Build a decision tree from those factors?

            # Preliminary effort.  Given my exact knowledge of the config, hardcode a successful spoofer.
            # Then try to make the decision points of the tree some kind of more generic ratio or feature,
            # one at a time, and ensure it still succeeds.

            ### For now, let's just keep an order of 1000 shares at order book depth 5, to see if
            ### that does anything.  (The OBI agent should look that far...)

            ### Will need position management code that repositions "spoofing" orders when they drift,
            ### but knows to ignore "position" orders.

            if len(bids) >= self.lurk_ticks + 1:
              spoof_price = bids[self.lurk_ticks][0]
              print (f"spoof_price is: {spoof_price}")
            else:
              spoof_price = None
              print (f"Insufficient book depth to spoof: {bids}")

            # Examine outstanding orders and add or adjust spoofing orders as needed.
            spoof_quantity = self.lurk_size

            for i, o in self.orders.items():
              if o.tag == "position":
                print (f"Ignoring position order.  WHICH WE SHOULD NOT HAVE: {o}")
              elif o.tag == "spoofing":
                print (f"Observed spoofing order: {o}")

                if not spoof_price or o.limit_price != spoof_price:
                  # Cancel this order.
                  print (f"Cancelling mispositioned order! {o}")
                  self.cancelOrder(o)
                else:
                  # Remember this order.
                  print (f"Retaining positioned quantity: {o.quantity}")
                  spoof_quantity -= o.quantity

                  # If we have too much quantity, start cancelling orders.
                  if spoof_quantity < 0:
                    print (f"Cancelling due to excess quantity! {o}")
                    self.cancelOrder(o)


            # If there is a spoof_price and our total quantity already at that price is below the
            # desired lurk_size, then place an order for the difference.
            if not spoof_price: return

            if spoof_quantity > 0:
              # TODO: remove hard-coded buy.
              print (f"Placed spoof order of size {spoof_quantity} at {spoof_price}.")
              self.placeLimitOrder(self.symbol, spoof_quantity, True, spoof_price, tag = 'spoofing')


        #self.plotme.append( { 'currentTime' : self.currentTime, 'midpoint' : (asks[0][0] + bids[0][0]) / 2, 'bid_pct' : bid_pct } )


    def cancelSpoofOrders(self):
        """ When something goes wrong, quickly cancel all spoofing orders, but not
            positioning orders which we hope to execute.
        """

        for i, o in self.orders.items():
          if o.tag == "spoofing":
            print (f"Emergency cancelling spoofing order: {o}")
            self.cancelOrder(o)


    def orderExecuted(self, order):
      """ Override TradingAgent.orderExecuted() to highlight anytime an order intended for spoofing
          is executed against, as we really want this not to happen.
      """

      if order.tag == 'position':
        print (f"Spoofing agent position order executed: {order}")
      elif order.tag == 'spoofing':
        print (f"----- Spoofing agent SPOOFING order executed: {order} -----")

      super().orderExecuted(order)


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


