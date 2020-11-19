from agent.TradingAgent import TradingAgent
from util.util import log_print

from math import sqrt
import numpy as np
import pandas as pd


class ValueAgent(TradingAgent):

    def __init__(self, id, name, type, symbol='IBM', starting_cash=100000, sigma_n=10000,
                 r_bar=100000, kappa=0.05, sigma_s=100000,
                 lambda_a=0.005, log_orders=False, log_to_file=True, random_state=None):

        # Base class init.
        super().__init__(id, name, type, starting_cash=starting_cash,
                         log_orders=log_orders, log_to_file=log_to_file, random_state=random_state)

        # Store important parameters particular to the ZI agent.
        self.symbol = symbol  # symbol to trade
        self.sigma_n = sigma_n  # observation noise variance
        self.r_bar = r_bar  # true mean fundamental value
        self.kappa = kappa  # mean reversion parameter
        self.sigma_s = sigma_s  # shock variance
        self.lambda_a = lambda_a  # mean arrival rate of ZI agents

        # The agent uses this to track whether it has begun its strategy or is still
        # handling pre-market tasks.
        self.trading = False

        # The agent begins in its "complete" state, not waiting for
        # any special event or condition.
        self.state = 'AWAITING_WAKEUP'

        # The agent maintains two priors: r_t and sigma_t (value and error estimates).
        self.r_t = r_bar
        self.sigma_t = 0

        # The agent must track its previous wake time, so it knows how many time
        # units have passed.
        self.prev_wake_time = None

        self.percent_aggr = 0.1                 #percent of time that the agent will aggress the spread
        self.size = np.random.randint(20, 50)   #size that the agent will be placing
        self.depth_spread = 2

    def kernelStarting(self, startTime):
        # self.kernel is set in Agent.kernelInitializing()
        # self.exchangeID is set in TradingAgent.kernelStarting()

        super().kernelStarting(startTime)

        self.oracle = self.kernel.oracle

    def kernelStopping(self):
        # Always call parent method to be safe.
        super().kernelStopping()

        # Print end of day valuation.
        H = int(round(self.getHoldings(self.symbol), -2) / 100)
        # May request real fundamental value from oracle as part of final cleanup/stats.

        #marked to fundamental
        rT = self.oracle.observePrice(self.symbol, self.currentTime, sigma_n=0, random_state=self.random_state)

        # final (real) fundamental value times shares held.
        surplus = rT * H

        log_print("surplus after holdings: {}", surplus)

        # Add ending cash value and subtract starting cash value.
        surplus += self.holdings['CASH'] - self.starting_cash
        surplus = float( surplus )/self.starting_cash

        self.logEvent('FINAL_VALUATION', surplus, True)

        log_print(
            "{} final report.  Holdings {}, end cash {}, start cash {}, final fundamental {}, surplus {}",
            self.name, H, self.holdings['CASH'], self.starting_cash, rT, surplus)

        #print("Final surplus", self.name, surplus)

    def wakeup(self, currentTime):
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
                log_print("{} is ready to start trading now.", self.name)

        # Steady state wakeup behavior starts here.

        # If we've been told the market has closed for the day, we will only request
        # final price information, then stop.
        if self.mkt_closed and (self.symbol in self.daily_close_price):
            # Market is closed and we already got the daily close price.
            return

        delta_time = self.random_state.exponential(scale=1.0 / self.lambda_a)
        self.setWakeup(currentTime + pd.Timedelta('{}ns'.format(int(round(delta_time)))))

        if self.mkt_closed and (not self.symbol in self.daily_close_price):
            self.getCurrentSpread(self.symbol)
            self.state = 'AWAITING_SPREAD'
            return

        self.cancelOrders()

        if type(self) == ValueAgent:
            self.getCurrentSpread(self.symbol)
            self.state = 'AWAITING_SPREAD'
        else:
            self.state = 'ACTIVE'

    def updateEstimates(self):
        # Called by a background agent that wishes to obtain a new fundamental observation,
        # update its internal estimation parameters, and compute a new total valuation for the
        # action it is considering.

        # The agent obtains a new noisy observation of the current fundamental value
        # and uses this to update its internal estimates in a Bayesian manner.
        obs_t = self.oracle.observePrice(self.symbol, self.currentTime, sigma_n=self.sigma_n,
                                         random_state=self.random_state)

        log_print("{} observed {} at {}", self.name, obs_t, self.currentTime)

        # Update internal estimates of the current fundamental value and our error of same.

        # If this is our first estimate, treat the previous wake time as "market open".
        if self.prev_wake_time is None: self.prev_wake_time = self.mkt_open

        # First, obtain an intermediate estimate of the fundamental value by advancing
        # time from the previous wake time to the current time, performing mean
        # reversion at each time step.

        # delta must be integer time steps since last wake
        delta = (self.currentTime - self.prev_wake_time) / np.timedelta64(1, 'ns')

        # Update r estimate for time advancement.
        r_tprime = (1 - (1 - self.kappa) ** delta) * self.r_bar
        r_tprime += ((1 - self.kappa) ** delta) * self.r_t

        # Update sigma estimate for time advancement.
        sigma_tprime = ((1 - self.kappa) ** (2 * delta)) * self.sigma_t
        sigma_tprime += ((1 - (1 - self.kappa) ** (2 * delta)) / (1 - (1 - self.kappa) ** 2)) * self.sigma_s

        # Apply the new observation, with "confidence" in the observation inversely proportional
        # to the observation noise, and "confidence" in the previous estimate inversely proportional
        # to the shock variance.
        self.r_t = (self.sigma_n / (self.sigma_n + sigma_tprime)) * r_tprime
        self.r_t += (sigma_tprime / (self.sigma_n + sigma_tprime)) * obs_t

        self.sigma_t = (self.sigma_n * self.sigma_t) / (self.sigma_n + self.sigma_t)

        # Now having a best estimate of the fundamental at time t, we can make our best estimate
        # of the final fundamental (for time T) as of current time t.  Delta is now the number
        # of time steps remaining until the simulated exchange closes.
        delta = max(0, (self.mkt_close - self.currentTime) / np.timedelta64(1, 'ns'))

        # IDEA: instead of letting agent "imagine time forward" to the end of the day,
        #       impose a maximum forward delta, like ten minutes or so.  This could make
        #       them think more like traders and less like long-term investors.  Add
        #       this line of code (keeping the max() line above) to try it.
        # delta = min(delta, 1000000000 * 60 * 10)

        r_T = (1 - (1 - self.kappa) ** delta) * self.r_bar
        r_T += ((1 - self.kappa) ** delta) * self.r_t

        # Our final fundamental estimate should be quantized to whole units of value.
        r_T = int(round(r_T))

        # Finally (for the final fundamental estimation section) remember the current
        # time as the previous wake time.
        self.prev_wake_time = self.currentTime

        log_print("{} estimates r_T = {} as of {}", self.name, r_T, self.currentTime)

        return r_T

    def placeOrder(self):
        #estimate final value of the fundamental price
        #used for surplus calculation
        r_T = self.updateEstimates()

        bid, bid_vol, ask, ask_vol = self.getKnownBidAsk(self.symbol)

        if bid and ask:
            mid = int((ask+bid)/2)
            spread = abs(ask - bid)

            if np.random.rand() < self.percent_aggr:
                adjust_int = 0
            else:
                adjust_int = np.random.randint( 0, self.depth_spread*spread )
                #adjustment to the limit price, allowed to post inside the spread
                #or deeper in the book as a passive order to maximize surplus

            if r_T < mid:
                #fundamental belief that price will go down, place a sell order
                buy = False
                p = bid + adjust_int #submit a market order to sell, limit order inside the spread or deeper in the book
            elif r_T >= mid:
                #fundamental belief that price will go up, buy order
                buy = True
                p = ask - adjust_int #submit a market order to buy, a limit order inside the spread or deeper in the book
        else:
            # initialize randomly
            buy = np.random.randint(0, 1 + 1)
            p = r_T

        # Place the order
        self.placeLimitOrder(self.symbol, self.size, buy, p)

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

                # But if the market is now closed, don't advance to placing orders.
                if self.mkt_closed: return

                # We now have the information needed to place a limit order with the eta
                # strategic threshold parameter.
                self.placeOrder()
                self.state = 'AWAITING_WAKEUP'

        # Cancel all open orders.
        # Return value: did we issue any cancellation requests?

    def cancelOrders(self):
        if not self.orders: return False

        for id, order in self.orders.items():
            self.cancelOrder(order)

        return True

    def getWakeFrequency(self):
        return pd.Timedelta(self.random_state.randint(low=0, high=100), unit='ns')