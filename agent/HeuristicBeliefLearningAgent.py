from agent.ZeroIntelligenceAgent import ZeroIntelligenceAgent
from message.Message import Message
from util.util import log_print

from math import sqrt
import numpy as np
import pandas as pd
import sys

np.set_printoptions(threshold=np.inf)


class HeuristicBeliefLearningAgent(ZeroIntelligenceAgent):

    def __init__(self, id, name, type, symbol='IBM', starting_cash=100000, sigma_n=1000,
                 r_bar=100000, kappa=0.05, sigma_s=100000, q_max=10, sigma_pv=5000000, R_min=0,
                 R_max=250, eta=1.0, lambda_a=0.005, L=8, log_orders=False,
                 random_state=None):

        # Base class init.
        super().__init__(id, name, type, symbol=symbol, starting_cash=starting_cash, sigma_n=sigma_n,
                         r_bar=r_bar, kappa=kappa, sigma_s=sigma_s, q_max=q_max, sigma_pv=sigma_pv, R_min=R_min,
                         R_max=R_max, eta=eta, lambda_a=lambda_a, log_orders=log_orders,
                         random_state=random_state)

        # Store important parameters particular to the HBL agent.
        self.L = L  # length of order book history to use (number of transactions)

    def wakeup(self, currentTime):
        # Parent class handles discovery of exchange times and market_open wakeup call.
        # Also handles ZI-style "background agent" needs that are not specific to HBL.
        super().wakeup(currentTime)

        # Only if the superclass leaves the state as ACTIVE should we proceed with our
        # trading strategy.
        if self.state != 'ACTIVE': return

        # To make trade decisions, the HBL agent requires recent order stream information.
        self.getOrderStream(self.symbol, length=self.L)
        self.state = 'AWAITING_STREAM'

    def placeOrder(self):
        # Called when it is time for the agent to determine a limit price and place an order.
        # This method implements the HBL strategy and falls back to the ZI (superclass)
        # strategy if there is not enough information for the HBL strategy.

        # See if there is enough history for HBL.  If not, we will _exactly_ perform the
        # ZI placeOrder().  If so, we will use parts of ZI but compute our limit price
        # differently.  Note that we are not given orders more recent than the most recent
        # trade.

        if len(self.stream_history[self.symbol]) < self.L:
            # Not enough history for HBL.
            log_print("Insufficient history for HBL: length {}, L {}", len(self.stream_history[self.symbol]), self.L)
            super().placeOrder()
            return

        # There is enough history for HBL.

        # Use the superclass (ZI) method to obtain an observation, update internal estimate
        # parameters, decide to buy or sell, and calculate the total unit valuation, because
        # all of this logic is identical to ZI.
        v, buy = self.updateEstimates()

        # Walk through the visible order history and accumulate values needed for HBL's
        # estimation of successful transaction by limit price.
        low_p = sys.maxsize
        high_p = 0

        # Find the lowest and highest observed prices in the order history.
        for h in self.stream_history[self.symbol]:
            for id, order in h.items():
                p = order['limit_price']
                if p < low_p: low_p = p
                if p > high_p: high_p = p

        # Set up the ndarray we will use for our computation.
        # idx 0-7 are sa, sb, ua, ub, num, denom, Pr, Es
        nd = np.zeros((high_p - low_p + 1, 8))

        # Iterate through the history and compile our observations.
        for h in self.stream_history[self.symbol]:
            # h follows increasing "transactions into the past", with index zero being orders
            # after the most recent transaction.
            for id, order in h.items():
                p = order['limit_price']
                if p < low_p: low_p = p
                if p > high_p: high_p = p

                # For now if there are any transactions, consider the order successful.  For single
                # unit orders, this is sufficient.  For multi-unit orders,
                # we may wish to switch to a proportion of shares executed.
                if order['is_buy_order']:
                    if order['transactions']:
                        nd[p - low_p, 1] += 1
                    else:
                        nd[p - low_p, 3] += 1
                else:
                    if order['transactions']:
                        nd[p - low_p, 0] += 1
                    else:
                        nd[p - low_p, 2] += 1

        # Compute the sums and cumulative sums required, from our observations,
        # to drive the HBL's transaction probability estimates.
        if buy:
            nd[:, [0, 1, 2]] = np.cumsum(nd[:, [0, 1, 2]], axis=0)
            nd[::-1, 3] = np.cumsum(nd[::-1, 3], axis=0)
            nd[:, 4] = np.sum(nd[:, [0, 1, 2]], axis=1)
        else:
            nd[::-1, [0, 1, 3]] = np.cumsum(nd[::-1, [0, 1, 3]], axis=0)
            nd[:, 2] = np.cumsum(nd[:, 2], axis=0)
            nd[:, 4] = np.sum(nd[:, [0, 1, 3]], axis=1)

        nd[:, 5] = np.sum(nd[:, 0:4], axis=1)

        # Okay to ignore divide by zero errors here because we expect that in
        # some cases (0/0 can happen) and we immediately convert the resulting
        # nan to zero, which is the right answer for us.

        # Compute probability estimates for successful transaction at all price levels.
        with np.errstate(divide='ignore', invalid='ignore'):
            nd[:, 6] = np.nan_to_num(np.divide(nd[:, 4], nd[:, 5]))

        # Compute expected surplus for all price levels.
        if buy:
            nd[:, 7] = nd[:, 6] * (v - np.arange(low_p, high_p + 1))
        else:
            nd[:, 7] = nd[:, 6] * (np.arange(low_p, high_p + 1) - v)

        # Extract the price and other data for the maximum expected surplus.
        best_idx = np.argmax(nd[:, 7])
        best_Es, best_Pr = nd[best_idx, [7, 6]]
        best_p = low_p + best_idx

        # If the best expected surplus is positive, go for it.
        if best_Es > 0:
            log_print("Numpy: {} selects limit price {} with expected surplus {} (Pr = {:0.4f})", self.name, best_p,
                      int(round(best_Es)), best_Pr)

            # Place the constructed order.
            self.placeLimitOrder(self.symbol, 100, buy, int(round(best_p)))
        else:
            # Do nothing if best limit price has negative expected surplus with below code.
            log_print("Numpy: {} elects not to place an order (best expected surplus <= 0)", self.name)

            # OTHER OPTION 1: Allow negative expected surplus with below code.
            # log_print ("Numpy: {} placing undesirable order (best expected surplus <= 0)", self.name)
            # self.placeLimitOrder(self.symbol, 1, buy, int(round(best_p)))

            # OTHER OPTION 2: Force fallback to ZI logic on negative surplus with below code (including return).
            # log_print ("Numpy: no desirable order for {}, acting as ZI", self.name)
            # super().placeOrder()

    def receiveMessage(self, currentTime, msg):

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