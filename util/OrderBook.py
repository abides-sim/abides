# Basic class for an order book for one symbol, in the style of the major US Stock Exchanges.
# List of bid prices (index zero is best bid), each with a list of LimitOrders.
# List of ask prices (index zero is best ask), each with a list of LimitOrders.
import sys

from message.Message import Message
from util.order.LimitOrder import LimitOrder
from util.util import log_print, be_silent

from copy import deepcopy
import pandas as pd
from pandas.io.json import json_normalize
from functools import reduce
from scipy.sparse import dok_matrix
from tqdm import tqdm


class OrderBook:

    # An OrderBook requires an owning agent object, which it will use to send messages
    # outbound via the simulator Kernel (notifications of order creation, rejection,
    # cancellation, execution, etc).
    def __init__(self, owner, symbol):
        self.owner = owner
        self.symbol = symbol
        self.bids = []
        self.asks = []
        self.last_trade = None

        # Create an empty list of dictionaries to log the full order book depth (price and volume) each time it changes.
        self.book_log = []
        self.quotes_seen = set()

        # Create an order history for the exchange to report to certain agent types.
        self.history = [{}]

        # Last timestamp the orderbook for that symbol was updated
        self.last_update_ts = None

        # Internal variable used for computing transacted volumes
        self._transacted_volume = {
            "unrolled_transactions": None,
            "self.history_previous_length": 0
        }

    def handleLimitOrder(self, order):
        # Matches a limit order or adds it to the order book.  Handles partial matches piecewise,
        # consuming all possible shares at the best price before moving on, without regard to
        # order size "fit" or minimizing number of transactions.  Sends one notification per
        # match.
        if order.symbol != self.symbol:
            log_print("{} order discarded.  Does not match OrderBook symbol: {}", order.symbol, self.symbol)
            return

        if (order.quantity <= 0) or (int(order.quantity) != order.quantity):
            log_print("{} order discarded.  Quantity ({}) must be a positive integer.", order.symbol, order.quantity)
            return

        # Add the order under index 0 of history: orders since the most recent trade.
        self.history[0][order.order_id] = {'entry_time': self.owner.currentTime,
                                           'quantity': order.quantity, 'is_buy_order': order.is_buy_order,
                                           'limit_price': order.limit_price, 'transactions': [],
                                           'modifications': [],
                                           'cancellations': []}

        matching = True

        self.prettyPrint()

        executed = []

        while matching:
            matched_order = deepcopy(self.executeOrder(order))

            if matched_order:
                # Decrement quantity on new order and notify traders of execution.
                filled_order = deepcopy(order)
                filled_order.quantity = matched_order.quantity
                filled_order.fill_price = matched_order.fill_price

                order.quantity -= filled_order.quantity

                log_print("MATCHED: new order {} vs old order {}", filled_order, matched_order)
                log_print("SENT: notifications of order execution to agents {} and {} for orders {} and {}",
                          filled_order.agent_id, matched_order.agent_id, filled_order.order_id, matched_order.order_id)

                self.owner.sendMessage(order.agent_id, Message({"msg": "ORDER_EXECUTED", "order": filled_order}))
                self.owner.sendMessage(matched_order.agent_id,
                                       Message({"msg": "ORDER_EXECUTED", "order": matched_order}))

                # Accumulate the volume and average share price of the currently executing inbound trade.
                executed.append((filled_order.quantity, filled_order.fill_price))

                if order.quantity <= 0:
                    matching = False

            else:
                # No matching order was found, so the new order enters the order book.  Notify the agent.
                self.enterOrder(deepcopy(order))

                log_print("ACCEPTED: new order {}", order)
                log_print("SENT: notifications of order acceptance to agent {} for order {}",
                          order.agent_id, order.order_id)

                self.owner.sendMessage(order.agent_id, Message({"msg": "ORDER_ACCEPTED", "order": order}))

                matching = False

        if not matching:
            # Now that we are done executing or accepting this order, log the new best bid and ask.
            if self.bids:
                self.owner.logEvent('BEST_BID', "{},{},{}".format(self.symbol,
                                                                  self.bids[0][0].limit_price,
                                                                  sum([o.quantity for o in self.bids[0]])))

            if self.asks:
                self.owner.logEvent('BEST_ASK', "{},{},{}".format(self.symbol,
                                                                  self.asks[0][0].limit_price,
                                                                  sum([o.quantity for o in self.asks[0]])))

            # Also log the last trade (total share quantity, average share price).
            if executed:
                trade_qty = 0
                trade_price = 0
                for q, p in executed:
                    log_print("Executed: {} @ {}", q, p)
                    trade_qty += q
                    trade_price += (p * q)

                avg_price = int(round(trade_price / trade_qty))
                log_print("Avg: {} @ ${:0.4f}", trade_qty, avg_price)
                self.owner.logEvent('LAST_TRADE', "{},${:0.4f}".format(trade_qty, avg_price))

                self.last_trade = avg_price

                # Transaction occurred, so advance indices.
                self.history.insert(0, {})

                # Truncate history to required length.
                self.history = self.history[:self.owner.stream_history + 1]

            # Finally, log the full depth of the order book, ONLY if we have been requested to store the order book
            # for later visualization.  (This is slow.)
            if self.owner.book_freq is not None:
                row = {'QuoteTime': self.owner.currentTime}
                for quote, volume in self.getInsideBids():
                    row[quote] = -volume
                    self.quotes_seen.add(quote)
                for quote, volume in self.getInsideAsks():
                    if quote in row:
                        if row[quote] is not None:
                            print(
                                "WARNING: THIS IS A REAL PROBLEM: an order book contains bids and asks at the same quote price!")
                    row[quote] = volume
                    self.quotes_seen.add(quote)
                self.book_log.append(row)
        self.last_update_ts = self.owner.currentTime
        self.prettyPrint()

    def handleMarketOrder(self, order):

        if order.symbol != self.symbol:
            log_print("{} order discarded.  Does not match OrderBook symbol: {}", order.symbol, self.symbol)
            return

        if (order.quantity <= 0) or (int(order.quantity) != order.quantity):
            log_print("{} order discarded.  Quantity ({}) must be a positive integer.", order.symbol, order.quantity)
            return

        orderbook_side = self.getInsideAsks() if order.is_buy_order else self.getInsideBids()

        limit_orders = {} # limit orders to be placed (key=price, value=quantity)
        order_quantity = order.quantity
        for price_level in orderbook_side:
            price, size = price_level[0], price_level[1]
            if order_quantity <= size:
                limit_orders[price] = order_quantity #i.e. the top of the book has enough volume for the full order
                break
            else:
                limit_orders[price] = size # i.e. not enough liquidity at the top of the book for the full order
                                           # therefore walk through the book until all the quantities are matched
                order_quantity -= size
                continue
        log_print("{} placing market order as multiple limit orders", order.symbol, order.quantity)
        for lo in limit_orders.items():
            p, q = lo[0], lo[1]
            limit_order = LimitOrder(order.agent_id, order.time_placed, order.symbol, q, order.is_buy_order, p)
            self.handleLimitOrder(limit_order)

    def executeOrder(self, order):
        # Finds a single best match for this order, without regard for quantity.
        # Returns the matched order or None if no match found.  DOES remove,
        # or decrement quantity from, the matched order from the order book
        # (i.e. executes at least a partial trade, if possible).

        # Track which (if any) existing order was matched with the current order.
        if order.is_buy_order:
            book = self.asks
        else:
            book = self.bids

        # TODO: Simplify?  It is ever possible to actually select an execution match
        # other than the best bid or best ask?  We may not need these execute loops.

        # First, examine the correct side of the order book for a match.
        if not book:
            # No orders on this side.
            return None
        elif not self.isMatch(order, book[0][0]):
            # There were orders on the right side, but the prices do not overlap.
            # Or: bid could not match with best ask, or vice versa.
            # Or: bid offer is below the lowest asking price, or vice versa.
            return None
        else:
            # There are orders on the right side, and the new order's price does fall
            # somewhere within them.  We can/will only match against the oldest order
            # among those with the best price.  (i.e. best price, then FIFO)

            # Note that book[i] is a LIST of all orders (oldest at index book[i][0]) at the same price.

            # The matched order might be only partially filled. (i.e. new order is smaller)
            if order.quantity >= book[0][0].quantity:
                # Consumed entire matched order.
                matched_order = book[0].pop(0)

                # If the matched price now has no orders, remove it completely.
                if not book[0]:
                    del book[0]

            else:
                # Consumed only part of matched order.
                matched_order = deepcopy(book[0][0])
                matched_order.quantity = order.quantity

                book[0][0].quantity -= matched_order.quantity

            # When two limit orders are matched, they execute at the price that
            # was being "advertised" in the order book.
            matched_order.fill_price = matched_order.limit_price

            # Record the transaction in the order history and push the indices
            # out one, possibly truncating to the maximum history length.

            # The incoming order is guaranteed to exist under index 0.
            self.history[0][order.order_id]['transactions'].append((self.owner.currentTime, order.quantity))

            # The pre-existing order may or may not still be in the recent history.
            for idx, orders in enumerate(self.history):
                if matched_order.order_id not in orders: continue

                # Found the matched order in history.  Update it with this transaction.
                self.history[idx][matched_order.order_id]['transactions'].append(
                    (self.owner.currentTime, matched_order.quantity))

            # Return (only the executed portion of) the matched order.
            return matched_order

    def isMatch(self, order, o):
        # Returns True if order 'o' can be matched against input 'order'.
        if order.is_buy_order == o.is_buy_order:
            print("WARNING: isMatch() called on orders of same type: {} vs {}".format(order, o))
            return False

        if order.is_buy_order and (order.limit_price >= o.limit_price):
            return True

        if not order.is_buy_order and (order.limit_price <= o.limit_price):
            return True

        return False

    def enterOrder(self, order):
        # Enters a limit order into the OrderBook in the appropriate location.
        # This does not test for matching/executing orders -- this function
        # should only be called after a failed match/execution attempt.

        if order.is_buy_order:
            book = self.bids
        else:
            book = self.asks

        if not book:
            # There were no orders on this side of the book.
            book.append([order])
        elif not self.isBetterPrice(order, book[-1][0]) and not self.isEqualPrice(order, book[-1][0]):
            # There were orders on this side, but this order is worse than all of them.
            # (New lowest bid or highest ask.)
            book.append([order])
        else:
            # There are orders on this side.  Insert this order in the correct position in the list.
            # Note that o is a LIST of all orders (oldest at index 0) at this same price.
            for i, o in enumerate(book):
                if self.isBetterPrice(order, o[0]):
                    book.insert(i, [order])
                    break
                elif self.isEqualPrice(order, o[0]):
                    book[i].append(order)
                    break

    def cancelOrder(self, order):
        # Attempts to cancel (the remaining, unexecuted portion of) a trade in the order book.
        # By definition, this pretty much has to be a limit order.  If the order cannot be found
        # in the order book (probably because it was already fully executed), presently there is
        # no message back to the agent.  This should possibly change to some kind of failed
        # cancellation message.  (?)  Otherwise, the agent receives ORDER_CANCELLED with the
        # order as the message body, with the cancelled quantity correctly represented as the
        # number of shares that had not already been executed.

        if order.is_buy_order:
            book = self.bids
        else:
            book = self.asks

        # If there are no orders on this side of the book, there is nothing to do.
        if not book: return

        # There are orders on this side.  Find the price level of the order to cancel,
        # then find the exact order and cancel it.
        # Note that o is a LIST of all orders (oldest at index 0) at this same price.
        for i, o in enumerate(book):
            if self.isEqualPrice(order, o[0]):
                # This is the correct price level.
                for ci, co in enumerate(book[i]):
                    if order.order_id == co.order_id:
                        # Cancel this order.
                        cancelled_order = book[i].pop(ci)

                        # Record cancellation of the order if it is still present in the recent history structure.
                        for idx, orders in enumerate(self.history):
                            if cancelled_order.order_id not in orders: continue

                            # Found the cancelled order in history.  Update it with the cancelation.
                            self.history[idx][cancelled_order.order_id]['cancellations'].append(
                                (self.owner.currentTime, cancelled_order.quantity))

                        # If the cancelled price now has no orders, remove it completely.
                        if not book[i]:
                            del book[i]

                        log_print("CANCELLED: order {}", order)
                        log_print("SENT: notifications of order cancellation to agent {} for order {}",
                                  cancelled_order.agent_id, cancelled_order.order_id)

                        self.owner.sendMessage(order.agent_id,
                                               Message({"msg": "ORDER_CANCELLED", "order": cancelled_order}))
                        # We found the order and cancelled it, so stop looking.
                        self.last_update_ts = self.owner.currentTime
                        return

    def modifyOrder(self, order, new_order):
        # Modifies the quantity of an existing limit order in the order book
        if not self.isSameOrder(order, new_order): return
        book = self.bids if order.is_buy_order else self.asks
        if not book: return
        for i, o in enumerate(book):
            if self.isEqualPrice(order, o[0]):
                for mi, mo in enumerate(book[i]):
                    if order.order_id == mo.order_id:
                        book[i][0] = new_order
                        for idx, orders in enumerate(self.history):
                            if new_order.order_id not in orders: continue
                            self.history[idx][new_order.order_id]['modifications'].append(
                                (self.owner.currentTime, new_order.quantity))
                            log_print("MODIFIED: order {}", order)
                            log_print("SENT: notifications of order modification to agent {} for order {}",
                                      new_order.agent_id, new_order.order_id)
                            self.owner.sendMessage(order.agent_id,
                                                   Message({"msg": "ORDER_MODIFIED", "new_order": new_order}))
        if order.is_buy_order:
            self.bids = book
        else:
            self.asks = book
        self.last_update_ts = self.owner.currentTime

    # Get the inside bid price(s) and share volume available at each price, to a limit
    # of "depth".  (i.e. inside price, inside 2 prices)  Returns a list of tuples:
    # list index is best bids (0 is best); each tuple is (price, total shares).
    def getInsideBids(self, depth=sys.maxsize):
        book = []
        for i in range(min(depth, len(self.bids))):
            qty = 0
            price = self.bids[i][0].limit_price
            for o in self.bids[i]:
                qty += o.quantity
            book.append((price, qty))

        return book

    # As above, except for ask price(s).
    def getInsideAsks(self, depth=sys.maxsize):
        book = []
        for i in range(min(depth, len(self.asks))):
            qty = 0
            price = self.asks[i][0].limit_price
            for o in self.asks[i]:
                qty += o.quantity
            book.append((price, qty))

        return book

    def _get_recent_history(self):
        """ Gets portion of self.history that has arrived since last call of self.get_transacted_volume.

            Also updates self._transacted_volume[self.history_previous_length]
        :return:
        """
        if self._transacted_volume["self.history_previous_length"] == 0:
            self._transacted_volume["self.history_previous_length"] = len(self.history)
            return self.history
        elif self._transacted_volume["self.history_previous_length"] == len(self.history):
            return {}
        else:
            idx = len(self.history) - self._transacted_volume["self.history_previous_length"] - 1
            recent_history = self.history[0:idx]
            self._transacted_volume["self.history_previous_length"] = len(self.history)
            return recent_history

    def _update_unrolled_transactions(self, recent_history):
        """ Updates self._transacted_volume["unrolled_transactions"] with data from recent_history

        :return:
        """
        new_unrolled_txn = self._unrolled_transactions_from_order_history(recent_history)
        old_unrolled_txn = self._transacted_volume["unrolled_transactions"]
        total_unrolled_txn = pd.concat([old_unrolled_txn, new_unrolled_txn], ignore_index=True)
        self._transacted_volume["unrolled_transactions"] = total_unrolled_txn

    def _unrolled_transactions_from_order_history(self, history):
        """ Returns a DataFrame with columns ['execution_time', 'quantity'] from a dictionary with same format as
            self.history, describing executed transactions.
        """
        # Load history into DataFrame
        unrolled_history = []
        for elem in history:
            for _, val in elem.items():
                unrolled_history.append(val)

        unrolled_history_df = pd.DataFrame(unrolled_history, columns=[
            'entry_time', 'quantity', 'is_buy_order', 'limit_price', 'transactions', 'modifications', 'cancellations'
        ])

        if unrolled_history_df.empty:
            return pd.DataFrame(columns=['execution_time', 'quantity'])

        executed_transactions = unrolled_history_df[unrolled_history_df['transactions'].map(lambda d: len(d)) > 0]  # remove cells that are an empty list

        #  Reshape into DataFrame with columns ['execution_time', 'quantity']
        transaction_list = [element for list_ in executed_transactions['transactions'].values for element in list_]
        unrolled_transactions = pd.DataFrame(transaction_list, columns=['execution_time', 'quantity'])
        unrolled_transactions = unrolled_transactions.sort_values(by=['execution_time'])
        unrolled_transactions = unrolled_transactions.drop_duplicates(keep='last')

        return unrolled_transactions

    def get_transacted_volume(self, lookback_period='10min'):
        """ Method retrieves the total transacted volume for a symbol over a lookback period finishing at the current
            simulation time.
        """

        # Update unrolled transactions DataFrame
        recent_history = self._get_recent_history()
        self._update_unrolled_transactions(recent_history)
        unrolled_transactions = self._transacted_volume["unrolled_transactions"]

        #  Get transacted volume in time window
        lookback_pd = pd.to_timedelta(lookback_period)
        window_start = self.owner.currentTime - lookback_pd
        executed_within_lookback_period = unrolled_transactions[unrolled_transactions['execution_time'] >= window_start]
        transacted_volume = executed_within_lookback_period['quantity'].sum()

        return transacted_volume

    # These could be moved to the LimitOrder class.  We could even operator overload them
    # into >, <, ==, etc.
    def isBetterPrice(self, order, o):
        # Returns True if order has a 'better' price than o.  (That is, a higher bid
        # or a lower ask.)  Must be same order type.
        if order.is_buy_order != o.is_buy_order:
            print("WARNING: isBetterPrice() called on orders of different type: {} vs {}".format(order, o))
            return False

        if order.is_buy_order and (order.limit_price > o.limit_price):
            return True

        if not order.is_buy_order and (order.limit_price < o.limit_price):
            return True

        return False

    def isEqualPrice(self, order, o):
        return order.limit_price == o.limit_price

    def isSameOrder(self, order, new_order):
        return order.order_id == new_order.order_id

    def book_log_to_df(self):
        """ Returns a pandas DataFrame constructed from the order book log, to be consumed by
            agent.ExchangeAgent.logOrderbookSnapshots.

            The first column of the DataFrame is `QuoteTime`. The succeeding columns are prices quoted during the
            simulation (as taken from self.quotes_seen).

            Each row is a snapshot at a specific time instance. If there is volume at a certain price level (negative
            for bids, positive for asks) this volume is written in the column corresponding to the price level. If there
            is no volume at a given price level, the corresponding column has a `0`.

            The data is stored in a sparse format, such that a value of `0` takes up no space.

        :return:
        """
        quotes = sorted(list(self.quotes_seen))
        log_len = len(self.book_log)
        quote_idx_dict = {quote: idx for idx, quote in enumerate(quotes)}
        quotes_times = []


        # Construct sparse matrix, where rows are timesteps, columns are quotes and elements are volume.
        S = dok_matrix((log_len, len(quotes)), dtype=int)  # Dictionary Of Keys based sparse matrix.

        for i, row in enumerate(tqdm(self.book_log, desc="Processing orderbook log")):
            quotes_times.append(row['QuoteTime'])
            for quote, vol in row.items():
                if quote == "QuoteTime":
                    continue
                S[i, quote_idx_dict[quote]] = vol

        S = S.tocsc()  # Convert this matrix to Compressed Sparse Column format for pandas to consume.
        df = pd.DataFrame.sparse.from_spmatrix(S, columns=quotes)
        df.insert(0, 'QuoteTime', quotes_times, allow_duplicates=True)
        return df

    # Print a nicely-formatted view of the current order book.
    def prettyPrint(self, silent=False):
        # Start at the highest ask price and move down.  Then switch to the highest bid price and move down.
        # Show the total volume at each price.  If silent is True, return the accumulated string and print nothing.

        # If the global silent flag is set, skip prettyPrinting entirely, as it takes a LOT of time.
        if be_silent: return ''

        book = "{} order book as of {}\n".format(self.symbol, self.owner.currentTime)
        book += "Last trades: simulated {:d}, historical {:d}\n".format(self.last_trade,
                                                                        self.owner.oracle.observePrice(self.symbol,
                                                                                                       self.owner.currentTime,
                                                                                                       sigma_n=0,
                                                                                                       random_state=self.owner.random_state))

        book += "{:10s}{:10s}{:10s}\n".format('BID', 'PRICE', 'ASK')
        book += "{:10s}{:10s}{:10s}\n".format('---', '-----', '---')

        for quote, volume in self.getInsideAsks()[-1::-1]:
            book += "{:10s}{:10s}{:10s}\n".format("", "{:d}".format(quote), "{:d}".format(volume))

        for quote, volume in self.getInsideBids():
            book += "{:10s}{:10s}{:10s}\n".format("{:d}".format(volume), "{:d}".format(quote), "")

        if silent: return book

        log_print(book)

