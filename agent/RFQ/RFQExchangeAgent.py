from agent.FinancialAgent import FinancialAgent
from message.Message import Message
from util.OrderBook import OrderBook
from util.util import log_print

import datetime as dt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import pandas as pd
pd.set_option('display.max_rows', 500)

from copy import deepcopy


class ExchangeAgent(FinancialAgent):

  def __init__(self, id, name, type, mkt_open, mkt_close, symbols, book_freq='S', wide_book=False, pipeline_delay = 40000,
               computation_delay = 1, stream_history = 0, log_orders = False, random_state = None):

    super().__init__(id, name, type, random_state)

    # Do not request repeated wakeup calls.
    self.reschedule = False

    # Store this exchange's open and close times.
    self.mkt_open = mkt_open
    self.mkt_close = mkt_close

    # Right now, only the exchange agent has a parallel processing pipeline delay.  This is an additional
    # delay added only to order activity (placing orders, etc) and not simple inquiries (market operating
    # hours, etc).
    self.pipeline_delay = pipeline_delay

    # Computation delay is applied on every wakeup call or message received.
    self.computation_delay = computation_delay

    # The exchange maintains an order stream of all orders leading to the last L trades
    # to support certain agents from the auction literature (GD, HBL, etc).
    self.stream_history = stream_history

    # Log all order activity?
    self.log_orders = log_orders

    #TODO: continue