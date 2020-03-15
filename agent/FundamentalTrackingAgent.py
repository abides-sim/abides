from agent.TradingAgent import TradingAgent
from util.util import log_print
import numpy as np
import pandas as pd


class FundamentalTrackingAgent(TradingAgent):
    """ Agent who collects and saves to disk noise-free observations of the fundamental. """

    def __init__(self, id, name, type, log_frequency, symbol, log_orders=False):
        """ Constructor for FundamentalTrackingAgent

            :param log_frequency: Frequency to update log (in nanoseconds)
            :param symbol: symbol for which fundamental is being logged
        """
        super().__init__(id, name, type, starting_cash=0, log_orders=log_orders,
                         random_state=np.random.RandomState(seed=np.random.randint(low=0,
                                                                                   high=2 ** 32,
                                                                                   dtype='uint64')))

        self.log_freqency = log_frequency
        self.fundamental_series = []
        self.symbol = symbol

    def kernelStarting(self, startTime):
        # self.kernel is set in Agent.kernelInitializing()
        # self.exchangeID is set in TradingAgent.kernelStarting()
        super().kernelStarting(startTime)
        self.oracle = self.kernel.oracle


    def kernelStopping(self):
        """ Stops kernel and saves fundamental series to disk. """
        # Always call parent method to be safe.
        super().kernelStopping()
        self.writeFundamental()

    def measureFundamental(self):
        """ Saves the fundamental value at self.currentTime to self.fundamental_series. """
        obs_t = self.oracle.observePrice(self.symbol, self.currentTime, sigma_n=0)
        self.fundamental_series.append({'FundamentalTime': self.currentTime, 'FundamentalValue': obs_t})

    def wakeup(self, currentTime):
        """ Advances agent in time and takes measurement of fundamental. """
        # Parent class handles discovery of exchange times and market_open wakeup call.
        super().wakeup(currentTime)

        if not self.mkt_open or not self.mkt_close:
            # No logging if market is closed
            return

        self.measureFundamental()
        self.setWakeup(currentTime + self.getWakeFrequency())

    def writeFundamental(self):
        """ Logs fundamental series to file. """
        dfFund = pd.DataFrame(self.fundamental_series)
        dfFund.set_index('FundamentalTime', inplace=True)
        self.writeLog(dfFund, filename='fundamental_{symbol}_freq_{self.log_frequency}_ns'.format(self.symbol))

        print("Noise-free fundamental archival complete.")

    def getWakeFrequency(self):
        return pd.Timedelta(self.log_freqency, unit='ns')
