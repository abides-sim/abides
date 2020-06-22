from metrics.metric import Metric
from metrics.minutely_returns import MinutelyReturns
from scipy.stats import kurtosis
import numpy as np


class ReturnsVolatilityCorrelation(Metric):

    def __init__(self, intervals=4):
        self.mr = MinutelyReturns()

    def compute(self, df):
        returns = np.array(self.mr.compute(df))
        volatility = abs(returns)
        return [np.corrcoef(returns, volatility)[0,1]]

    def visualize(self, simulated):
        self.hist(simulated, title="Returns/Volatility Correlation", xlabel="Correlation coefficient", bins=50)
