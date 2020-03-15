from metrics.metric import Metric
from metrics.minutely_returns import MinutelyReturns
from scipy.stats import kurtosis
import numpy as np

class VolumeVolatilityCorrelation(Metric):

    def __init__(self, intervals=4):
        self.mr = MinutelyReturns()

    def compute(self, df):
        volatility = abs(np.array(self.mr.compute(df)))
        volume = df["volume"].iloc[1:].values
        return [np.corrcoef(volume, volatility)[0,1]]

    def visualize(self, simulated, real, plot_real=True):
        self.hist(simulated, real, title="Volume/Volatility Correlation", xlabel="Correlation coefficient", plot_real=plot_real)
