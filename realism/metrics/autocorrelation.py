import random
from metrics.metric import Metric
from metrics.minutely_returns import MinutelyReturns
import pandas as pd

class Autocorrelation(Metric):

    def __init__(self, lag=1, window=30):
        self.lag = lag
        self.window = window
        self.mr = MinutelyReturns()
        super().__init__()

    def compute(self, df):
        df = pd.Series(self.mr.compute(df))
        df = df.rolling(self.window, center=True).apply(lambda x: x.autocorr(lag=self.lag), raw=False)
        return df.dropna().tolist()

    def visualize(self, simulated, real, plot_real=True):
        min_sim = min([len(x) for x in simulated.values()])
        min_size = min(min_sim, len(real))
        random.shuffle(real)
        real = real[:min_size]
        for k, v in simulated.items():
            random.shuffle(v)
            simulated[k] = v[:min_size]
        self.hist(simulated, real, title="Autocorrelation (lag={}, window={})".format(self.lag, self.window), xlabel="Correlation coefficient", log=False, plot_real=plot_real)

