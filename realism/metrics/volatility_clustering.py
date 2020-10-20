import random
from metrics.metric import Metric
from metrics.minutely_returns import MinutelyReturns
import pandas as pd
import numpy as np


class VolatilityClustering(Metric):

    def __init__(self, lags=10, mode="abs"):
        self.lags = lags
        modes = ["abs", "square"]
        if mode not in modes:
            raise ValueError("`mode` must be one of " + str(modes) + ".")
        self.mode = mode
        self.mr = MinutelyReturns()
        super().__init__()

    def compute(self, df):
        df = pd.Series(self.mr.compute(df))
        if self.mode == "abs":
            df = abs(df)
        elif self.mode == "square":
            df = df ** 2
        return [[df.autocorr(lag) for lag in range(1, self.lags+1)]]

    def visualize(self, simulated):
        self.line(simulated, "Volatility Clustering/Long Range Dependence", "Lag", "Correlation coefficient")
