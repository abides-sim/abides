from metrics.metric import Metric
import matplotlib.pyplot as plt
import numpy as np

class MinutelyReturns(Metric):

    def compute(self, df):
        df = df["close"]
        df = np.log(df)
        df = df.diff().dropna()
        return df.tolist()

    def visualize(self, simulated, real, plot_real=True):
        self.hist(simulated, real, title="Minutely Log Returns", xlabel="Log Returns", log=True, clip=.05, plot_real=plot_real)