import matplotlib.pyplot as plt
import numpy as np
from metrics.metric import Metric


class MinutelyReturns(Metric):

    def compute(self, df):
        df = df["close"]
        df = np.log(df)
        df = df.diff().dropna()
        return df.tolist()

    def visualize(self, simulated):
        self.hist(
            simulated,
            title="Minutely Log Returns",
            xlabel="Log Returns",
            log=True,
            clip=0.05,
        )
