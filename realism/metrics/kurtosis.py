from metrics.metric import Metric
from metrics.minutely_returns import MinutelyReturns
from scipy.stats import kurtosis


class Kurtosis(Metric):

    def __init__(self, intervals=4):
        self.intervals = intervals
        self.mr = MinutelyReturns()

    def compute(self, df):
        ks = []
        for i in range(1,self.intervals+1):
            temp = df[["close"]].resample("{}T".format(i)).last()
            rets = self.mr.compute(temp)
            ks.append(kurtosis(rets))
        return [ks]

    def visualize(self, simulated):
        self.line(simulated, title="Kurtosis", xlabel="Time scale (min)", ylabel="Average kurtosis", logy=True)
