from metrics.metric import Metric
from metrics.minutely_returns import MinutelyReturns

class AggregationNormality(Metric):

    def __init__(self):
        self.mr = MinutelyReturns()

    def compute(self, df):
        df = df[["close"]].resample("10T").last()
        return self.mr.compute(df)

    def visualize(self, simulated):
        self.hist(simulated, "Aggregation Normality (10 minutes)", "Log Returns", log=True, clip=.05)
