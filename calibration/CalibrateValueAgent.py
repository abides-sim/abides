from CalculationKernel import CalculationKernel, HiddenPrints
from model.LatencyModel import LatencyModel
from agent.ExchangeAgent import ExchangeAgent
from util.oracle.SparseMeanRevertingOracle import SparseMeanRevertingOracle
from agent.ValueAgent import ValueAgent
from util.order import LimitOrder
from util import util
from util import OrderBook

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
import time
import multiprocessing as mp


class CalibrateValueAgent:
    def __init__(self, symbol="JPM", fund_path=None, freq="min", num_agents=100, lambda_a=1e-12,
                 starting_cash=1e7, initial_sigma_n=1, historical_date="2019-06-28", seed=None):
        self.symbol = symbol
        try:
            self.fundamental_series = pd.read_pickle(fund_path, compression="bz2")
            self.fundamental_series.fillna(method="ffill", inplace=True)
            self.fundamental_returns = np.log(1 + self.fundamental_series.pct_change()).dropna().values
        except FileNotFoundError:
            raise FileNotFoundError("Path of fundamental prices does not exist.")

        self.freq = freq
        self.num_agents = num_agents
        self.lambda_a = lambda_a
        self.starting_cash = starting_cash
        self.midnight = pd.to_datetime(historical_date)
        self.initial_sigma_n = initial_sigma_n

        self.r_bar, self.kappa, self.sigma_s = CalibrateValueAgent.calibrateOU(self.fundamental_series, freq)

        if not seed:
            seed = int(pd.Timestamp.now().timestamp() * 1000000) % (2 ** 32 - 1)
        np.random.seed(seed)

        # What is the earliest available time for an agent to act during the
        # simulation?
        self.kernelStartTime = self.midnight

        # When should the Kernel shut down?  (This should be after market close.)
        # Here we go for 5 PM the same day.
        self.kernelStopTime = self.midnight + pd.to_timedelta('17:00:00')

        # This will configure the kernel with a default computation delay
        # (time penalty) for each agent's wakeup and recvMsg.  An agent
        # can change this at any time for itself.  (nanoseconds)
        self.defaultComputationDelay = 1000000000  # one second

        print("Calibrate freq: {}".format(self.freq))
        print("Configuration seed: {}\n".format(seed))

    def calibrateModel(self, batch_size=10):
        pass

    def evaluateLoss(self, sigma_n, batch_size=10, parallel=False):
        if not parallel:
            dist_sim = np.array([])
            for i in range(batch_size):
                mid_prices = self.generateMidPrices(sigma_n)
                mid_prices["return"] = np.log(1 + mid_prices["price"].pct_change())
                mid_prices.dropna(inplace=True)
                dist_sim = np.hstack([mid_prices["return"].values, dist_sim])
        else:
            dist_sim = np.array([])
            # num_cores = int(mp.cpu_count())
            # print("Cores on the computer is", num_cores)
            pool = mp.Pool(batch_size)
            results = [pool.apply_async(self.generateMidPrices, args=(sigma_n,)) for i in range(batch_size)]
            for p in results:
                mid_prices = p.get()
                mid_prices["return"] = np.log(1 + mid_prices["price"].pct_change())
                mid_prices.dropna(inplace=True)
                dist_sim = np.hstack([mid_prices["return"].values, dist_sim])
            pool.close()
            pool.join()
        print(len(self.fundamental_returns),len(dist_sim))
        return CalibrateValueAgent.KSDistance(self.fundamental_returns, dist_sim)

    def generateMidPrices(self, sigma_n):
        # Note: sigma_s is no longer used by the agents or the fundamental (for sparse discrete simulation).
        symbols = {
            self.symbol: {'r_bar': self.r_bar, 'kappa': self.kappa, 'agent_kappa': 1e-15, 'sigma_s': self.sigma_s,
                          'fund_vol': self.sigma_s, 'megashock_lambda_a': 1e-15, 'megashock_mean': 0.,
                          'megashock_var': 1e-15, "random_state": np.random.RandomState(
                    seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64'))}}

        util.silent_mode = True
        LimitOrder.silent_mode = True
        OrderBook.tqdm = False

        kernel = CalculationKernel("Calculation Kernel", random_state=np.random.RandomState(
            seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')))

        ### Configure the agents.  When conducting "agent of change" experiments, the
        ### new agents should be added at the END only.
        agent_count = 0
        agents = []
        agent_types = []

        # Let's open the exchange at 9:30 AM.
        mkt_open = self.midnight + pd.to_timedelta('09:30:00')

        # And close it at 4:00 PM.
        mkt_close = self.midnight + pd.to_timedelta('16:00:00')

        # Configure an appropriate oracle for all traded stocks.
        # All agents requiring the same type of Oracle will use the same oracle instance.
        oracle = SparseMeanRevertingOracle(mkt_open, mkt_close, symbols)

        # Create the exchange.
        num_exchanges = 1
        agents.extend([ExchangeAgent(j, "Exchange Agent {}".format(j), "ExchangeAgent", mkt_open, mkt_close,
                                     [s for s in symbols], log_orders=False, book_freq=self.freq, pipeline_delay=0,
                                     computation_delay=0, stream_history=10, random_state=np.random.RandomState(
                seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')))
                       for j in range(agent_count, agent_count + num_exchanges)])
        agent_types.extend(["ExchangeAgent" for j in range(num_exchanges)])
        agent_count += num_exchanges

        symbol = self.symbol
        s = symbols[symbol]

        # Some value agents.
        agents.extend([ValueAgent(j, "Value Agent {}".format(j),
                                  "ValueAgent {}".format(j),
                                  random_state=np.random.RandomState(
                                      seed=np.random.randint(low=0, high=2 ** 32, dtype='uint64')),
                                  log_orders=False, symbol=symbol, starting_cash=self.starting_cash,
                                  sigma_n=sigma_n, r_bar=s['r_bar'], kappa=s['agent_kappa'],
                                  sigma_s=s['fund_vol'],
                                  lambda_a=self.lambda_a) for j in range(agent_count, agent_count + self.num_agents)])
        agent_types.extend(["ValueAgent {}".format(j) for j in range(self.num_agents)])
        agent_count += self.num_agents

        # Config the latency model

        latency = None
        noise = None
        latency_model = None

        USE_NEW_MODEL = False

        ### BEGIN OLD LATENCY ATTRIBUTE CONFIGURATION ###

        ### Configure a simple message latency matrix for the agents.  Each entry is the minimum
        ### nanosecond delay on communication [from][to] agent ID.

        # Square numpy array with dimensions equal to total agent count.  Most agents are handled
        # at init, drawn from a uniform distribution from:
        # Times Square (3.9 miles from NYSE, approx. 21 microseconds at the speed of light) to:
        # Pike Place Starbucks in Seattle, WA (2402 miles, approx. 13 ms at the speed of light).
        # Other agents can be explicitly set afterward (and the mirror half of the matrix is also).

        if not USE_NEW_MODEL:
            # This configures all agents to a starting latency as described above.
            latency = np.random.uniform(low=21000, high=13000000, size=(len(agent_types), len(agent_types)))

            # Overriding the latency for certain agent pairs happens below, as does forcing mirroring
            # of the matrix to be symmetric.
            for i, t1 in zip(range(latency.shape[0]), agent_types):
                for j, t2 in zip(range(latency.shape[1]), agent_types):
                    # Three cases for symmetric array.  Set latency when j > i, copy it when i > j, same agent when i == j.
                    if j > i:
                        # Presently, strategy agents shouldn't be talking to each other, so we set them to extremely high latency.
                        if (t1 == "ZeroIntelligenceAgent" and t2 == "ZeroIntelligenceAgent"):
                            latency[i, j] = 1000000000 * 60 * 60 * 24  # Twenty-four hours.
                    elif i > j:
                        # This "bottom" half of the matrix simply mirrors the top.
                        latency[i, j] = latency[j, i]
                    else:
                        # This is the same agent.  How long does it take to reach localhost?  In our data center, it actually
                        # takes about 20 microseconds.
                        latency[i, j] = 20000

            # Configure a simple latency noise model for the agents.
            # Index is ns extra delay, value is probability of this delay being applied.
            noise = [0.25, 0.25, 0.20, 0.15, 0.10, 0.05]

        ### END OLD LATENCY ATTRIBUTE CONFIGURATION ###

        ### BEGIN NEW LATENCY MODEL CONFIGURATION ###

        else:
            # Get a new-style cubic LatencyModel from the networking literature.
            pairwise = (len(agent_types), len(agent_types))

            model_args = {'connected': True,

                          # All in NYC.
                          'min_latency': np.random.uniform(low=21000, high=100000, size=pairwise),
                          'jitter': 0.3,
                          'jitter_clip': 0.05,
                          'jitter_unit': 5,
                          }

            latency_model = LatencyModel(latency_model='cubic', random_state=np.random.RandomState(
                seed=np.random.randint(low=0, high=2 ** 31)), kwargs=model_args)

        # Start the kernel running.
        with HiddenPrints():
            midprices = kernel.runner(agents=agents, startTime=self.kernelStartTime,
                                      stopTime=self.kernelStopTime,
                                      agentLatencyModel=latency_model,
                                      agentLatency=latency, latencyNoise=noise,
                                      defaultComputationDelay=self.defaultComputationDelay,
                                      oracle=oracle, log_dir=None, return_value={self.symbol: "midprices"})

        return midprices[self.symbol]

    @staticmethod
    def calibrateOU(fundamental_series, freq="s"):
        # The most reasonable and accurate calibration requires freq="ns", since our
        # trading system take these parameters in time unit "ns". However it will be
        # out of memory to use "ns", so we take default freq="s" as a approximation.

        if 'FundamentalValue' in fundamental_series.columns:
            fundamental_series = fundamental_series[['FundamentalValue']]

        fundamental_series = fundamental_series.resample(freq).ffill()
        fundamental_series["diff"] = -fundamental_series['FundamentalValue'].diff(-1)
        fundamental_series.dropna(inplace=True)

        # Regression
        slope, intercept, r_value, p_value, std_err = linregress(fundamental_series['FundamentalValue'],
                                                                 fundamental_series["diff"])

        # Adjust parameters to freq "ns"
        freq_multiplier = pd.Timedelta("1" + freq) / pd.Timedelta("1ns")
        r_bar = -intercept / slope
        kappa = -slope / freq_multiplier
        sigma_s = (1 - r_value ** 2) * fundamental_series["diff"].var() / freq_multiplier

        return r_bar, kappa, sigma_s

    @staticmethod
    def KSDistance(data1, data2, rule_out_zero=True):
        if rule_out_zero:
            data1 = data1[data1 != 0.]
            data2 = data2[data2 != 0.]
        data1 = np.sort(data1)
        data2 = np.sort(data2)
        n1 = data1.shape[0]
        n2 = data2.shape[0]
        data_all = np.concatenate([data1, data2])
        cdf1 = np.searchsorted(data1, data_all, side='right') / (1.0 * n1)
        cdf2 = np.searchsorted(data2, data_all, side='right') / (1.0 * n2)
        d = np.max(np.absolute(cdf1 - cdf2))
        return d


if __name__ == "__main__":
    fund_path = "C:/Users/BJC/abides_fork/data/mid_prices/fundamental_AAPL.bz2"
    model = CalibrateValueAgent(fund_path=fund_path, symbol="JPM", lambda_a=1e-12, num_agents=100)

    sigma_n_grid=np.array([1,5,10,15,20,25,30,35,40],dtype=float)
    sigma_n_grid=sigma_n_grid**2

    time_start = time.time()
    loss_dict=dict()
    for sigma_n in sigma_n_grid:
        loss_dict[sigma_n]=model.evaluateLoss(sigma_n, batch_size=10,parallel=True)
        print("{} finished.".format(sigma_n))
    time_end = time.time()
    print('totally time', time_end - time_start)


    def plot_score_curve(score_dict, title=""):
        param_list = []
        score_list = []
        for param in score_dict:
            param_list.append(param)
            score_list.append(score_dict[param])

        best_idx = np.argmin(score_list)

        plt.plot(param_list, score_list)
        plt.plot([param_list[best_idx]], [score_list[best_idx]], marker='.', markersize=5, color="red")
        print((param_list[best_idx], score_list[best_idx]))

        plt.annotate('Best $d_{KS}$', xy=(param_list[best_idx], score_list[best_idx]), xycoords='data',
                     xytext=(0.8, 0.25), textcoords='axes fraction',
                     arrowprops=dict(facecolor='black', shrink=0.05, width=0.01, headwidth=5),
                     horizontalalignment='right', verticalalignment='top',
                     )

        plt.title(title)
        plt.show()

        return

    plot_score_curve(loss_dict)


    # time_start = time.time()
    # midprices_JPM = model.generateMidPrices(sigma_n=1)
    # time_end = time.time()
    # print('totally time', time_end - time_start)

    # midprices_JPM.plot()
    # plt.show()
    #
    # midprices_JPM["return"] = np.log(1 + midprices_JPM["price"].pct_change())
    # midprices_JPM.dropna(inplace=True)
    # print(sum(midprices_JPM["return"].values == 0.) / len(midprices_JPM["return"]))
    #
    # sns.distplot(midprices_JPM["return"].values)
    # plt.show()
