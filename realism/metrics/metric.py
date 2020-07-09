import numpy as np
import matplotlib.pyplot as plt


class Metric:

    # Returns a list of computed metrics
    def compute(self, df):
        raise NotImplementedError

    # Visualizes metric form list of simulated metric values and list of real metric values.
    def visualize(self, simulated):
        raise NotImplementedError

    # Create an overlapping histogram of the provided data.
    def hist(self, simulated, title="Simulation data histogram", xlabel="Values", log=False, bins=75, clip=None):
        for k, v in simulated.items():
            simulated[k] = np.array(v).reshape((len(v), 1))
        first_sim = simulated[list(simulated.keys())[0]]
        as_numpy = np.vstack(list(simulated.values()))
        left = min(as_numpy.min(), min(first_sim))
        right = max(as_numpy.max(), max(first_sim))
        bins = np.linspace(left, right, bins)

        # Show histograms
        for k, v in simulated.items():
            plt.hist(v, bins=bins, color=k[1], log=log, alpha=1, label=k[0], histtype="step", linewidth=3)

        plt.title(title + (" (log scale)" if log else "") + ("" if clip is None else " (clipped @ ±{})".format(clip)))
        plt.xlabel(xlabel)
        plt.ylabel(("Log " if log else "") + "Frequency")
        plt.legend()

    # Create a line plot of the simulated and real metrics. Also calculates error bars.
    def line(self, simulated, title="Simulation data", xlabel="X", ylabel="Y", logy=False):
        for k, v in simulated.items():
            simulated[k] = np.array(v)
        first_sim = simulated[list(simulated.keys())[0]]
        x = np.arange(first_sim.shape[1])+1

        for k, v in simulated.items():
            err_simulated = np.nanstd(v, axis=0)
            v = np.nanmean(v, axis=0)
            plt.plot(x, v, color=k[1], linewidth=4, label=k[0])
            #plt.fill_between(x, v-err_simulated, v+err_simulated, alpha=.1, color=k[1]).set_linestyle('dashed')

        plt.xticks(x)
        plt.title(title + (" (log scale)" if logy else ""))
        plt.xlabel(xlabel)
        plt.ylabel(ylabel + (" (log)" if logy else ""))
        if (logy):
            plt.yscale("log")
        plt.legend()
