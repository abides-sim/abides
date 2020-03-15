import numpy as np
import matplotlib.pyplot as plt

class Metric:

    # Returns a list of computed metrics
    def compute(self, df):
        raise NotImplementedError

    # Visualizes metric form list of simulated metric values and list of real metric values.
    def visualize(self, simulated, real, plot_real=True):
        raise NotImplementedError

    # Create an overlapping histogram of the provided data.
    def hist(self, simulated, real, title="Simulated vs. Historical Histogram", xlabel="Values", log=False, bins=75, clip=None, plot_real=True):

        real = np.array(real)
        for k, v in simulated.items():
            simulated[k] = np.array(v)

        if clip is not None:
            for k, v in simulated.items():
                simulated[k] = v[(v <= clip) & (v >= -1*clip)]
            real = real[(real <= clip) & (real >= -1*clip)]

        as_numpy = np.stack(list(simulated.values()))
        left = min(as_numpy.min(), min(real))
        right = max(as_numpy.max(), max(real))
        bins = np.linspace(left, right, bins)
        
        if plot_real:
            plt.hist(real, bins=bins, color="red", log=log, alpha=1, label="Historical", histtype="step", linewidth=3)

        # Show histograms
        for k, v in simulated.items():
            plt.hist(v, bins=bins, color=k[1], log=log, alpha=1, label=k[0], histtype="step", linewidth=3)

        plt.title(title + (" (log scale)" if log else "") + ("" if clip is None else " (clipped @ ±{})".format(clip)))
        plt.xlabel(xlabel)
        plt.ylabel(("Log " if log else "") + "Frequency")
        plt.legend()

    # Create a line plot of the simulated and real metrics. Also calculates error bars.
    def line(self, simulated, real, title="Simulated vs. Historical", xlabel="X", ylabel="Y", logy=False, plot_real=True):
        real = np.array(real)
        for k, v in simulated.items():
            simulated[k] = np.array(v)

        x = np.arange(real.shape[1])+1

        err_real = np.nanstd(real, axis=0)
        real = np.nanmean(real, axis=0)

        if plot_real:
            plt.plot(x, real, color="red", linewidth=4, label="Historical")
            #plt.fill_between(x, real-err_real, real+err_real, alpha=.1, color="red").set_linestyle('dashed')

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
