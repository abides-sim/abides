import numpy as np
import pandas as pd
from contextlib import contextmanager
import warnings
from scipy.spatial.distance import pdist, squareform


# General purpose utility functions for the simulator, attached to no particular class.
# Available to any agent or other module/utility.  Should not require references to
# any simulator object (kernel, agent, etc).

# Module level variable that can be changed by config files.
silent_mode = False


# This optional log_print function will call str.format(args) and print the
# result to stdout.  It will return immediately when silent mode is active.
# Use it for all permanent logging print statements to allow fastest possible
# execution when verbose flag is not set.  This is especially fast because
# the arguments will not even be formatted when in silent mode.
def log_print (str, *args):
  if not silent_mode: print (str.format(*args))


# Accessor method for the global silent_mode variable.
def be_silent ():
  return silent_mode


# Utility method to flatten nested lists.
def delist(list_of_lists):
    return [x for b in list_of_lists for x in b]

# Utility function to get agent wake up times to follow a U-quadratic distribution.
def get_wake_time(open_time, close_time, a=0, b=1):
    """ Draw a time U-quadratically distributed between open_time and close_time.
        For details on U-quadtratic distribution see https://en.wikipedia.org/wiki/U-quadratic_distribution
    """
    def cubic_pow(n):
        """ Helper function: returns *real* cube root of a float"""
        if n < 0:
            return -(-n) ** (1.0 / 3.0)
        else:
            return n ** (1.0 / 3.0)

    #  Use inverse transform sampling to obtain variable sampled from U-quadratic
    def u_quadratic_inverse_cdf(y):
        alpha = 12 / ((b - a) ** 3)
        beta = (b + a) / 2
        result = cubic_pow((3 / alpha) * y - (beta - a)**3 ) + beta
        return result

    uniform_0_1 = np.random.rand()
    random_multiplier = u_quadratic_inverse_cdf(uniform_0_1)
    wake_time = open_time + random_multiplier * (close_time - open_time)

    return wake_time

def numeric(s):
    """ Returns numeric type from string, stripping commas from the right.
        Adapted from https://stackoverflow.com/a/379966."""
    s = s.rstrip(',')
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s

def get_value_from_timestamp(s, ts):
    """ Get the value of s corresponding to closest datetime to ts.

        :param s: pandas Series with pd.DatetimeIndex
        :type s: pd.Series
        :param ts: timestamp at which to retrieve data
        :type ts: pd.Timestamp

    """

    ts_str = ts.strftime('%Y-%m-%d %H:%M:%S')
    s = s.loc[~s.index.duplicated(keep='last')]
    locs = s.index.get_loc(ts_str, method='nearest')
    out = s[locs][0] if (isinstance(s[locs], np.ndarray) or isinstance(s[locs], pd.Series)) else s[locs]

    return out

@contextmanager
def ignored(warning_str, *exceptions):
    """ Context manager that wraps the code block in a try except statement, catching specified exceptions and printing
        warning supplied by user.

        :param warning_str: Warning statement printed when exception encountered
        :param exceptions: an exception type, e.g. ValueError

        https://stackoverflow.com/a/15573313
    """
    try:
        yield
    except exceptions:
        warnings.warn(warning_str, UserWarning, stacklevel=1)
        if not silent_mode:
            print(warning_str)


def generate_uniform_random_pairwise_dist_on_line(left, right, num_points, random_state=None):
    """ Uniformly generate points on an interval, and return numpy array of pairwise distances between points.

    :param left: left endpoint of interval
    :param right: right endpoint of interval
    :param num_points: number of points to use
    :param random_state: np.RandomState object


    :return:
    """

    x_coords = random_state.uniform(low=left, high=right, size=num_points)
    x_coords = x_coords.reshape((x_coords.size, 1))
    out = pdist(x_coords, 'euclidean')
    return squareform(out)


def meters_to_light_ns(x):
    """ Converts x in units of meters to light nanoseconds

    :param x:
    :return:
    """
    x_lns = x / 299792458e-9
    x_lns = x_lns.astype(int)
    return x_lns


def validate_window_size(s):
    """ Check if s is integer or string 'adaptive'. """
    try:
        return int(s)
    except ValueError:
        if s.lower() == 'adaptive':
            return s.lower()
        else:
            raise ValueError(f'String {s} must be integer or string "adaptive".')


def sigmoid(x, beta):
    """ Numerically stable sigmoid function.
    Adapted from https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/"
    """
    if x >= 0:
        z = np.exp(-beta*x)
        return 1 / (1 + z)
    else:
        # if x is less than zero then z will be small, denom can't be
        # zero because it's 1+z.
        z = np.exp(beta*x)
        return z / (1 + z)
