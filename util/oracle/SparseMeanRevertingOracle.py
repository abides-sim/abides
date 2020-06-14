### The SparseMeanRevertingOracle produces a fundamental value time series for
### each requested symbol, and provides noisy observations of the fundamental
### value upon agent request.  This "sparse discrete" fundamental uses a
### combination of two processes to produce relatively realistic synthetic
### "values": a continuous mean-reverting Ornstein-Uhlenbeck process plus
### periodic "megashocks" which arrive following a Poisson process and have
### magnitude drawn from a bimodal normal distribution (overall mean zero,
### but with modes well away from zero).  This is necessary because OU itself
### is a single noisy return to the mean (from a perturbed initial state)
### that does not then depart the mean except in terms of minor "noise".

### Historical dates are effectively meaningless to this oracle.  It is driven by
### the numpy random number seed contained within the experimental config file.
### This oracle uses the nanoseconds portion of the current simulation time as
### discrete "time steps".

### This version of the MeanRevertingOracle expects agent activity to be spread
### across a large amount of time, with relatively sparse activity.  That is,
### agents each acting at realistic "retail" intervals, on the order of seconds
### or minutes, spread out across the day.

from util.oracle.MeanRevertingOracle import MeanRevertingOracle

import datetime as dt
import numpy as np
import pandas as pd
import os, random, sys

from math import exp, sqrt
from util.util import log_print


class SparseMeanRevertingOracle(MeanRevertingOracle):

  def __init__(self, mkt_open, mkt_close, symbols):
    # Symbols must be a dictionary of dictionaries with outer keys as symbol names and
    # inner keys: r_bar, kappa, sigma_s.
    self.mkt_open = mkt_open
    self.mkt_close = mkt_close
    self.symbols = symbols
    self.f_log = {}

    # The dictionary r holds the most recent fundamental values for each symbol.
    self.r = {}

    # The dictionary megashocks holds the time series of megashocks for each symbol.
    # The last one will always be in the future (relative to the current simulation time).
    #
    # Without these, the OU process just makes a noisy return to the mean and then stays there
    # with relatively minor noise.  Here we want them to follow a Poisson process, so we sample
    # from an exponential distribution for the separation intervals.
    self.megashocks = {}

    then = dt.datetime.now()

    # Note that each value in the self.r dictionary is a 2-tuple of the timestamp at
    # which the series was computed and the true fundamental value at that time.
    for symbol in symbols:
      s = symbols[symbol]
      log_print ("SparseMeanRevertingOracle computing initial fundamental value for {}", symbol)
      self.r[symbol] = (mkt_open, s['r_bar'])
      self.f_log[symbol] = [{ 'FundamentalTime' : mkt_open, 'FundamentalValue' : s['r_bar'] }]

      # Compute the time and value of the first megashock.  Note that while the values are
      # mean-zero, they are intentionally bimodal (i.e. we always want to push the stock
      # some, but we will tend to cancel out via pushes in opposite directions).
      ms_time_delta = np.random.exponential(scale=1.0 / s['megashock_lambda_a'])
      mst = self.mkt_open + pd.Timedelta(ms_time_delta, unit='ns')
      msv = s['random_state'].normal(loc = s['megashock_mean'], scale = sqrt(s['megashock_var']))
      msv = msv if s['random_state'].randint(2) == 0 else -msv

      self.megashocks[symbol] = [{ 'MegashockTime' : mst, 'MegashockValue' : msv }]


    now = dt.datetime.now()

    log_print ("SparseMeanRevertingOracle initialized for symbols {}", symbols)
    log_print ("SparseMeanRevertingOracle initialization took {}", now - then)


  # This method takes a requested timestamp to which we should advance the fundamental,
  # a value adjustment to apply after advancing time (must pass zero if none),
  # a symbol for which to advance time, a previous timestamp, and a previous fundamental
  # value.  The last two parameters should relate to the most recent time this method
  # was invoked.  It returns the new value.  As a side effect, it updates the log of
  # computed fundamental values.

  def compute_fundamental_at_timestamp(self, ts, v_adj, symbol, pt, pv):
    s = self.symbols[symbol]

    # This oracle uses the Ornstein-Uhlenbeck Process.  It is quite close to being a
    # continuous version of the discrete mean reverting process used in the regular
    # (dense) MeanRevertingOracle.

    # Compute the time delta from the previous time to the requested time.
    d = int((ts - pt) / np.timedelta64(1, 'ns'))

    # Extract the parameters for the OU process update.
    mu = s['r_bar']
    gamma = s['kappa']
    theta = s['fund_vol']

    # The OU process is able to skip any amount of time and sample the next desired value
    # from the appropriate distribution of possible values.
    v = s['random_state'].normal(loc = mu + (pv - mu) * (exp(-gamma * d)),
                                 scale = ((theta) / (2*gamma)) * (1 - exp(-2 * gamma * d)))

    # Apply the value adjustment that was passed in.
    v += v_adj

    # The process is not permitted to become negative.
    v = max(0, v)

    # For our purposes, the value must be rounded and converted to integer cents.
    v = int(round(v))

    # Cache the new time and value as the "previous" fundamental values.
    self.r[symbol] = (ts, v)
    
    # Append the change to the permanent log of fundamental values for this symbol.
    self.f_log[symbol].append({ 'FundamentalTime' : ts, 'FundamentalValue' : v })

    # Return the new value for the requested timestamp.
    return v


  # This method advances the fundamental value series for a single stock symbol,
  # using the OU process.  It may proceed in several steps due to our periodic
  # application of "megashocks" to push the stock price around, simulating
  # exogenous forces.
  def advance_fundamental_value_series(self, currentTime, symbol):

    # Generation of the fundamental value series uses a separate random state object
    # per symbol, which is part of the dictionary we maintain for each symbol.
    # Agent observations using the oracle will use an agent's random state object.
    s = self.symbols[symbol]

    # This is the previous fundamental time and value.
    pt, pv = self.r[symbol]

    # If time hasn't changed since the last advance, just use the current value.
    if currentTime <= pt: return pv

    # Otherwise, we have some work to do, advancing time and computing the fundamental.

    # We may not jump straight to the requested time, because we periodically apply
    # megashocks to push the series around (not always away from the mean) and we need
    # to compute OU at each of those times, so the aftereffects of the megashocks
    # properly affect the remaining OU interval.

    mst = self.megashocks[symbol][-1]['MegashockTime']
    msv = self.megashocks[symbol][-1]['MegashockValue']

    while mst < currentTime:
      # A megashock is scheduled to occur before the new time to which we are advancing.  Handle it.

      # Advance time from the previous time to the time of the megashock using the OU process and
      # then applying the next megashock value.
      v = self.compute_fundamental_at_timestamp(mst, msv, symbol, pt, pv)

      # Update our "previous" values for the next computation.
      pt, pv = mst, v

      # Since we just surpassed the last megashock time, compute the next one, which we might or
      # might not immediately consume.  This works just like the first time (in __init__()).

      mst = pt + pd.Timedelta('{}ns'.format(np.random.exponential(scale = 1.0 / s['megashock_lambda_a'])))
      msv = s['random_state'].normal(loc = s['megashock_mean'], scale = sqrt(s['megashock_var']))
      msv = msv if s['random_state'].randint(2) == 0 else -msv

      self.megashocks[symbol].append({ 'MegashockTime' : mst, 'MegashockValue' : msv })

      # The loop will continue until there are no more megashocks before the time requested
      # by the calling method.


    # Once there are no more megashocks to apply (i.e. the next megashock is in the future, after
    # currentTime), then finally advance using the OU process to the requested time.
    v = self.compute_fundamental_at_timestamp(currentTime, 0, symbol, pt, pv)

    return (v)


  # Return the daily open price for the symbol given.  In the case of the MeanRevertingOracle,
  # this will simply be the first fundamental value, which is also the fundamental mean.
  # We will use the mkt_open time as given, however, even if it disagrees with this.
  def getDailyOpenPrice (self, symbol, mkt_open=None):
  
    # The sparse oracle doesn't maintain full fundamental value history, but rather
    # advances on demand keeping only the most recent price, except for the opening
    # price.  Thus we cannot honor a mkt_open that isn't what we already expected.

    log_print ("Oracle: client requested {} at market open: {}", symbol, self.mkt_open)
  
    open = self.symbols[symbol]['r_bar']
    log_print ("Oracle: market open price was was {}", open)
  
    return open


  # Return a noisy observation of the current fundamental value.  While the fundamental
  # value for a given equity at a given time step does not change, multiple agents
  # observing that value will receive different observations.
  #
  # Only the Exchange or other privileged agents should use sigma_n==0.
  #
  # sigma_n is experimental observation variance.  NOTE: NOT STANDARD DEVIATION.
  #
  # Each agent must pass its RandomState object to observePrice.  This ensures that
  # each agent will receive the same answers across multiple same-seed simulations
  # even if a new agent has been added to the experiment.
  def observePrice(self, symbol, currentTime, sigma_n = 1000, random_state = None):
    # If the request is made after market close, return the close price.
    if currentTime >= self.mkt_close:
      r_t = self.advance_fundamental_value_series(self.mkt_close - pd.Timedelta('1ns'), symbol)
    else:
      r_t = self.advance_fundamental_value_series(currentTime, symbol)
 
    # Generate a noisy observation of fundamental value at the current time.
    if sigma_n == 0:
      obs = r_t
    else:
      obs = int(round(random_state.normal(loc=r_t, scale=sqrt(sigma_n))))
 
    log_print ("Oracle: current fundamental value is {} at {}", r_t, currentTime)
    log_print ("Oracle: giving client value observation {}", obs)
 
    # Reminder: all simulator prices are specified in integer cents.
    return obs