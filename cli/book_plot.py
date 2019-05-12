import copy
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import sys

from matplotlib.colors import LogNorm

from joblib import Memory

# Auto-detect terminal width.
pd.options.display.width = None
pd.options.display.max_rows = 1000
pd.options.display.max_colwidth = 200

# Initialize a persistent memcache.
mem = Memory(cachedir='./.cached_plot_book', verbose=0)
mem_hist = Memory(cachedir='./.cached_plot_book_historical', verbose=0)
mem_hist_plot = Memory(cachedir='./.cached_plot_book_historical_heatmap', verbose=0)


# Turn these into command line parameters.
SHOW_BEST = False
TIME_STEPS = False
PLOT_HISTORICAL = False


# Used to read and cache simulated quotes (best bid/ask).
# Doesn't actually pay attention to symbols yet.
#@mem.cache
def read_book_quotes (file):
  print ("Simulated quotes were not cached.  This will take a minute.")
  df = pd.read_pickle(file, compression='bz2')

  if len(df) <= 0:
    print ("There appear to be no simulated quotes.")
    sys.exit()

  print ("Cached simulated quotes.")
  return df


# Used to read historical national best bid/ask spread.
@mem_hist.cache
def read_historical_quotes (file, symbol):
  print ("Historical quotes were not cached.  This will take a minute.")
  df = pd.read_pickle(file, compression='bz2')

  if len(df) <= 0:
    print ("There appear to be no historical quotes.")
    sys.exit()

  df = df.loc[symbol]

  return df


# Used to cache the transformed historical dataframe for a symbol.
@mem_hist_plot.cache
def prepare_histogram (df_hist):
  print ("Historical dataframe transformation was not cached.  This will take a minute.")

  min_quote = df_hist['BEST_BID'].min()
  max_quote = df_hist['BEST_ASK'].max()

  quote_range = pd.Series(np.arange(min_quote, max_quote + 0.01, 0.01)).round(2).map(str)
  quote_range = quote_range.str.pad(6, side='right', fillchar='0')

  df = pd.DataFrame(index=df_hist.index, columns=quote_range)
  df[:] = 0

  i = 0

  for idx in df.index:
    if i % 1000 == 0: print ("Caching {}".format(idx))

    col = '{:0.2f}'.format(round(df_hist.loc[idx].BEST_BID, 2))
    val = -df_hist.loc[idx].BEST_BIDSIZ
    df.loc[idx,col] = val

    col = '{:0.2f}'.format(round(df_hist.loc[idx].BEST_ASK, 2))
    val = df_hist.loc[idx].BEST_ASKSIZ
    df.loc[idx,col] = val

    i += 1

  return df


# Main program starts here.

if len(sys.argv) < 2:
  print ("Usage: python book_plot.py <Simulator Order Book DataFrame file>")
  sys.exit()

book_file = sys.argv[1]

print ("Visualizing order book from {}".format(book_file))

sns.set()

df_book = read_book_quotes(book_file)
#df_hist = read_historical_quotes('./data/nbbo/nbbo_2018/nbbom_20180518.bgz', 'IBM')

fig = plt.figure(figsize=(12,9))

# Use this to make all volume positive (ASK volume is negative in the dataframe).
#df_book.Volume = df_book.Volume.abs()

# Use this to swap the sign of BID vs ASK volume (to better fit a colormap, perhaps).
#df_book.Volume = df_book.Volume * -1

# Use this to clip volume to an upper limit.
#df_book.Volume = df_book.Volume.clip(lower=-400,upper=400)

# Use this to turn zero volume into np.nan (useful for some plot types).
#df_book.Volume[df_book.Volume == 0] = np.nan

# This section colors the best bid, best ask, and bid/ask midpoint
# differently from the rest of the heatmap below, by substituting
# special values at those indices.  It is important to do this while
# the price index is still numeric.  We use a value outside the
# range of actual order book volumes (selected dynamically).
min_volume = df_book.Volume.min()
max_volume = df_book.Volume.max()

best_bid_value = min_volume - 1000
best_ask_value = min_volume - 1001
midpoint_value = min_volume - 1002

# This converts the DateTimeIndex to integer nanoseconds since market open.  We use
# these as our time steps for discrete time simulations (e.g. SRG config).
if TIME_STEPS:
  df_book = df_book.unstack(1)
  t = df_book.index.get_level_values(0) - df_book.index.get_level_values(0)[0]
  df_book.index = (t / np.timedelta64(1, 'ns')).astype(np.int64)
  df_book = df_book.stack()


# Use this to restrict plotting to a certain time of day.  Depending on quote frequency,
# plotting could be very slow without this.
#df_book = df_book.unstack(1)
#df_book = df_book.between_time('11:50:00', '12:10:00')
#df_book = df_book.stack()



if SHOW_BEST:

  df_book = df_book.unstack(1)
  df_book.columns = df_book.columns.droplevel(0)

  # Now row (single) index is time.  Column (single) index is quote price.

  # In temporary data frame, find best bid per (time) row.
  # Copy bids only.
  best_bid = df_book[df_book < 0].copy()

  # Replace every non-zero bid volume with the column header (quote price) instead.
  for col in best_bid.columns:
    c = best_bid[col]
    c[c < 0] = col

  # Copy asks only.
  best_ask = df_book[df_book > 0].copy()

  # Replace every non-zero ask volume with the column header (quote price) instead.
  for col in best_ask.columns:
    c = best_ask[col]
    c[c > 0] = col

  # In a new column in each temporary data frame, compute the best bid or ask.
  best_bid['best'] = best_bid.idxmax(axis=1)
  best_ask['best'] = best_ask.idxmin(axis=1)

  # Iterate over the index (all three DF have the same index) and set the special
  # best bid/ask value in the correct column(s) per row.  Also compute and include
  # the midpoint where possible.
  for idx in df_book.index:
    bb = best_bid.loc[idx,'best']
    #if bb: df_book.loc[idx,bb] = best_bid_value

    ba = best_ask.loc[idx,'best']
    #if ba: df_book.loc[idx,ba] = best_ask_value

    if ba and bb: df_book.loc[idx,round((ba+bb)/2)] = midpoint_value


  # Put the data frame indices back the way they were and ensure it is a DataFrame,
  # not a Series.
  df_book = df_book.stack()
  df_book = pd.DataFrame(data=df_book)
  df_book.columns = ['Volume']


# Change the MultiIndex to time and dollars.
df_book['Time'] = df_book.index.get_level_values(0)
df_book['Price'] = df_book.index.get_level_values(1)

# Use this to restrict plotting to a certain range of prices.
#df_book = df_book.loc[(df_book.Price > 98500) & (df_book.Price < 101500)]

# Use this to pad price strings for appearance.
#df_book.Price = df_book.Price.map(str)
#df_book.Price = df_book.Price.str.pad(6, side='right', fillchar='0')

df_book.set_index(['Time', 'Price'], inplace=True)

# This section makes a 2-D histogram (time vs price, color == volume)
unstacked = df_book.unstack(1)
if not TIME_STEPS: unstacked.index = unstacked.index.time
unstacked.columns = unstacked.columns.droplevel(0)

with sns.axes_style("white"):
  ax = sns.heatmap(unstacked, cmap='seismic', mask=unstacked < min_volume, vmin=min_volume, cbar_kws={'label': 'Shares Available'}, center=0, antialiased = False)

ax.set(xlabel='Quoted Price', ylabel='Quote Time')

# Plot layers of best bid, best ask, and midpoint in special colors.
#best_bids = unstacked[unstacked == best_bid_value].copy().notnull()
midpoints = unstacked[unstacked == midpoint_value].copy().notnull()
#best_asks = unstacked[unstacked == best_ask_value].copy().notnull()

if SHOW_BEST:
  #sns.heatmap(best_bids, cmap=['xkcd:hot purple'], mask=~best_bids, cbar=False, ax=ax)
  #sns.heatmap(midpoints, cmap=['xkcd:hot green'], mask=~midpoints, cbar=False, ax=ax)
  sns.heatmap(midpoints, cmap=['black'], mask=~midpoints, cbar=False, ax=ax)
  #sns.heatmap(best_asks, cmap=['xkcd:hot pink'], mask=~best_asks, cbar=False, ax=ax)

plt.tight_layout()

# This section plots the historical order book (no depth available).
if PLOT_HISTORICAL:
  fig = plt.figure(figsize=(12,9))

  df_hist = df_hist.between_time('9:30', '16:00')
  #df_hist = df_hist.between_time('10:00', '10:05')
  df_hist = df_hist.resample('1S').last().ffill()

  df = prepare_histogram(df_hist)
  df.index = df.index.time

  # There's no order book depth anyway, so make all bids the same volume
  # and all asks the same volume, so they're easy to see.
  df[df > 0] = 1
  df[df < 0] = -1

  ax = sns.heatmap(df, cmap=sns.color_palette("coolwarm", 7), cbar_kws={'label': 'Shares Available'}, center=0)
  ax.set(xlabel='Quoted Price', ylabel='Quote Time')

  plt.tight_layout()

# Show all the plots.
plt.show()

