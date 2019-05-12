import ast
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

from joblib import Memory

# Auto-detect terminal width.
pd.options.display.width = None
pd.options.display.max_rows = 1000
pd.options.display.max_colwidth = 200

# Initialize a persistent memcache.
mem_hist = Memory(cachedir='./.cached_plot_hist', verbose=0)
mem_sim = Memory(cachedir='./.cached_plot_sim', verbose=0)


PRINT_BASELINE = True
PRINT_DELTA_ONLY = True

BETWEEN_START = pd.to_datetime('09:30').time()
BETWEEN_END = pd.to_datetime('09:30:00.000001').time()

# Linewidth for plots.
LW = 2

# Used to read and cache simulated quotes.
# Doesn't actually pay attention to symbols yet.
#@mem_sim.cache
def read_simulated_quotes (file, symbol):
  print ("Simulated quotes were not cached.  This will take a minute.")
  df = pd.read_pickle(file, compression='bz2')
  df['Timestamp'] = df.index

  # Keep only the last bid and last ask event at each timestamp.
  df = df.drop_duplicates(subset=['Timestamp','EventType'], keep='last')

  del df['Timestamp']

  df_bid = df[df['EventType'] == 'BEST_BID'].copy()
  df_ask = df[df['EventType'] == 'BEST_ASK'].copy()

  if len(df) <= 0:
    print ("There appear to be no simulated quotes.")
    sys.exit()

  df_bid['BEST_BID'] = [b for s,b,bv in df_bid['Event'].str.split(',')]
  df_bid['BEST_BID_VOL'] = [bv for s,b,bv in df_bid['Event'].str.split(',')]
  df_ask['BEST_ASK'] = [a for s,a,av in df_ask['Event'].str.split(',')]
  df_ask['BEST_ASK_VOL'] = [av for s,a,av in df_ask['Event'].str.split(',')]

  df_bid['BEST_BID'] = df_bid['BEST_BID'].str.replace('$','').astype('float64')
  df_ask['BEST_ASK'] = df_ask['BEST_ASK'].str.replace('$','').astype('float64')

  df_bid['BEST_BID_VOL'] = df_bid['BEST_BID_VOL'].astype('float64')
  df_ask['BEST_ASK_VOL'] = df_ask['BEST_ASK_VOL'].astype('float64')

  df = df_bid.join(df_ask, how='outer', lsuffix='.bid', rsuffix='.ask')
  df['BEST_BID'] = df['BEST_BID'].ffill().bfill()
  df['BEST_ASK'] = df['BEST_ASK'].ffill().bfill()
  df['BEST_BID_VOL'] = df['BEST_BID_VOL'].ffill().bfill()
  df['BEST_ASK_VOL'] = df['BEST_ASK_VOL'].ffill().bfill()

  df['MIDPOINT'] = (df['BEST_BID'] + df['BEST_ASK']) / 2.0

  return df



# Main program starts here.

if len(sys.argv) < 3:
  print ("Usage: python midpoint_plot.py <Ticker symbol> <Simulator DataFrame file>")
  sys.exit()

# TODO: only really works for one symbol right now.

symbol = sys.argv[1]
sim_file = sys.argv[2]

print ("Visualizing simulated {} from {}".format(symbol, sim_file))
df_sim = read_simulated_quotes(sim_file, symbol)

if PRINT_BASELINE:
  baseline_file = os.path.join(os.path.dirname(sim_file) + '_baseline', os.path.basename(sim_file))
  print (baseline_file)
  df_baseline = read_simulated_quotes(baseline_file, symbol)

plt.rcParams.update({'font.size': 12})



# Use to restrict time to plot.
df_sim = df_sim.between_time(BETWEEN_START, BETWEEN_END)

if PRINT_BASELINE:
  df_baseline = df_baseline.between_time(BETWEEN_START, BETWEEN_END)

fig,ax = plt.subplots(figsize=(12,9), nrows=1, ncols=1)
axes = [ax]

# For smoothing...
#hist_window = 100
#sim_window = 100

hist_window = 1
sim_window = 1

if PRINT_BASELINE:
  # For nanosecond experiments, turn it into int index.  Pandas gets weird if all
  # the times vary only by a few nanoseconds.
  rng = pd.date_range(start=df_sim.index[0], end=df_sim.index[-1], freq='1N')

  df_baseline = df_baseline[~df_baseline.index.duplicated(keep='last')]
  df_baseline = df_baseline.reindex(rng,method='ffill')
  df_baseline = df_baseline.reset_index(drop=True)

  df_sim = df_sim[~df_sim.index.duplicated(keep='last')]
  df_sim = df_sim.reindex(rng,method='ffill')
  df_sim = df_sim.reset_index(drop=True)

  # Print both separately.
  if PRINT_DELTA_ONLY:
    # Print the difference as a single series.
    df_diff = df_sim['MIDPOINT'] - df_baseline['MIDPOINT']

    # Smoothing.
    df_diff = df_diff.rolling(window=10).mean()

    df_diff.plot(color='C0', grid=True, linewidth=LW, ax=axes[0])

    axes[0].legend(['Bid-ask Midpoint Delta'])
  else:
    df_baseline['MIDPOINT'].plot(color='C0', grid=True, linewidth=LW, ax=axes[0])
    df_sim['MIDPOINT'].plot(color='C1', grid=True, linewidth=LW, alpha=0.9, ax=axes[0])

    axes[0].legend(['Baseline', 'With Impact'])

else:
  #df_sim['PRICE'] = df_sim['PRICE'].rolling(window=sim_window).mean()

  # For nanosecond experiments, turn it into int index.  Pandas gets weird if all
  # the times vary only by a few nanoseconds.
  rng = pd.date_range(start=df_sim.index[0], end=df_sim.index[-1], freq='1N')
  df_sim = df_sim[~df_sim.index.duplicated(keep='last')]
  df_sim = df_sim.reindex(rng,method='ffill')
  df_sim = df_sim.reset_index(drop=True)
  df_sim['MIDPOINT'].plot(color='C1', grid=True, linewidth=LW, alpha=0.9, ax=axes[0])
  axes[0].legend(['Simulated'])


plt.suptitle('Bid-Ask Midpoint: {}'.format(symbol))

axes[0].set_ylabel('Quote Price')
axes[0].set_xlabel('Quote Time')

#plt.savefig('background_{}.png'.format(b))

plt.show()

