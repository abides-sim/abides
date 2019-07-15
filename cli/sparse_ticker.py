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

# We could use some good argparse parameters here instead of
# a bunch of constants to fiddle with.
PRINT_HISTORICAL = False
PRINT_BASELINE = False
PRINT_VOLUME = False

BETWEEN_START = pd.to_datetime('09:30').time()
#BETWEEN_END = pd.to_datetime('09:30:00.000001').time()
BETWEEN_END = pd.to_datetime('16:00:00').time()

# Linewidth for plots.
LW = 2

# Used to read and cache real historical trades.
#@mem_hist.cache
def read_historical_trades (file, symbol):
  print ("Historical trades were not cached.  This will take a minute.")
  df = pd.read_pickle(file, compression='bz2')

  df = df.loc[symbol]
  df = df.between_time('9:30', '16:00')

  return df


# Used to read and cache simulated trades.
# Doesn't actually pay attention to symbols yet.
#@mem_sim.cache
def read_simulated_trades (file, symbol):
  print ("Simulated trades were not cached.  This will take a minute.")
  df = pd.read_pickle(file, compression='bz2')
  df = df[df['EventType'] == 'LAST_TRADE']

  if len(df) <= 0:
    print ("There appear to be no simulated trades.")
    sys.exit()

  df['PRICE'] = [y for x,y in df['Event'].str.split(',')]
  df['SIZE'] = [x for x,y in df['Event'].str.split(',')]

  df['PRICE'] = df['PRICE'].str.replace('$','').astype('float64')
  df['SIZE'] = df['SIZE'].astype('float64')

  return df


# Main program starts here.

if len(sys.argv) < 3:
  print ("Usage: python ticker_plot.py <Ticker symbol> <Simulator DataFrame file> [agent trade log]")
  sys.exit()

# TODO: only really works for one symbol right now.

symbol = sys.argv[1]
sim_file = sys.argv[2]

agent_log = None
if len(sys.argv) >= 4: agent_log = sys.argv[3]

print ("Visualizing simulated {} from {}".format(symbol, sim_file))

df_sim = read_simulated_trades(sim_file, symbol)

print (df_sim)

if PRINT_BASELINE:
  baseline_file = os.path.join(os.path.dirname(sim_file) + '_baseline', os.path.basename(sim_file))
  print (baseline_file)
  df_baseline = read_simulated_trades(baseline_file, symbol)

# Take the date from the first index and use that to pick the correct historical date for comparison.
if PRINT_HISTORICAL: 
  hist_date = pd.to_datetime(df_sim.index[0])
  hist_year = hist_date.strftime('%Y')
  hist_date = hist_date.strftime('%Y%m%d')
  hist_file = "/nethome/cb107/emh/data/trades/trades_{}/ct{}_{}.bgz".format(hist_year, 'm' if int(hist_year) > 2014 else '', hist_date)

  print ("Visualizing historical {} from {}".format(symbol, hist_file)) 
  df_hist = read_historical_trades(hist_file, symbol)

plt.rcParams.update({'font.size': 12})



# Use to restrict time to plot.
df_sim = df_sim.between_time(BETWEEN_START, BETWEEN_END)
print ("Total simulated volume:", df_sim['SIZE'].sum())

if PRINT_BASELINE:
  df_baseline = df_baseline.between_time(BETWEEN_START, BETWEEN_END)
  print ("Total baseline volume:", df_baseline['SIZE'].sum())

if PRINT_VOLUME:
  fig,axes = plt.subplots(figsize=(12,9), nrows=2, ncols=1)
else:
  fig,ax = plt.subplots(figsize=(12,9), nrows=1, ncols=1)
  axes = [ax]

# Crop figures to desired times and price scales.
#df_hist = df_hist.between_time('9:46', '13:30')

# For smoothing...
#hist_window = 100
#sim_window = 100

hist_window = 1
sim_window = 1

if PRINT_HISTORICAL:
  df_hist = df_hist.between_time(BETWEEN_START, BETWEEN_END)
  print ("Total historical volume:", df_hist['SIZE'].sum())

  df_hist['PRICE'] = df_hist['PRICE'].rolling(window=hist_window).mean()
  df_sim['PRICE'] = df_sim['PRICE'].rolling(window=sim_window).mean()

  df_hist['PRICE'].plot(color='C0', grid=True, linewidth=LW, ax=axes[0])
  df_sim['PRICE'].plot(color='C1', grid=True, linewidth=LW, alpha=0.9, ax=axes[0])
  axes[0].legend(['Historical', 'Simulated'])

  if PRINT_VOLUME:
    df_hist['SIZE'].plot(color='C0', linewidth=LW, ax=axes[1])
    df_sim['SIZE'].plot(color='C1', linewidth=LW, alpha=0.9, ax=axes[1])
    axes[1].legend(['Historical Vol', 'Simulated Vol'])
elif PRINT_BASELINE:
  # For nanosecond experiments, turn it into int index.  Pandas gets weird if all
  # the times vary only by a few nanoseconds.
  rng = pd.date_range(start=df_sim.index[0], end=df_sim.index[-1], freq='1N')

  df_baseline = df_baseline[~df_baseline.index.duplicated(keep='last')]
  df_baseline = df_baseline.reindex(rng,method='ffill')
  df_baseline = df_baseline.reset_index(drop=True)

  df_sim = df_sim[~df_sim.index.duplicated(keep='last')]
  df_sim = df_sim.reindex(rng,method='ffill')
  df_sim = df_sim.reset_index(drop=True)

  df_baseline['PRICE'].plot(color='C0', grid=True, linewidth=LW, ax=axes[0])
  df_sim['PRICE'].plot(color='C1', grid=True, linewidth=LW, alpha=0.9, ax=axes[0])

  axes[0].legend(['Baseline', 'With Impact'])

else:
  #df_sim['PRICE'] = df_sim['PRICE'].rolling(window=sim_window).mean()

  # For nanosecond experiments, turn it into int index.  Pandas gets weird if all
  # the times vary only by a few nanoseconds.

  # Frequency needs to be a CLI arg.
  #rng = pd.date_range(start=df_sim.index[0], end=df_sim.index[-1], freq='1N')
  #rng = pd.date_range(start=df_sim.index[0], end=df_sim.index[-1], freq='1S')

  # Resample obviates this need.
  #df_sim = df_sim[~df_sim.index.duplicated(keep='last')]
  #df_sim = df_sim.resample('1S').mean()

  # When printing volume, we'll need to split series, because price can be mean
  # (or avg share price) but volume should be sum.

  #df_sim = df_sim.reindex(rng,method='ffill')
  #df_sim = df_sim.reset_index(drop=True)
  df_sim['PRICE'].plot(color='C1', grid=True, linewidth=LW, alpha=0.9, ax=axes[0])
  axes[0].legend(['Simulated'])

  if PRINT_VOLUME:
    df_sim['SIZE'].plot(color='C1', linewidth=LW, alpha=0.9, ax=axes[1])
    axes[1].legend(['Simulated Vol'])

# Superimpose a particular trading agent's trade decisions on top of the ticker
# plot to make it easy to visually see if it is making sensible choices.
if agent_log:
  df_agent = pd.read_pickle(agent_log, compression='bz2')
  df_agent = df_agent.between_time(BETWEEN_START, BETWEEN_END)
  df_agent = df_agent[df_agent.EventType == 'HOLDINGS_UPDATED']

  first = True

  for idx in df_agent.index:
    event = df_agent.loc[idx,'Event']
    if symbol in event:
      shares = event[symbol]
      if shares > 0:
        print ("LONG at {}".format(idx))
        axes[0].axvline(x=idx, linewidth=LW, color='g')
      elif shares < 0:
        print ("SHORT at {}".format(idx))
        axes[0].axvline(x=idx, linewidth=LW, color='r')
      else:
        print ("EXIT at {}".format(idx))
        axes[0].axvline(x=idx, linewidth=LW, color='k')
    else:
      print ("EXIT at {}".format(idx))
      axes[0].axvline(x=idx, linewidth=LW, color='k')

plt.suptitle('Execution Price/Volume: {}'.format(symbol))

axes[0].set_ylabel('Executed Price')

if PRINT_VOLUME:
  axes[1].set_xlabel('Execution Time')
  axes[1].set_ylabel('Executed Volume')
  axes[0].get_xaxis().set_visible(False)
else:
  axes[0].set_xlabel('Execution Time')

#plt.savefig('background_{}.png'.format(b))

plt.show()

