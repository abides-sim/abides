import ast
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
import sys

from joblib import Memory

# Auto-detect terminal width.
pd.options.display.width = None
pd.options.display.max_rows = 1000
pd.options.display.max_colwidth = 200

# Initialize a persistent memcache.
mem_sim = Memory(cachedir='./.cached_plot_sim', verbose=0)


# Linewidth for plots.
LW = 2

# Rolling window for smoothing.
#SIM_WINDOW = 250
SIM_WINDOW = 1


# Used to read and cache simulated trades.
# Doesn't actually pay attention to symbols yet.
#@mem_sim.cache
def read_simulated_trades (file, symbol):
  #print ("Simulated trades were not cached.  This will take a minute.")
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
  print ("Usage: python mean_std_event.py <Ticker symbol> <Simulator DataFrame file(s)>")
  sys.exit()

# TODO: only really works for one symbol right now.

symbol = sys.argv[1]
sim_files = sys.argv[2:]

fig,ax = plt.subplots(figsize=(12,9), nrows=1, ncols=1)



# Plot each impact simulation with the baseline subtracted (i.e. residual effect).
i = 1
legend = []
#legend = ['baseline']

events = []

first_date = None
impact_time = 200

for sim_file in sim_files:

  # Skip baseline files.
  if 'baseline' in sim_file: continue

  baseline_file = os.path.join(os.path.dirname(sim_file) + '_baseline', os.path.basename(sim_file))
  print ("Visualizing simulation baseline from {}".format(baseline_file))

  df_baseline = read_simulated_trades(baseline_file, symbol)

  # Read the event file.
  print ("Visualizing simulated {} from {}".format(symbol, sim_file))

  df_sim = read_simulated_trades(sim_file, symbol)

  plt.rcParams.update({'font.size': 12})

  # Given nanosecond ("time step") data, we can just force everything to
  # fill out an integer index of nanoseconds.
  rng = pd.date_range(start=df_sim.index[0], end=df_sim.index[-1], freq='1N')

  df_baseline = df_baseline[~df_baseline.index.duplicated(keep='last')]
  df_baseline = df_baseline.reindex(rng,method='ffill')
  df_baseline = df_baseline.reset_index(drop=True)

  df_sim = df_sim[~df_sim.index.duplicated(keep='last')]
  df_sim = df_sim.reindex(rng,method='ffill')
  df_sim = df_sim.reset_index(drop=True)

  s = df_sim['PRICE'] - df_baseline['PRICE']
  s = s.rolling(window=SIM_WINDOW).mean()

  s.name = sim_file
  events.append(s.copy())

  i += 1


# Now have a list of series (each an event) that are time-aligned.
df = pd.DataFrame()

for s in events:
  print ("Joining {}".format(s.name))
  df = df.join(s, how='outer')

df.dropna(how='all', inplace=True)
df = df.ffill().bfill()

# Smooth after combining means at each instant-of-trade.
#df.mean(axis=1).rolling(window=250).mean().plot(grid=True, linewidth=LW, ax=ax)

# No additional smoothing.
m = df.mean(axis=1)
s = df.std(axis=1)

# Plot mean and std.
m.plot(grid=True, linewidth=LW, ax=ax)

# Shade the stdev region?
ax.fill_between(m.index, m-s, m+s, alpha=0.2)

# Override prettier axis ticks...
#ax.set_xticklabels(['T-30', 'T-20', 'T-10', 'T', 'T+10', 'T+20', 'T+30'])

# Force y axis limits to match some other plot.
#ax.set_ylim(-0.1, 0.5)

# Set a super title if required.
plt.suptitle('Impact Event Study: {}'.format(symbol))

ax.set_xlabel('Relative Time')
ax.set_ylabel('Baseline-Relative Price')

#plt.savefig('background_{}.png'.format(b))

plt.show()


