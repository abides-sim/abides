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
#mem_sim = Memory(cachedir='./.cached_plot_sim', verbose=0)


# Linewidth for plots.
LW = 2

# Rolling window for smoothing.
#SIM_WINDOW = 250
SIM_WINDOW = 1


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
  print ("Usage: python event_midpoint.py <Ticker symbol> <Simulator DataFrame file(s)>")
  sys.exit()

# TODO: only really works for one symbol right now.

symbol = sys.argv[1]
sim_files = sys.argv[2:]

fig,ax = plt.subplots(figsize=(12,9), nrows=1, ncols=1)



# Plot each impact simulation with the baseline subtracted (i.e. residual effect).
i = 1
legend = []
#legend = ['baseline']

# Events is now a dictionary of event lists (key == greed parameter).
events = {}

first_date = None
impact_time = 200

for sim_file in sim_files:

  # Skip baseline files.
  if 'baseline' in sim_file: continue

  if 'greed' in os.path.dirname(sim_file):
    # Group plots by greed parameter.
    m = re.search("greed(\d\d\d)_", sim_file)
    g = m.group(1)
  else:
    g = 'greed'

  baseline_file = os.path.join(os.path.dirname(sim_file) + '_baseline', os.path.basename(sim_file))
  print ("Visualizing simulation baseline from {}".format(baseline_file))

  df_baseline = read_simulated_quotes(baseline_file, symbol)

  # Read the event file.
  print ("Visualizing simulated {} from {}".format(symbol, sim_file))

  df_sim = read_simulated_quotes(sim_file, symbol)

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

  # Absolute price difference.
  #s = df_sim['MIDPOINT'] - df_baseline['MIDPOINT']

  # Relative price difference.
  s = (df_sim['MIDPOINT'] / df_baseline['MIDPOINT']) - 1.0

  s = s.rolling(window=SIM_WINDOW).mean()
  s.name = sim_file

  if g not in events: events[g] = []

  events[g].append(s.copy())

  i += 1


# Now have a list of series (each an event) that are time-aligned.  BUT the data is
# still aperiodic, so they may not have the exact same index timestamps.

legend = []

for g in events:
  df = pd.DataFrame()
  legend.append("greed = " + str(g))
  
  for s in events[g]:
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
  m.plot(grid=True, linewidth=LW, ax=ax, fontsize=12, label="Relative mean mid-price")

  # Fill std region?
  #ax.fill_between(m.index, m-s, m+s, alpha=0.2)
  

# Do the rest a single time for the whole plot.

# If we need a vertical "time of event" line...
ax.axvline(x=200, color='0.5', linestyle='--', linewidth=2, label="Order placement time")
  
# Absolute or relative time labels...
ax.set_xticklabels(['0','10000','20000','30000','40000','50000','60000','70000'])
#ax.set_xticklabels(['T-30', 'T-20', 'T-10', 'T', 'T+10', 'T+20', 'T+30'])

ax.legend(legend)
#ax.legend()
  
# Force y axis limits to make multiple plots line up exactly...
#ax.set_ylim(-0.0065,0.0010)
#ax.set_ylim(-0.0010,0.0065)

# If an in-figure super title is required...
#plt.suptitle('Impact Event Study: {}'.format(symbol))
  
ax.set_xlabel('Relative Time (ms)', fontsize=12, fontweight='bold')
ax.set_ylabel('Baseline-Relative Price', fontsize=12, fontweight='bold')
  
#plt.savefig('IABS_SELL_100_multi_size.png')
#plt.savefig('abides_impact_sell.png')
#plt.savefig('abides_multi_buy.png')
#plt.savefig('abides_multi_sell.png')
  
plt.show()
  
