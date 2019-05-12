import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import sys

from joblib import Memory

# Auto-detect terminal width.
pd.options.display.width = None
pd.options.display.max_rows = 1000
pd.options.display.max_colwidth = 200

# Initialize a persistent memcache.
mem_sim = Memory(cachedir='./.cached_plot_sim', verbose=0)


# Used to read and cache simulated quotes (best bid/ask).
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

if len(sys.argv) < 2:
  print ("Usage: python ticker_plot.py <Ticker symbol> <Simulator DataFrame file>")
  sys.exit()

# TODO: only really works for one symbol right now.

symbol = sys.argv[1]
sim_file = sys.argv[2]

print ("Visualizing {} from {}".format(symbol, sim_file))

plt.rcParams.update({'font.size': 12})

df_sim = read_simulated_quotes(sim_file, symbol)

fig,axes = plt.subplots(figsize=(12,9), nrows=2, ncols=1)

# Crop figures to desired times and price scales.
#df_hist = df_hist.between_time('9:46', '13:30')
#df_sim = df_sim.between_time('10:00:00', '10:00:30')

# For nanosecond experiments, turn it into int index.  Pandas gets weird if all
# the times vary only by a few nanoseconds.
df_sim = df_sim.reset_index(drop=True)

ax = df_sim['BEST_BID'].plot(color='C0', grid=True, linewidth=1, ax=axes[0])
df_sim['BEST_ASK'].plot(color='C1', grid=True, linewidth=1, ax=axes[0])
#df_sim['MIDPOINT'].plot(color='C2', grid=True, linewidth=1, ax=axes[0])

df_sim['BEST_BID_VOL'].plot(color='C3', linewidth=1, ax=axes[1])
df_sim['BEST_ASK_VOL'].plot(color='C4', linewidth=1, ax=axes[1])

axes[0].legend(['Best Bid', 'Best Ask', 'Midpoint'])
axes[1].legend(['Best Bid Vol', 'Best Ask Vol'])

plt.suptitle('Best Bid/Ask: {}'.format(symbol))

axes[0].set_ylabel('Quote Price')
axes[1].set_xlabel('Quote Time')
axes[1].set_ylabel('Quote Volume')

axes[0].get_xaxis().set_visible(False)

#plt.savefig('background_{}.png'.format(b))

plt.show()

