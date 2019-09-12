# This file calculates the intraday index in a similar method to the Dow,
# after the conclusion of a simulation run. It takes the average price
# of the symbols in the index at each timestep, and graphs it. The
# price refers to the mid.
# Also graphs the mid of the underlying symbols

import ast
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import numpy as np

from joblib import Memory

# Auto-detect terminal width.
pd.options.display.width = None
pd.options.display.max_rows = 1000
pd.options.display.max_colwidth = 200

# Initialize a persistent memcache.
mem_hist = Memory(cachedir='./.cached_plot_hist', verbose=0)
mem_sim = Memory(cachedir='./.cached_plot_sim', verbose=0)


PRINT_BASELINE = False
PRINT_DELTA_ONLY = False

BETWEEN_START = pd.to_datetime('09:30').time()
BETWEEN_END = pd.to_datetime('09:30:00.000001').time()

# Linewidth for plots.
LW = 2

# Used to read and cache simulated quotes.
# Doesn't actually pay attention to symbols yet.
#@mem_sim.cache
def read_simulated_quotes (file):
  print ("Simulated quotes were not cached.  This will take a minute.")
  df = pd.read_pickle(file, compression='bz2')
  df['Timestamp'] = df.index

  df_bid = df[df['EventType'] == 'BEST_BID'].copy()
  df_ask = df[df['EventType'] == 'BEST_ASK'].copy()

  if len(df) <= 0:
    print ("There appear to be no simulated quotes.")
    sys.exit()

  df_bid['SYM'] = [s for s,b,bv in df_bid['Event'].str.split(',')]
  df_bid['BEST_BID'] = [b for s,b,bv in df_bid['Event'].str.split(',')]
  df_bid['BEST_BID_VOL'] = [bv for s,b,bv in df_bid['Event'].str.split(',')]
  df_ask['SYM'] = [s for s,a,av in df_ask['Event'].str.split(',')]
  df_ask['BEST_ASK'] = [a for s,a,av in df_ask['Event'].str.split(',')]
  df_ask['BEST_ASK_VOL'] = [av for s,a,av in df_ask['Event'].str.split(',')]

  df_bid['BEST_BID'] = df_bid['BEST_BID'].str.replace('$','').astype('float64')
  df_ask['BEST_ASK'] = df_ask['BEST_ASK'].str.replace('$','').astype('float64')

  df_bid['BEST_BID_VOL'] = df_bid['BEST_BID_VOL'].astype('float64')
  df_ask['BEST_ASK_VOL'] = df_ask['BEST_ASK_VOL'].astype('float64')
    

  # Keep only the last bid and last ask event at each timestamp.
  df_bid = df_bid.drop_duplicates(subset=['Timestamp', 'SYM'], keep='last')
  df_ask = df_ask.drop_duplicates(subset=['Timestamp', 'SYM'], keep='last')

  #df = df_bid.join(df_ask, how='outer', lsuffix='.bid', rsuffix='.ask')
    
  # THIS ISN'T TRUE, YOU CAN'T GET THE MID FROM FUTURE ORDERS!!!
  #df['BEST_BID'] = df['BEST_BID'].ffill().bfill()
  #df['BEST_ASK'] = df['BEST_ASK'].ffill().bfill()
  #df['BEST_BID_VOL'] = df['BEST_BID_VOL'].ffill().bfill()
  #df['BEST_ASK_VOL'] = df['BEST_ASK_VOL'].ffill().bfill()
  df = pd.merge(df_bid, df_ask,  how='left', left_on=['Timestamp','SYM'], right_on = ['Timestamp','SYM'])

  df['MIDPOINT'] = (df['BEST_BID'] + df['BEST_ASK']) / 2.0

  #ts = df['Timestamp.bid']
  ts = df['Timestamp']
  #print(df)
  df['INTRADAY_INDEX'] = 0
  #df['Timestamp.bid'] = df.index.to_series()
  #df['Timestamp'] = df.index.to_series()
  #df['SYM.bid'] = df['SYM.bid'].fillna(df['SYM.ask'])
  #symbols = df['SYM.bid'].unique()
  symbols = df['SYM'].unique()
  #print(df)
  for i,x in enumerate(symbols):
      #df_one_sym = df[df['SYM.bid']==x]
      df_one_sym = df[df['SYM']==x]
      #df_one_sym = df_one_sym[['Timestamp.bid','MIDPOINT','BEST_BID','BEST_ASK']]
      df_one_sym = df_one_sym[['Timestamp','MIDPOINT','BEST_BID','BEST_ASK']]
      #md = pd.merge_asof(ts, df_one_sym, on='Timestamp.bid')
      #print(ts)
      #print(df_one_sym)
      md = pd.merge_asof(ts, df_one_sym, on='Timestamp')
      md = md.set_index(df.index)
      df['MIDPOINT.' + x] = md['MIDPOINT']
      df['BID.' + x] = md['BEST_BID']
      df['ASK.' + x] = md['BEST_ASK']
      if x != 'ETF':
        df['INTRADAY_INDEX'] = df['INTRADAY_INDEX'] + md['MIDPOINT']
  df['MSE'] = (df['INTRADAY_INDEX'] - df['MIDPOINT.ETF'])**2

  #del df['Timestamp.bid']
  #del df['Timestamp.ask']
  df = df.set_index(df['Timestamp'])
  del df['Timestamp']
    
  return df



# Main program starts here.

if len(sys.argv) < 2:
  print ("Usage: python midpoint_plot.py <Ticker symbol> <Simulator DataFrame file>")
  sys.exit()

# TODO: only really works for one symbol right now.

#symbols = sys.argv[1]
sim_file = sys.argv[1]

print ("Visualizing simulated {} from {}".format(12,sim_file))
df_sim = read_simulated_quotes(sim_file)

if PRINT_BASELINE:
  baseline_file = os.path.join(os.path.dirname(sim_file) + '_baseline', os.path.basename(sim_file))
  print (baseline_file)
  df_baseline = read_simulated_quotes(baseline_file)

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
  df_time = df_sim.copy()
  df_sim = df_sim.reset_index(drop=True)
    
  #symbols = df_sim['SYM.bid'].unique()
  symbols = df_sim['SYM'].unique()
  for i,x in enumerate(symbols):
      #df_sim[df_sim['SYM.bid']==x]['MIDPOINT.' + x].plot(color='C1', grid=True, linewidth=LW, alpha=0.9, ax=axes[0])
      df_sim['MIDPOINT.' + x].plot(color='C1', grid=True, linewidth=LW, alpha=0.9, ax=axes[0])
      #df_sim['BID.' + x].plot(color='C2', grid=True, linewidth=LW, alpha=0.9, ax=axes[0])
      #df_sim['ASK.' + x].plot(color='C3', grid=True, linewidth=LW, alpha=0.9, ax=axes[0])
      if x != 'ETF':
        axes[0].legend(['Simulated'])

      plt.suptitle('Bid-Ask Midpoint: {}'.format(x))

      axes[0].set_ylabel('Quote Price')
      axes[0].set_xlabel('Quote Time')

      plt.savefig('graphs/background_' + str(x) + '_{}.png'.format('png'))
      plt.cla()
    
      if x == 'ETF':
        df_sim['MIDPOINT.' + x].plot(color='C1', grid=True, linewidth=LW, alpha=0.9, ax=axes[0], label = 'ETF Mid')
        i = np.argwhere(symbols=='ETF')
        symbols_portfolio = np.delete(symbols, i)
        df_sim['INTRADAY_INDEX'].plot(color='C4', grid=True, linewidth=LW, alpha=0.9, ax=axes[0], label = 'Index')
        #axes[0].legend(['Simulated'])
        plt.suptitle('65 ZI, 0 ETF Arb, gamma = 500: {}'.format(symbols_portfolio))
        
        axes[0].set_ylabel('Quote Price')
        axes[0].set_xlabel('Quote Time')
        axes[0].legend()
        ymin = 249000
        ymax = 251000
        axes[0].set_ylim([ymin,ymax])

        plt.savefig('graphs/index_vs_etf_ten_arb_gamma_500'.format('png'))
        plt.cla()
    
  i = np.argwhere(symbols=='ETF')
  symbols_portfolio = np.delete(symbols, i)
  df_sim['INTRADAY_INDEX'].plot(color='C4', grid=True, linewidth=LW, alpha=0.9, ax=axes[0])
  axes[0].legend(['Simulated'])
  plt.suptitle('Intraday Index: {}'.format(symbols_portfolio))

  plt.savefig('graphs/intraday_index_' + str(symbols_portfolio) + '_{}.png'.format('png'))
  plt.cla()

  df_sim['MSE'].plot(color='C5', grid=True, linewidth=LW, alpha=0.9, ax=axes[0])
  #axes[0].legend(['Simulated'])
  plt.suptitle('65 ZI, 10 ETF Arb, gamma = 500')
  axes[0].set_ylabel('Mean Squared Error')
  axes[0].set_xlabel('Quote Time')
  ymin = -1000
  ymax = 700000
  axes[0].set_ylim([ymin,ymax])

  plt.savefig('graphs/mse_index_etf_ten_arb_gamma_500'.format('png'))
  plt.close()
  #df_time.to_csv('test.csv')

#plt.show()

