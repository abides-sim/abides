import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import sys

# Auto-detect terminal width.
pd.options.display.width = None
pd.options.display.max_rows = 1000
pd.options.display.max_colwidth = 200

BETWEEN_START = pd.to_datetime('09:30').time()
BETWEEN_END = pd.to_datetime('16:00:00').time()

# Linewidth for plots.
LW = 2

# Main program starts here.

if len(sys.argv) < 2:
  print ("Usage: python sparse_fundamental.py <Simulator Fundamental file>")
  sys.exit()

sim_file = sys.argv[1]

df_sim = pd.read_pickle(sim_file, compression='bz2')

#print(df_sim)

df_bid = df_sim.loc[df_sim['EventType'] == 'BEST_BID']
df_bid = df_bid.assign( BID_PRICE = lambda x: x['Event'].str.split(',').str[1].astype('float64'))

df_ask = df_sim.loc[df_sim['EventType'] == 'BEST_ASK']
df_ask = df_ask.assign( ASK_PRICE = lambda x: x['Event'].str.split(',').str[1].astype('float64'))

df_trade = df_sim.loc[df_sim['EventType'] == 'LAST_TRADE']
df_trade = df_trade.assign( TRADE_PRICE = lambda x: x['Event'].str.replace("$", " ").str.split(',').str[1].astype('float64'))
df_trade = df_trade.assign( TRADE_SIZE = lambda x: x['Event'].str.replace("$", " ").str.split(',').str[0].astype('float64'))

#print(df_trade)

plt.rcParams.update({'font.size': 12})

fig,ax = plt.subplots(figsize=(12,9), nrows=1, ncols=1)
axes = [ax]

df_bid['BID_PRICE'].plot(color='C1', grid=True, linewidth=LW, alpha=0.9, ax=axes[0])
df_ask['ASK_PRICE'].plot(color='C2', grid=True, linewidth=LW, alpha=0.9, ax=axes[0])
df_trade['TRADE_PRICE'].plot(color='C3', marker = 'o', markersize = 10, linewidth =0, ax=axes[0])

axes[0].legend(['BID_PRICE', 'ASK_PRICE', "TRADE_PRICE"])
ax.set_title("Exchange Agent")

axes[0].set_ylabel('Price')
axes[0].set_xlabel('Time')

plt.savefig('value_noise_MM_2'
            '.png')
