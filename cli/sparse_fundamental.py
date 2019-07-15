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

BETWEEN_START = pd.to_datetime('09:30').time()
BETWEEN_END = pd.to_datetime('16:00:00').time()

# Linewidth for plots.
LW = 2

# Main program starts here.

if len(sys.argv) < 2:
  print ("Usage: python sparse_fundamental.py <Simulator Fundamental file>")
  sys.exit()

# TODO: only really works for one symbol right now.
sim_file = sys.argv[1]

m = re.search(r'fundamental_(.+?)\.', sim_file)

if not m:
  print ("Usage: python sparse_fundamental.py <Simulator Fundamental file>")
  print ("{} does not appear to be a fundamental value log.".format(sim_file))
  print ()
  sys.exit()

symbol = m.group(1)

print ("Visualizing simulated fundamental from {}".format(sim_file))
df_sim = pd.read_pickle(sim_file, compression='bz2')

plt.rcParams.update({'font.size': 12})

print (df_sim.head())

# Use to restrict time to plot.
#df_sim = df_sim.between_time(BETWEEN_START, BETWEEN_END)

fig,ax = plt.subplots(figsize=(12,9), nrows=1, ncols=1)
axes = [ax]

# For smoothing...
#hist_window = 100
#sim_window = 100

hist_window = 1
sim_window = 1

#df_sim['PRICE'] = df_sim['PRICE'].rolling(window=sim_window).mean()

df_sim['FundamentalValue'].plot(color='C1', grid=True, linewidth=LW, alpha=0.9, ax=axes[0])
axes[0].legend(['Simulated'])


plt.suptitle('Fundamental Value: {}'.format(symbol))

axes[0].set_ylabel('Fundamental Value')
axes[0].set_xlabel('Fundamental Time')

#plt.savefig('background_{}.png'.format(b))

plt.show()

