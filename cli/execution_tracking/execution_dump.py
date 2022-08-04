import pandas as pd
import sys

# Auto-detect terminal width.
pd.options.display.width = None
pd.options.display.max_rows = 500000
pd.options.display.max_colwidth = 200

if len(sys.argv) < 2:
  print ("Usage: python dump.py <DataFrame file> [List of Event Types]")
  sys.exit()

file = sys.argv[1]

df = pd.read_pickle(file, compression='bz2')

df2 = df.reset_index(drop=True)

# get starting cash and ending cash rows
ending_cash = df2[df2['EventType'] == "ENDING_CASH"]
starting_cash = df2[df2['EventType'] == "STARTING_CASH"]

df2 = df2.tail(10)

# drop marked to market rows
df2 = df2[df2['EventType'] != 'MARKED_TO_MARKET']
df2 = df2[df2['EventType'] != 'MARK_TO_MARKET']

df2 = df2.append([starting_cash, ending_cash], ignore_index=True)


print(df2)


# TODO: export to file?
