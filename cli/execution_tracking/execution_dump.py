import pandas as pd
import sys
import os
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

df2 = df2.tail(12)

# drop marked to market rows
df2 = df2[df2['EventType'] != 'MARKED_TO_MARKET']
df2 = df2[df2['EventType'] != 'MARK_TO_MARKET']

df2 = df2.append([starting_cash, ending_cash], ignore_index=True)


print(df2)

# converting to csv
path = "CSVs\\Dumps\\" + file.split("\\")[1] + '\\'
if not os.path.isdir(path):
   os.makedirs(path)

# drop log directory from file
file = '\\' + file.split("\\")[2]
file = file.split('.bz2')[0]
df2.to_csv(path + file + '_DUMP.csv', index=False)