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

if len(sys.argv) > 2:
  events = sys.argv[2:]
  event = "|".join(events)
  df = df[df['EventType'].str.contains(event)]

print(df)

