import os
import pandas as pd
import sys

# Auto-detect terminal width.
pd.options.display.width = None
pd.options.display.max_rows = 500000
pd.options.display.max_colwidth = 200

if len(sys.argv) < 2:
  print ("Usage: python read_agent_logs.py <log directory>")
  sys.exit()


# read_agent_logs.py takes a log directory, reads all agent log files, and produces a summary of
# desired totals or statistics by strategy (type + parameter settings).


# If more than one directory is given, the program aggregates across all of them.

log_dirs = sys.argv[1:]
stats = []

dir_count = 0
file_count = 0

for log_dir in log_dirs:
  if dir_count % 100 == 0: print ("Completed {} directories".format(dir_count))
  dir_count += 1
  for file in os.listdir(log_dir):
    try:
      df = pd.read_pickle(os.path.join(log_dir,file), compression='bz2')
      # print(df)
      events = [ 'AGENT_TYPE', 'STARTING_CASH', 'ENDING_CASH', 'FINAL_CASH_POSITION', 'MARKED_TO_MARKET' ]
      event = "|".join(events)
      df = df[df['EventType'].str.contains(event)]

      at = df.loc[df['EventType'] == 'AGENT_TYPE', 'Event'][0]
      if 'Exchange' in at:
        # There may be different fields to look at later on.
        continue

      file_count += 1

      sc = df.loc[df['EventType'] == 'STARTING_CASH', 'Event'][0]
      ec = df.loc[df['EventType'] == 'ENDING_CASH', 'Event'][0]
      fcp = df.loc[df['EventType'] == 'FINAL_CASH_POSITION', 'Event'][0]
      fv = df.loc[df['EventType'] == 'MARKED_TO_MARKET', 'Event'][0]

      ret = fcp - sc
      surp = fv - sc
      stats.append({ 'AgentType' : at, 'Return' : ret, 'Surplus' : surp })
    except (IndexError, KeyError):
      continue

df_stats = pd.DataFrame(stats)

print (df_stats.groupby('AgentType').mean())

print ("\nRead {} files in {} log directories.".format(file_count, dir_count))
