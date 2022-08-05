import os
import pandas as pd
import sys

# Auto-detect terminal width.
pd.options.display.width = None
pd.options.display.max_rows = 500000
pd.options.display.max_colwidth = 200

if len(sys.argv) < 2:
  print ("Usage: python retail_execution_summary.py <log directory>")
  sys.exit()


# read_agent_logs.py takes a log directory, reads all agent log files, and produces a summary of
# desired totals or statistics by strategy (type + parameter settings).

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
      events = [ 'AGENT_TYPE', 'TOTAL_ORDERS', 'AVG_ABS_SLIPPAGE', 'NET_SLIPPAGE', 'MAX_ABS_SLIPPAGE', 'PCT_IN', 'PCT_OUT', 'AVG_TIME', 'FINAL_PCT_PROFIT', 'SLIP_ADJ_PCT_PROFIT']
      event = "|".join(events)
      df = df[df['EventType'].str.contains(event)]

      at = df.loc[df['EventType'] == 'AGENT_TYPE', 'Event'][0]
      file_count += 1

      # TODO: calculate any derivative stats?
      to = df.loc[df['EventType'] == 'TOTAL_ORDERS', 'Event'][0]
      aas = df.loc[df['EventType'] == 'AVG_ABS_SLIPPAGE', 'Event'][0]
      ns = df.loc[df['EventType'] == 'NET_SLIPPAGE', 'Event'][0]
      mas = df.loc[df['EventType'] == 'MAX_ABS_SLIPPAGE', 'Event'][0]
      pin = df.loc[df['EventType'] == 'PCT_IN', 'Event'][0]
      pout = df.loc[df['EventType'] == 'PCT_OUT', 'Event'][0]
      ati = df.loc[df['EventType'] == 'AVG_TIME', 'Event'][0]
      fpp = df.loc[df['EventType'] == 'FINAL_PCT_PROFIT', 'Event'][0]
      sapp = df.loc[df['EventType'] == 'SLIP_ADJ_PCT_PROFIT', 'Event'][0]

      stats.append({ 'AgentType' : at, 'TotalOrders': to, 'AvgAbsSlippage': aas, 'NetSlippage': ns, 'MaxAbsSlippage': mas, 'PctIn': pin, 'PctOut': pout, 'AvgTime': ati, 'FinalPctProfit': fpp, 'SlipAdjPctProfit': sapp })

    except (IndexError, KeyError):
      continue

df_stats = pd.DataFrame(stats)
print (df_stats.groupby('AgentType').mean())

print ("\nRead {} files in {} log directories.".format(file_count, dir_count))

# converting to csv
path = "CSVs\\Summaries\\"
if not os.path.isdir(path):
   os.makedirs(path)

# drop log directory from file
dir_name = log_dirs[0].split("\\")[-1]
df_stats.to_csv(path + dir_name + '_ALL.csv', index=False)
df_stats.groupby('AgentType').mean().to_csv(path + dir_name + '_SUMMARY.csv')