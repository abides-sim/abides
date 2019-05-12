import os
import pandas as pd
import sys

# Auto-detect terminal width.
pd.options.display.width = None
pd.options.display.max_rows = 500000
pd.options.display.max_colwidth = 200

if len(sys.argv) < 2:
  print ("Usage: python dump.py <log directory>")
  sys.exit()


# stats.py takes one or more log directories, reads the summary log files, and produces a summary of
# the agent surpluses and returns by strategy (type + parameter settings).


# If more than one directory is given, the program aggregates across all of them.

log_dirs = sys.argv[1:]
agents = {}
games = []
stats = []

dir_count = 0

for log_dir in log_dirs:
  if dir_count % 100 == 0: print ("Completed {} directories".format(dir_count))
  dir_count += 1
  for file in os.listdir(log_dir):
    if 'summary' not in file: continue

    df = pd.read_pickle(os.path.join(log_dir,file), compression='bz2')
  
    events = [ 'STARTING_CASH', 'ENDING_CASH', 'FINAL_CASH_POSITION', 'FINAL_VALUATION' ]
    event = "|".join(events)
    df = df[df['EventType'].str.contains(event)]
  
    for x in df.itertuples():
      id = x.AgentID
      if id not in agents:
        agents[id] = { 'AGENT_TYPE' : x.AgentStrategy }
      agents[id][x.EventType] = x.Event

    game_ret = 0
    game_surp = 0

    for id, agent in agents.items():
      at = agent['AGENT_TYPE']

      if 'Impact' in at: continue

      sc = agent['STARTING_CASH']
      ec = agent['ENDING_CASH']
      fcp = agent['FINAL_CASH_POSITION']
      fv = agent['FINAL_VALUATION']
  
      ret = ec - sc
      surp = fcp - sc + fv

      game_ret += ret
      game_surp += surp

      stats.append({ 'AgentType' : at, 'Return' : ret, 'Surplus' : surp })

    games.append({ 'GameReturn' : game_ret, 'GameSurplus' : game_surp })


df_stats = pd.DataFrame(stats)
df_game = pd.DataFrame(games)

print ("Agent Mean")
print (df_stats.groupby('AgentType').mean())
print ("Agent Std")
print (df_stats.groupby('AgentType').std())
print ("Game Mean")
print (df_game.mean())
print ("Game Std")
print (df_game.std())

print ("\nRead summary files in {} log directories.".format(dir_count))
