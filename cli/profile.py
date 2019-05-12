import pstats
import sys

if len(sys.argv) < 2:
  print ('Usage: python cli/profile.py <sort by field>')
  sys.exit()

field = sys.argv[1]

if field not in ['time', 'cumulative', 'tottime', 'cumtime', 'ncalls']:
  print ('Sort by field must be one of: time, cumulative, tottime, cumtime, ncalls.')
  sys.exit()

p = pstats.Stats('runstats.prof')
p.strip_dirs().sort_stats(field).print_stats(50)

