#!/bin/bash

if [ $# -le 0 ]; then
  nsims=1
else
  nsims=$1
fi

if [ $# -le 1 ]; then
  loops=1
else
  loops=$2
fi

lev=10
lat=1
#verb="-v"
verb=""

if [ $nsims -lt 20 ]; then
  simul=$nsims
else
  simul=20
fi

for seed in `seq 1 $nsims`;
do

  echo "Launching simulation with nsims $nsims, loops $loops, lev $lev, lat $lat, seed $seed."

  let "mod = $seed % $simul"

  if [ $mod -eq 0 ]; then
    python -u abides.py -c manual_spoof -t=IBM -d=2011-01-01 -n=$loops -e=$lev -f1000000000 --entry_threshold=0.17 --trail_dist=0.085 -s=$seed -l obi_latency_nsims${nsims}_loops${loops}_lev${lev}_lat${lat}_${seed} $verb > ./batch_output/obi_latency_nsims${nsims}_loops${loops}_lev${lev}_lat${lat}_${seed}
  else
    python -u abides.py -c manual_spoof -t=IBM -d=2011-01-01 -n=$loops -e=$lev -f1000000000 --entry_threshold=0.17 --trail_dist=0.085 -s=$seed -l obi_latency_nsims${nsims}_loops${loops}_lev${lev}_lat${lat}_${seed} $verb > ./batch_output/obi_latency_nsims${nsims}_loops${loops}_lev${lev}_lat${lat}_${seed} &
  fi

  sleep 2.0
done

