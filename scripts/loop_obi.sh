#!/bin/bash

if [ $# -eq 0 ]; then
  nsims=1
else
  nsims=$1
fi

for i in `seq 1 19`;
do
  echo "Launching simulation $i"
  python -u abides.py -c loop_obi -n=$nsims -e10 -f1000000000 --entry_threshold=0.17 --trail_dist=0.085 -s ${i} > ./batch_output/loop_obi_e10_${i} &

  sleep 2.0
done

python -u abides.py -c loop_obi -n=$nsims -e10 -f1000000000 --entry_threshold=0.17 --trail_dist=0.085 -s 20 > ./batch_output/loop_obi_e10_20

for i in `seq 1 19`;
do
  echo "Launching simulation $i"
  python -u abides.py -c loop_obi -n=$nsims -e1 -f1000000000 --entry_threshold=0.17 --trail_dist=0.085 -s ${i} > ./batch_output/loop_obi_e1_${i} &
  sleep 2.0
done

python -u abides.py -c loop_obi -n=$nsims -e1 -f1000000000 --entry_threshold=0.17 --trail_dist=0.085 -s 20 > ./batch_output/loop_obi_e1_20

for i in `seq 1 19`;
do
  echo "Launching simulation $i"
  python -u abides.py -c loop_obi -n=$nsims -e5 -f1000000000 --entry_threshold=0.17 --trail_dist=0.085 -s ${i} > ./batch_output/loop_obi_e5_${i} &
  sleep 2.0
done

python -u abides.py -c loop_obi -n=$nsims -e5 -f1000000000 --entry_threshold=0.17 --trail_dist=0.085 -s 20 > ./batch_output/loop_obi_e5_20

for i in `seq 1 19`;
do
  echo "Launching simulation $i"
  python -u abides.py -c loop_obi -n=$nsims -e3 -f1000000000 --entry_threshold=0.17 --trail_dist=0.085 -s ${i} > ./batch_output/loop_obi_e3_${i} &
  sleep 2.0
done

python -u abides.py -c loop_obi -n=$nsims -e3 -f1000000000 --entry_threshold=0.17 --trail_dist=0.085 -s 20 > ./batch_output/loop_obi_e3_20

for i in `seq 1 19`;
do
  echo "Launching simulation $i"
  python -u abides.py -c loop_obi -n=$nsims -e20 -f1000000000 --entry_threshold=0.17 --trail_dist=0.085 -s ${i} > ./batch_output/loop_obi_e20_${i} &
  sleep 2.0
done

python -u abides.py -c loop_obi -n=$nsims -e20 -f1000000000 --entry_threshold=0.17 --trail_dist=0.085 -s 20 > ./batch_output/loop_obi_e20_20

