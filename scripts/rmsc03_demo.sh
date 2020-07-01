#!/bin/bash

NUM_JOBS=48  # number of jobs to run in parallel, may need to reduce to satisfy computational constraints

python -u abides.py -c rmsc03 -t ABM -d 20200603 -s 1234 -l rmsc03_two_hour

cd util/plotting && python -u liquidity_telemetry.py ../../log/rmsc03_two_hour/EXCHANGE_AGENT.bz2 ../../log/rmsc03_two_hour/ORDERBOOK_ABM_FULL.bz2 \
-o rmsc03_two_hour.png -c configs/plot_09.30_11.30.json && cd ../../

for pov in  0.01 0.05 0.1 0.5; do
    sem -j${NUM_JOBS} --line-buffer python -u abides.py -c rmsc03 -t ABM -d 20200603 -s 1234 -l rmsc03_two_hour_pov_${pov} -e -p ${pov}
done
sem --wait

cd realism && python -u impact_single_day_pov.py plot_configs/plot_configs/single_day/rmsc03_demo_single_day.json && cd ..

# Multiple seeds for execution experiment
for seed in $(seq 100 120); do
  sem -j${NUM_JOBS} --line-buffer python -u abides.py -c rmsc03 -t ABM -d 20200605 -s ${seed} -l rmsc03_demo_no_${seed}_20200605
  for pov in  0.01 0.05 0.1 0.5; do
      sem -j${NUM_JOBS} --line-buffer python -u abides.py -c rmsc03 -t ABM -d 20200605 -s ${seed} -l rmsc03_demo_yes_${seed}_pov_${pov}_20200605 -e -p ${pov}
  done
done
sem --wait

# Plot multiple seed experiment
cd realism && python -u impact_multiday_pov.py plot_configs/plot_configs/multiday/rmsc03_demo_multiday.json -n ${NUM_JOBS} && cd ..

