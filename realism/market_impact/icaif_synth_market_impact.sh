#!/bin/bash

stock='ABM'
date=20200101
seeds=$(seq 100 200)
num_parallel_runs=5
tao=60

for seed in ${seeds[*]}
  do
    nohup sem -j${num_parallel_runs} --line-buffer \
      python -u ${PWD}/realism/market_impact/abm_market_impact.py --stock ${stock} --date ${date} \
      --log ${PWD}/log/icaif_synth_${stock}_${date}_${seed} 2>&1 --tao ${tao} > icaif_synth_market_impact.log &
  done
sem --wait