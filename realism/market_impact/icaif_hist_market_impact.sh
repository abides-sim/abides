#!/bin/bash

stocks=(IBM JPM PG)
dates=(20190628 20190603)
seeds=$(seq 1 20)
num_parallel_runs=2
tao=60

for stock in ${stocks[*]}
  do
    for date in ${dates[*]}
      do
        for seed in ${seeds[*]}
          do
            nohup sem -j${num_parallel_runs} --line-buffer \
              python -u ${PWD}/realism/market_impact/abm_market_impact.py --stock ${stock} --date ${date} \
              --log ${PWD}/log/icaif_hist_${stock}_${date}_${seed} 2>&1 --tao ${tao} > icaif_hist_market_impact.log &
          done
      done
  done
sem --wait