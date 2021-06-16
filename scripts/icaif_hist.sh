#!/bin/bash

stocks=(IBM GS PG)
dates=(20190628 20190603)
seeds=$(seq 1 20)
num_parallel_runs=10

for stock in ${stocks[*]}
  do
    for date in ${dates[*]}
      do
        for seed in ${seeds[*]}
          do
            nohup sem  -j${num_parallel_runs} --line-buffer \
              python -u abides.py -c icaif_hist -t ${stock} -d ${date} \
              -f /home/ec2-user/efs/data/get_real_data/mid_prices/ORDERBOOK_${stock}_FREQ_ALL_${date}_mid_price.bz2 \
              -l icaif_hist_${stock}_${date}_${seed} -s ${seed} > icaif_hist.log &
          done
      done
  done
sem --wait