#!/usr/bin/env bash

stocks=(IBM GS PG)
dates=(20190628 20190603)
seeds=$(seq 1 20)
num_parallel_runs=16
povs=(0.01 0.025 0.05 0.1 0.25 0.5)

echo "Running simulations -- to supervise results, run command:"
echo
echo "tail -f batch_output/icaif_hist_no_*.err batch_output/icaif_hist_yes_*.err nohup.out"
echo

for stock in ${stocks[*]}
  do
    for date in ${dates[*]}
      do
        for seed in ${seeds[*]}
          do
            base_name=icaif_hist_no_${seed}_${stock}_${date}
            nohup sem  -j${num_parallel_runs} --line-buffer \
              python -u abides.py -c icaif_hist -t ${stock} -d ${date} \
              -f /home/ec2-user/efs/data/get_real_data/mid_prices/ORDERBOOK_${stock}_FREQ_ALL_${date}_mid_price.bz2 \
              -l ${base_name} -s ${seed} > batch_output/${base_name}.log &
          done
      done
  done
sem --wait

for pov in ${povs[*]}
  do
  for stock in ${stocks[*]}
  do
    for date in ${dates[*]}
      do
        for seed in ${seeds[*]}
          do
            base_name=icaif_hist_${yes}_${seed}_${stock}_pov_${pov}_${date}
            nohup sem  -j${num_parallel_runs} --line-buffer \
              python -u abides.py -c icaif_hist -t ${stock} -d ${date} \
              -f /home/ec2-user/efs/data/get_real_data/mid_prices/ORDERBOOK_${stock}_FREQ_ALL_${date}_mid_price.bz2 \
              -l ${base_name} -s ${seed} -e -p ${pov} > batch_output/${base_name}.log &
          done
      done
  done
sem --wait
  done

