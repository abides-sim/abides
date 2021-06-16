#!/usr/bin/env bash


log_home_dir="/home/ec2-user/efs/_abides/dev/mm/abides-icaif/abides/log"
num_parallel_runs=16
# Convert order streams
stream_dir=streams
stocks=(IBM GS PG)
dates=(20190628 20190603)
seeds=$(seq 1 20)
#seeds=$(seq 17 17)

mkdir -p ${stream_dir}
mkdir -p batch_output

mkdir -p viz/icaif_hist
for stock in ${stocks[*]}
  do
    for date in ${dates[*]}
      do
        for seed in ${seeds[*]}
          do
            base_name=icaif_hist_${stock}_${date}_${seed}
            base_log=${log_home_dir}/${base_name}
            echo "Converting stream ${base_log}"
            err_file=${base_name}.err
            sem -j${num_parallel_runs} --line-buffer \
             python -u util/formatting/convert_order_stream.py ${base_log}/EXCHANGE_AGENT.bz2 hist 5 plot-scripts \
             -o ${stream_dir} --suffix _${stock}_${seed} 2>&1> batch_output/${err_file}
            sleep 0.5
            echo "Stream ${base_log} done"

          done
      done
  done