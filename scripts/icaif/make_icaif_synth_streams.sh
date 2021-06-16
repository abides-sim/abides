#!/usr/bin/env bash


log_home_dir="/home/ec2-user/efs/_abides/dev/mm/abides-icaif/abides/log"
num_parallel_runs=16
# Convert order streams
stream_dir=streams

## ICAIF synth

stock='ABM'
date=20200101
seeds=$(seq 100 300)
#seeds=$(seq 100 105)


#rm -f ${stream_dir}/*
mkdir -p ${stream_dir}
mkdir -p batch_output

for seed in ${seeds[*]}
  do
    base_name=icaif_synth_${stock}_${date}_${seed}
    base_log=${log_home_dir}/${base_name}
    echo "Converting stream ${base_log}"
    err_file=${base_name}.err
    sem -j${num_parallel_runs} --line-buffer \
     python -u util/formatting/convert_order_stream.py ${base_log}/EXCHANGE_AGENT.bz2 ${stock} 5 plot-scripts \
     -o ${stream_dir} --suffix _${seed} 2>&1> batch_output/${err_file}
    sleep 0.5
    echo "Stream ${base_log} done"
  done
sem --wait
