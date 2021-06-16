#!/bin/bash

## Script to check results of ICAIF configs with telemetry plots

log_home_dir="/home/ec2-user/efs/_abides/dev/mm/abides-icaif/abides/log"
num_parallel_runs=6

cd util/plotting

## ICAIF HIST

stocks=(IBM GS PG)
dates=(20190628 20190603)
#seeds=$(seq 1 20)
seeds=$(seq 17 17)

mkdir -p viz/icaif_hist
for stock in ${stocks[*]}
  do
    for date in ${dates[*]}
      do
        for seed in ${seeds[*]}
          do
            base_name=icaif_hist_${stock}_${date}_${seed}
            base_log=${log_home_dir}/${base_name}
            nohup sem -j${num_parallel_runs} --line-buffer \
              python -u liquidity_telemetry.py ${base_log}/EXCHANGE_AGENT.bz2 ${base_log}/ORDERBOOK_${stock}_FULL.bz2 \
              -o viz/icaif_hist/${base_name}.png --plot-config configs/plot_09.30_16.00.json &
          done
      done
  done

## ICAIF SYNTH
stock='ABM'
date=20200101
#seeds=$(seq 100 200)
seeds=$(seq 100 105)

mkdir -p viz/icaif_synth
for seed in ${seeds[*]}
  do
    base_name=icaif_synth_${stock}_${date}_${seed}
    base_log=${log_home_dir}/${base_name}
    nohup sem -j${num_parallel_runs} --line-buffer \
      python -u liquidity_telemetry.py ${base_log}/EXCHANGE_AGENT.bz2 ${base_log}/ORDERBOOK_${stock}_FULL.bz2 \
              -o viz/icaif_synth/${base_name}.png --plot-config configs/plot_09.30_16.00.json &
  done

cd ../..