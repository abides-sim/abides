#!/bin/bash

stock='ABM'
date=20200101
seeds=$(seq 100 200)
num_parallel_runs=50

for seed in ${seeds[*]}
  do
    nohup sem -j${num_parallel_runs} --line-buffer \
      python -u abides.py -c icaif_synth -t 'ABM' -d 20200101 -l icaif_synth_${stock}_${date}_${seed} -s ${seed} > icaif_synth.log &
  done
sem --wait