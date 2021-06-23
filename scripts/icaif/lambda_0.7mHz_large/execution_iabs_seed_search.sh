#!/usr/bin/env bash

dates=(20190628 20190627 20190626 20190625 20190624
       20190621 20190620 20190619 20190618 20190617
       20190614 20190613 20190612 20190611 20190610
       20190607 20190606 20190605 20190604 20190603)

seeds=$(seq 10 15)

num_parallel_runs=88

for seed in ${seeds[@]}
do
    for date in ${dates[@]}
    do
         sem -j${num_parallel_runs} --line-buffer \
            ./scripts/icaif/lambda_0.7mHz_large/execution_iabs_plots.sh $seed $date
    done
done

sem --wait