#!/bin/bash

ticker=IBM
dates=(20190628 20190627 20190626 20190625 20190624
       20190621 20190620 20190619 20190618 20190617
       20190614 20190613 20190612 20190611 20190610
       20190607 20190606 20190605 20190604 20190603)
seeds=$(seq 10 15)
povs=(0.01 0.1 0.5)
experiment_name=lambda_0.7mHz_impact

for seed in ${seeds[@]}
do
    for date in ${dates[@]}
    do
        # Baseline (No Execution Agents)
        baseline_log=${experiment_name}_no_${seed}_${date}
        mv log/${baseline_log}/ORDERBOOK_${ticker}_FULL.bz2 log/${baseline_log}/ORDERBOOK_POV_FULL.bz2

        # With Execution Agents

        for pov in ${povs[@]}
            do
                execution_log=${experiment_name}_yes_${seed}_${pov}_${date}
                        mv log/${execution_log}/ORDERBOOK_${ticker}_FULL.bz2 log/${execution_log}/ORDERBOOK_POV_FULL.bz2

        done
    done
done
wait


