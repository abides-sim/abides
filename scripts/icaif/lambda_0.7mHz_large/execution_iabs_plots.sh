#!/usr/bin/env bash

seed=${1:-123456789}
ticker=IBM
date=${2:-20190628}
EFS_DIR=/home/ec2-user/efs 

experiment_name=lambda_0.7mHz_impact
fundamental_path=${EFS_DIR}/data/get_real_data/mid_prices/ORDERBOOK_${ticker}_FREQ_1S_${date}_mid_price.bz2

mkdir -p batch_output

# Baseline (No Execution Agents)
baseline_log=${experiment_name}_no_${seed}_${date}
nohup python -u abides.py -c icaif_market_impact_expt -t ${ticker} -d $date -f ${fundamental_path} -l ${baseline_log} -s $seed --wide-book > batch_output/${baseline_log}.err 2>&1 &

# With Execution Agents
povs=(0.01 0.1 0.5)

for pov in ${povs[@]}
do
    execution_log=${experiment_name}_yes_${seed}_${pov}_${date}
    nohup python -u abides.py -c icaif_market_impact_expt -t $ticker -d $date -f ${fundamental_path} -l ${execution_log} -s $seed -e -p ${pov} --wide-book > batch_output/${execution_log}.err 2>&1 &
    echo $pov
done

wait
