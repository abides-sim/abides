#!/usr/bin/env bash

experiment_name=lambda_0.7kHz_impact
seed=15
date=20190610
ticker=IBM

stream="../../log/${experiment_name}_no_${seed}_${date}/EXCHANGE_AGENT.bz2"
book="../../log/${experiment_name}_no_${seed}_${date}/ORDERBOOK_${ticker}_FULL.bz2"
out_file="viz/${experiment_name}_no_${seed}_${date}.png"

python -u liquidity_telemetry.py ${stream} ${book} -o ${out_file}

stream="../../log/${experiment_name}_yes_${seed}_0.01_${date}/EXCHANGE_AGENT.bz2"
book="../../log/${experiment_name}_yes_${seed}_0.01_${date}/ORDERBOOK_${ticker}_FULL.bz2"
out_file="viz/${experiment_name}_yes_${seed}_0.01_${date}.png"

python -u liquidity_telemetry.py ${stream} ${book} -o ${out_file}

stream="../../log/${experiment_name}_yes_${seed}_0.05_${date}/EXCHANGE_AGENT.bz2"
book="../../log/${experiment_name}_yes_${seed}_0.05_${date}/ORDERBOOK_${ticker}_FULL.bz2"
out_file="viz/${experiment_name}_yes_${seed}_0.05_${date}.png"

python -u liquidity_telemetry.py ${stream} ${book} -o ${out_file}

stream="../../log/${experiment_name}_yes_${seed}_0.1_${date}/EXCHANGE_AGENT.bz2"
book="../../log/${experiment_name}_yes_${seed}_0.1_${date}/ORDERBOOK_${ticker}_FULL.bz2"
out_file="viz/${experiment_name}_yes_${seed}_0.1_${date}.png"

python -u liquidity_telemetry.py ${stream} ${book} -o ${out_file}

stream="../../log/${experiment_name}_yes_${seed}_0.5_${date}/EXCHANGE_AGENT.bz2"
book="../../log/${experiment_name}_yes_${seed}_0.5_${date}/ORDERBOOK_${ticker}_FULL.bz2"
out_file="viz/${experiment_name}_yes_${seed}_0.5_${date}.png"

python -u liquidity_telemetry.py ${stream} ${book} -o ${out_file}
