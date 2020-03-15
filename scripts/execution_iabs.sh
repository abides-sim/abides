#!/bin/bash

obc_script=${PWD}/util/formatting/convert_order_book.py
osc_script=${PWD}/util/formatting/convert_order_stream.py

seed=123456789
ticker=IABS
date=19910602
nlevels=10
fundamental_path=/efs/data/get_real_data/mid_prices/ORDERBOOK_IBM_FREQ_ALL_20190628_mid_price.bz2

# Baseline (No Execution Agents)
baseline_log=execution_iabs_no_${seed}
python -u abides.py -c execution_iabs -t ${ticker} -d $date -f ${fundamental_path} -l ${baseline_log} -s $seed
python -u $obc_script ${PWD}/log/${baseline_log}/ORDERBOOK_${ticker}_FULL.bz2 ${ticker} ${nlevels} -o ${PWD}/log/${baseline_log}
python -u $osc_script ${PWD}/log/${baseline_log}/EXCHANGE_AGENT.bz2 ${ticker} ${nlevels} plot-scripts -o ${PWD}/log/${baseline_log}

# With Execution Agents
execution_log=execution_iabs_yes_${seed}
python -u abides.py -c execution_iabs -t $ticker -d $date -f ${fundamental_path} -l ${execution_log} -s $seed -e
python -u $obc_script ${PWD}/log/${execution_log}/ORDERBOOK_${ticker}_FULL.bz2 ${ticker} ${nlevels} -o ${PWD}/log/${execution_log}
python -u $osc_script ${PWD}/log/${execution_log}/EXCHANGE_AGENT.bz2 ${ticker} ${nlevels} plot-scripts -o ${PWD}/log/${execution_log}

rm -rf ${PWD}/log/${baseline_log}/NOISE*
rm -rf ${PWD}/log/${baseline_log}/VALUE*
rm -rf ${PWD}/log/${baseline_log}/MOMENTUM*

rm -rf ${PWD}/log/${execution_log}/NOISE*
rm -rf ${PWD}/log/${execution_log}/VALUE*
rm -rf ${PWD}/log/${execution_log}/MOMENTUM*