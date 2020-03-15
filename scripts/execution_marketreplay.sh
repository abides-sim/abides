#!/bin/bash

obc_script=${PWD}/util/formatting/convert_order_book.py
osc_script=${PWD}/util/formatting/convert_order_stream.py

seed=123456789
ticker=IBM
date=20190628
nlevels=50

# Baseline (No Execution Agents)
baseline_log=marketreplay_${ticker}_${date}
python -u abides.py -c execution_marketreplay -t $ticker -d $date -l ${baseline_log} -s $seed
python -u $obc_script ${PWD}/log/${baseline_log}/ORDERBOOK_${ticker}_FULL.bz2 ${ticker} ${nlevels} -o ${PWD}/log/${baseline_log}
python -u $osc_script ${PWD}/log/${baseline_log}/EXCHANGE_AGENT.bz2 ${ticker} ${nlevels} plot-scripts -o ${PWD}/log/${baseline_log}

# With Execution Agents
execution_log=execution_marketreplay_${ticker}_${date}
python -u abides.py -c execution_marketreplay -t $ticker -d $date -l ${execution_log} -s $seed -e
python -u $obc_script ${PWD}/log/${execution_log}/ORDERBOOK_${ticker}_FULL.bz2 ${ticker} ${nlevels} -o ${PWD}/log/${execution_log}
python -u $osc_script ${PWD}/log/${execution_log}/EXCHANGE_AGENT.bz2 ${ticker} ${nlevels} plot-scripts -o ${PWD}/log/${execution_log}