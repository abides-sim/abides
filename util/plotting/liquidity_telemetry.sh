#!/usr/bin/env bash

stream='../../log/lambda_0.7kHz_impact_no_15_20190610/EXCHANGE_AGENT.bz2'
book='../../log/lambda_0.7kHz_impact_no_15_20190610/ORDERBOOK_IBM_FULL.bz2'
out_file='lambda_0.7kHz_impact_no_15_20190610.png'

python -u liquidity_telemetry.py ${stream} ${book} -o ${out_file}

stream='../../log/lambda_0.7kHz_impact_yes_15_0.01_20190610/EXCHANGE_AGENT.bz2'
book='../../log/lambda_0.7kHz_impact_yes_15_0.01_20190610/ORDERBOOK_IBM_FULL.bz2'
out_file='lambda_0.7kHz_impact_yes_15_0.01_20190610.png'

python -u liquidity_telemetry.py ${stream} ${book} -o ${out_file}

stream='../../log/lambda_0.7kHz_impact_yes_15_0.05_20190610/EXCHANGE_AGENT.bz2'
book='../../log/lambda_0.7kHz_impact_yes_15_0.05_20190610/ORDERBOOK_IBM_FULL.bz2'
out_file='lambda_0.7kHz_impact_yes_15_0.05_20190610.png'

python -u liquidity_telemetry.py ${stream} ${book} -o ${out_file}

stream='../../log/lambda_0.7kHz_impact_yes_15_0.1_20190610/EXCHANGE_AGENT.bz2'
book='../../log/lambda_0.7kHz_impact_yes_15_0.1_20190610/ORDERBOOK_IBM_FULL.bz2'
out_file='lambda_0.7kHz_impact_yes_15_0.1_20190610.png'

python -u liquidity_telemetry.py ${stream} ${book} -o ${out_file}

stream='../../log/lambda_0.7kHz_impact_yes_15_0.5_20190610/EXCHANGE_AGENT.bz2'
book='../../log/lambda_0.7kHz_impact_yes_15_0.5_20190610/ORDERBOOK_IBM_FULL.bz2'
out_file='lambda_0.7kHz_impact_yes_15_0.5_20190610.png'

python -u liquidity_telemetry.py ${stream} ${book} -o ${out_file}
