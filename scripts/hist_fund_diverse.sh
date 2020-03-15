#!/bin/bash

fund_path=/efs/data/get_real_data/mid_prices/ORDERBOOK_IBM_FREQ_ALL_20190628_mid_price.bz2
python -u abides.py -c hist_fund_diverse -t 'IBM' -d 20190628 -f $fund_path -l hist_fund_diverse -s 123456789
