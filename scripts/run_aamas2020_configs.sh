#!/usr/bin/env bash

#  Generate ABIDES runs for asset returns stylized fact plots

cd ..
COUNTER=1
while :
do
    echo "Cleaning new historical time series"
    python util/formatting/clean_ohlc_price_series.py /media/nick/FinData/intraday/1m_ohlc/1m_ohlc_2011
    sleep 3
    echo "hist_fund_value"
    python abides.py -c hist_fund_value -d 20000101 -f clean.pkl -s HIST.VAL -l hist_fund_value/$COUNTER
    rm -r log/*/*/*Agent*
    echo "hist_fund_diverse"
    python abides.py -c hist_fund_diverse -d 20000101 -f clean.pkl -s HIST.DIV -l hist_fund_diverse/$COUNTER
    rm -r log/*/*/*Agent*
    rm -r log/*/*/*MOMENTUM*
    echo "random_fund_value"
    python abides.py -c random_fund_value -d 20000101 -s RAND.VAL -l random_fund_value/$COUNTER
    rm -r log/*/*/*Agent*
    echo "random_fund_diverse"
    python abides.py -c random_fund_diverse -d 20000101 -s RAND.DIV -l random_fund_diverse/$COUNTER
    rrm -r log/*/*/*Agent*
    rm -r log/*/*/*MOMENTUM*
    COUNTER=$((COUNTER + 1))
    echo "COUNTER->$COUNTER"
done
