#!/bin/bash

tickers=('MMM' 'AXP' 'BA' 'CAT' 'CVX' 'CSCO' 'KO' 'DWDP' 'XOM' 'HD' 'IBM' 'INTC'
         'JNJ' 'MCD' 'MRK' 'MSFT' 'NKE' 'PFE' 'PG' 'TRV' 'UNH' 'UTX' 'VZ' 'V' 'WMT'
         'WBA' 'DIS' 'AAPL' 'JPM' 'GS')

dates=('2019-06-28' '2019-06-27' '2019-06-26' '2019-06-25' '2019-06-24'
       '2019-06-21' '2019-06-20' '2019-06-19' '2019-06-18' '2019-06-17'
       '2019-06-14' '2019-06-13' '2019-06-12' '2019-06-11' '2019-06-10'
       '2019-06-07' '2019-06-06' '2019-06-05' '2019-06-04' '2019-06-03')

num_parallel_runs=10

for ticker in ${tickers[*]}
  do
    for date in ${dates[*]}
      do
        sem -j${num_parallel_runs} --line-buffer \
        python -u ${PWD}/realism/marketreplay_market_impact.py --tao 60 --ticker ${ticker} --date ${date} 2>&1 > mr_market_impact.log
      done
  done
sem --wait