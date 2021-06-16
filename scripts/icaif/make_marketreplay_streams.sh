#!/usr/bin/env bash


# Convert order streams
stream_dir=streams

tickers=(AAPL CSCO DWDP IBM JPM MMM V XOM AXP CVX INTC KO MRK PFE TRV VZ BA DIA GS MSFT PG UNH WBA CAT DIS HD JNJ MCD NKE UTX WMT)
#tickers=(DWDP MMM V XOM AXP CVX KO MRK PFE TRV VZ BA UNH WBA CAT DIS HD JNJ MCD NKE UTX WMT)
#tickers=(MSFT CSCO DIA INTC JPM AAPL)
#tickers=(GS PG JPM)

dates=(20190628 20190627 20190626 20190625 20190624
       20190621 20190620 20190619 20190618 20190617
       20190614 20190613 20190612 20190611 20190610
       20190607 20190606 20190605 20190604 20190603)

num_parallel_runs=16
log_home=/home/ec2-user/efs/data/get_real_data/marketreplay-logs/log

mkdir -p batch_output

for ticker in ${tickers[*]}
    do
    for date in ${dates[*]}
      do
        base_name=marketreplay_${ticker}_${date}
        base_log=${log_home}/${base_name}
        echo "Converting stream ${base_log}"
        err_file=${base_name}.err
        sem -j${num_parallel_runs} --line-buffer \
          python -u util/formatting/convert_order_stream.py ${base_log}/EXCHANGE_AGENT.bz2 marketreplay 5 plot-scripts \
             -o ${stream_dir} --suffix _${ticker} 2>&1> batch_output/${err_file}
        sleep 0.5
        echo "Stream ${base_log} done"
      done
done
sem --wait
