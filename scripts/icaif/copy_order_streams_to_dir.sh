#!/usr/bin/env bash

# dir to hold all streams

mkdir -p return_streams

# subdirs for synth, hist and marketreplay
mkdir -p return_streams/icaif_synth
mkdir -p return_streams/icaif_hist
mkdir -p return_streams/marketreplay

log_home_dir="/home/ec2-user/efs/_abides/dev/mm/abides-icaif/abides/log"
num_parallel_runs=16

## ICAIF synth

stock='ABM'
date=20200101
seeds=$(seq 100 300)
#seeds=$(seq 100 105)

for seed in ${seeds[*]}
  do
    base_name=icaif_synth_${stock}_${date}_${seed}
    base_log=${log_home_dir}/${base_name}
    echo "Copying stream ${base_log}"
    sem -j${num_parallel_runs} --line-buffer \
      mkdir -p return_streams/icaif_synth/${base_name} && cp ${base_log}/EXCHANGE_AGENT.bz2 return_streams/icaif_synth/${base_name}/EXCHANGE_AGENT.bz2
    echo "Stream ${base_log} done"
  done
sem --wait


# copy hist files

# Convert order streams
stocks=(IBM GS PG)
dates=(20190628 20190603)
seeds=$(seq 1 20)
#seeds=$(seq 17 17)

for stock in ${stocks[*]}
  do
    for date in ${dates[*]}
      do
        for seed in ${seeds[*]}
          do
            base_name=icaif_hist_${stock}_${date}_${seed}
            base_log=${log_home_dir}/${base_name}
            echo "Copying stream ${base_log}"
            sem -j${num_parallel_runs} --line-buffer \
            mkdir -p return_streams/icaif_hist/${base_name} && cp ${base_log}/EXCHANGE_AGENT.bz2 return_streams/icaif_hist/${base_name}/EXCHANGE_AGENT.bz2
            echo "Stream ${base_log} done"
          done
      done
  done

# copy marketreplay streams


tickers=(AAPL CSCO DWDP IBM JPM MMM V XOM AXP CVX INTC KO MRK PFE TRV VZ BA DIA GS MSFT PG UNH WBA CAT DIS HD JNJ MCD NKE UTX WMT)
#tickers=(DWDP MMM V XOM AXP CVX KO MRK PFE TRV VZ BA UNH WBA CAT DIS HD JNJ MCD NKE UTX WMT)
#tickers=(MSFT CSCO DIA INTC JPM AAPL)
#tickers=(GS PG JPM)

dates=(20190628 20190627 20190626 20190625 20190624
       20190621 20190620 20190619 20190618 20190617
       20190614 20190613 20190612 20190611 20190610
       20190607 20190606 20190605 20190604 20190603)

log_home=/home/ec2-user/efs/data/get_real_data/marketreplay-logs/log


for ticker in ${tickers[*]}
    do
    for date in ${dates[*]}
      do
        base_name=marketreplay_${ticker}_${date}
        base_log=${log_home}/${base_name}
        echo "Copying stream ${base_log}"
        sem -j${num_parallel_runs} --line-buffer \
            mkdir -p return_streams/marketreplay/${base_name} && cp ${base_log}/EXCHANGE_AGENT.bz2 return_streams/marketreplay/${base_name}/EXCHANGE_AGENT.bz2
        echo "Stream ${base_log} done"
      done
done
sem --wait