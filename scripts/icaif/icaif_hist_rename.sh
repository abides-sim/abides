
stocks=(IBM GS PG)
dates=(20190628 20190603)
seeds=$(seq 1 20)
num_parallel_runs=16
povs=(0.01 0.025 0.05 0.1 0.25 0.5)


for pov in ${povs[*]}
  do
  for stock in ${stocks[*]}
  do
    for date in ${dates[*]}
      do
        for seed in ${seeds[*]}
          do
             ## Rename ${yes} -> yes
#            base_name=icaif_hist_${yes}_${seed}_${stock}_pov_${pov}_${date}
#            new_base_name=icaif_hist_yes_${seed}_${stock}_pov_${pov}_${date}
#            sem  -j${num_parallel_runs} --line-buffer \
#            mv log/${base_name} log/${new_base_name}

             ## Rename orderbooks
             base_name=icaif_hist_yes_${seed}_${stock}_pov_${pov}_${date}
             mv log/${base_name}/ORDERBOOK_${stock}_FULL.bz2 log/${base_name}/ORDERBOOK_POV_FULL.bz2
          done
      done
  done
sem --wait
done


for stock in ${stocks[*]}
  do
    for date in ${dates[*]}
      do
        for seed in ${seeds[*]}
          do
            base_name=icaif_hist_no_${seed}_${stock}_${date}
            mv log/${base_name}/ORDERBOOK_${stock}_FULL.bz2 log/${base_name}/ORDERBOOK_POV_FULL.bz2
          done
      done
  done
sem --wait