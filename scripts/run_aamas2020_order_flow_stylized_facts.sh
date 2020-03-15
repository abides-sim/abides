#!/usr/bin/env bash

dates=(20190628 20190627 20190626 20190625 20190624
       20190621 20190620 20190619 20190618 20190617
       20190614 20190613 20190612 20190611 20190610
       20190607 20190606 20190605 20190604 20190603)



usage()
{
    echo "usage: run_config [[[-r ]] [[-c config]] | [-h]]"
}

run_all=false

while getopts "c:rh" opt; do
  case $opt in
    c) config="$OPTARG"
    ;;
    r) run_all=true
    ;;
    h) usage
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done

err_file=~/code-local/abides-dev/aamas_${config}.err
log_dir=aamas_${config}
num_parallel_runs=4
CONFIG_DIR=~/efs/data/get_real_data/order_flow_data/${config}
value_name="config_3"
diverse_name="config_4"

rm ${err_file}

if [ "${run_all}" = true ]
then
	echo "Running experiments for config ${config}" 2>&1>> ${err_file}

	for d in ${dates[*]}
	  do 
	    echo "Running AAMAS3 smile experiment for date ${d}" 2>&1>> ${err_file}
	    sem -j${num_parallel_runs} python -u abides.py -c ${config}_value -l ${log_dir}_value_$d -d $d -r $d -f "mid_prices/ORDERBOOK_IBM_FREQ_1S_${d}_mid_price.bz2" -s ${value_name} 2>&1>> ${err_file}
	    sleep 0.5
	    sem -j${num_parallel_runs} python -u abides.py -c ${config}_diverse -l ${log_dir}_diverse_$d -d $d -r $d -f "mid_prices/ORDERBOOK_IBM_FREQ_1S_${d}_mid_price.bz2" -s ${diverse_name} 2>&1>> ${err_file}
            sleep 0.5		
	  done
	sem --wait

	echo "Making stream files" 2>&1>> ${err_file}

	for d in ${dates[*]}
	  do
	    aamas_file=log/${log_dir}_value_${d}/EXCHANGE_AGENT.bz2 
	    echo "Processing file ${aamas_file}"
	    sem -j${num_parallel_runs} python -u util/formatting/convert_order_stream.py $aamas_file ${value_name} 5 plot-scripts -o streams 2>&1>> ${err_file}  
	    sleep 0.5
            aamas_file=log/${log_dir}_diverse_${d}/EXCHANGE_AGENT.bz2 
	    echo "Processing file ${aamas_file}"
	    sem -j${num_parallel_runs} python -u util/formatting/convert_order_stream.py $aamas_file ${diverse_name} 5 plot-scripts -o streams 2>&1>> ${err_file}  
	    sleep 0.5
		
	  done
	sem --wait

	rm -r ${CONFIG_DIR}	
	mkdir -p ${CONFIG_DIR}

	mv streams/* ${CONFIG_DIR}
	ln ${CONFIG_DIR}/../aamas3/orders_IBM_*.pkl ${CONFIG_DIR}

fi

echo "Plotting order flow stylized facts for config ${config}" 2>&1>> ${err_file}

cd realism

mkdir -p ${CONFIG_DIR}/plots

python -u order_flow_stylized_facts.py --recompute ${CONFIG_DIR} -o ${CONFIG_DIR}/plots 2>&1>> ${err_file}

echo "Done!" 2>&1>> ${err_file}
