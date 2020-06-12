#!/usr/bin/env bash


if [ -n "$1" ]; then
  echo "Running experiment config ${1}..."
  source $1  # get variables seeds, mm_povs, mm_min_order_sizes, mm_window_sizes, mm_num_ticks, mm_wake_up_freqs, mm_skew_betas (if needed), mm_spread_alphas, mm_level_spacings
             # date, ticker, experiment_name, global_seed, num_runs, num_parallel_runs, start_time, end_time, plot_config,
             # abides_config,
else
  echo " Experiment config not supplied."
  exit 1
fi


### GENERATE PARAM CONFIGS

python -u util/random_search.py -l ${seeds} -l ${mm_povs} -l ${mm_min_order_sizes} -l ${mm_window_sizes} \
       -l ${mm_num_ticks} -l ${mm_wake_up_freqs} -l ${mm_skew_betas} -l ${mm_spread_alphas} -l ${mm_level_spacings} \
       -n ${num_runs} -s ${global_seed} > /tmp/vars.txt


### RUN EXPERIMENTS

echo "Running simulations -- to supervise results, run command:"
echo
echo "tail -f batch_output/${experiment_name}*.err nohup.out"
echo

rm -f nohup.out
mkdir -p batch_output

while IFS= read -r line; do

    a=($(echo "$line" | tr ',' '\n'))

    seed="${a[0]}"
    pov="${a[1]}"
    min_order_size="${a[2]}"
    window_size="${a[3]}"
    num_ticks="${a[4]}"
    wake_up_freq="${a[5]}"
    skew_beta="${a[6]}"
    spread_alpha="${a[7]}"
    level_spacing="${a[8]}"

    baseline_log=${experiment_name}_${seed}_${date}_${pov}_${min_order_size}_${window_size}_${num_ticks}_${wake_up_freq}_${skew_beta}_${spread_alpha}_${level_spacing}

    rm -f batch_output/${baseline_log}.err
    sem -j${num_parallel_runs} --line-buffer python -u abides.py -c ${abides_config} \
                        -t ${ticker} \
                        -d ${date} \
                        -l ${baseline_log} \
                        -s ${seed} \
                        --start-time ${start_time} \
                        --end-time ${end_time} \
                        --mm-pov ${pov} \
                        --mm-min-order-size ${min_order_size} \
                        --mm-window-size ${window_size} \
                        --mm-num-ticks ${num_ticks} \
                        --mm-skew-beta ${skew_beta} \
                        --mm-spread_alpha ${spread_alpha} \
                        --mm-level-spacing ${level_spacing} \
                        --mm-wake-up-freq ${wake_up_freq} > batch_output/${baseline_log}.err 2>&1

done < /tmp/vars.txt

sem --wait

### PLOT OUTPUTS

cd util/plotting

echo "Running plots -- to supervise results, run command:"
echo
echo "tail -f util/plotting/batch_output/${experiment_name}*.err nohup.out"
echo

mkdir -p batch_output
mkdir -p viz/${experiment_name}

while IFS= read -r line; do

    a=($(echo "$line" | tr ',' '\n'))

    seed="${a[0]}"
    pov="${a[1]}"
    min_order_size="${a[2]}"
    window_size="${a[3]}"
    num_ticks="${a[4]}"
    wake_up_freq="${a[5]}"
    skew_beta="${a[6]}"
    spread_alpha="${a[7]}"
    level_spacing="${a[8]}"

    baseline_log=${experiment_name}_${seed}_${date}_${pov}_${min_order_size}_${window_size}_${num_ticks}_${wake_up_freq}_${skew_beta}_${spread_alpha}_${level_spacing}

    stream="../../log/${baseline_log}/EXCHANGE_AGENT.bz2"
    book="../../log/${baseline_log}/ORDERBOOK_${ticker}_FULL.bz2"
    out_file="viz/${experiment_name}/${baseline_log}.png"

    sem -j${num_parallel_runs} --line-buffer python -u liquidity_telemetry.py ${stream} ${book} -o ${out_file} \
      --plot-config ${plot_config} > batch_output/${baseline_log}.err 2>&1

done < /tmp/vars.txt

sem --wait

cd ../..

echo "Experiment ${experiment_name} complete!"
echo