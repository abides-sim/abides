date=20190628
ticker="ABM"
experiment_name="expt_rmsc03_adaptive_mm_fund_vol_1e-3_lambda_a_7e-11"
global_seed=123456
num_runs=96
num_parallel_runs=6
start_time='09:30:00'
end_time='10:15:00'
abides_config=rmsc03


seeds="20, 21, 22, 23, 24, 25"
mm_povs="0.05, 0.1, 0.25"
mm_min_order_sizes="1, 20, 50"
mm_window_sizes="'adaptive'"
mm_num_ticks="1, 10, 20, 50"
mm_wake_up_freqs="'1S', '10S', '30S'"
mm_skew_betas="0, 1e-3, 1e-2, 1e-5"
mm_spread_alphas="0.75, 0.85, 0.95"
mm_level_spacings="0.5"

plot_config="configs/plot_09.30_10.15.json"  # relative to util/plotting