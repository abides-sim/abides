#!/bin/bash

# Example script to run multiple ABIDES simulations in parallel

seed=123456789
num_simulations=10             # Total number of ABIDES simulations
num_parallel=5                 # Number of simulations to run in parallel
config=sparse_zi_100            # Name of the config file
log_folder=sparse_zi_100        # Name of the Log Folder

python -u config/parallel.py \
       --seed ${seed} \
       --num_simulations ${num_simulations} \
       --num_parallel ${num_parallel} \
       --config ${config} \
       --log_folder ${log_folder}