#!/bin/bash

NUM_JOBS=48  # number of jobs to run in parallel, may need to reduce to satisfy computational constraints

for seed in $(seq 100 130); do
      sem -j${NUM_JOBS} --line-buffer python -u abides.py -c exp_agent_demo -t ABM -d 20200603 -s ${seed} \
       -l experimental_agent_demo_short_1s_long_30s_${seed} -e --ea-short-window '1s' --ea-long-window '30s'
done

for seed in $(seq 100 130); do
      sem -j${NUM_JOBS} --line-buffer python -u abides.py -c exp_agent_demo -t ABM -d 20200603 -s ${seed} \
       -l experimental_agent_demo_short_2min_long_5min_${seed} -e --ea-short-window '2min' --ea-long-window '5min'
done
sem --wait

python -u cli/read_agent_logs.py log/experimental_agent_demo_short_1s_long_30s_*

python -u cli/read_agent_logs.py log/experimental_agent_demo_short_2min_long_5min_*
