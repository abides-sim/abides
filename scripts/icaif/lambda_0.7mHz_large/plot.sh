#!/bin/bash

NUM_JOBS=52  # number of jobs to run in parallel, may need to reduce to satisfy computational constraints

# Plot multiple seed experiment
cd realism && python -u impact_multiday_pov.py ../scripts/icaif/lambda_0.7mHz_large/lambda_0.7mHz_large_impact_multiday.json -n ${NUM_JOBS} -r && cd ..
