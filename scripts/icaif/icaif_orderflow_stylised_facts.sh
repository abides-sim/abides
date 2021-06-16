#!/usr/bin/env bash

#### Script to plot order stylised facts for ICAIF

stream_dir=streams_one_trace



# Plot orderflow stylised facts
cd realism
echo "Plotting order flow stylised facts for directory ${stream_dir}..."
mkdir -p ../${stream_dir}/plots
err_file=../${stream_dir}/order_flow_plot.log
python -u order_flow_stylized_facts.py --recompute ../${stream_dir} -o ../${stream_dir}/plots 2>&1> ${err_file}
cd ..


