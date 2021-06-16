#!/usr/bin/env bash

# dir to hold all streams

base_stream_dir=return_streams_rand_20

# subdirs for synth, hist and marketreplay
#synth_streams=${base_stream_dir}/icaif_synth
#hist_streams=${base_stream_dir}/icaif_hist
#marketreplay_streams=${base_stream_dir}/marketreplay

synth_streams=${base_stream_dir}/ABM
hist_streams=${base_stream_dir}/hist
marketreplay_streams=${base_stream_dir}/marketreplay

mkdir -p ${base_stream_dir}/plots
python -u realism/asset_returns_stylized_facts.py -s ${synth_streams} -s ${hist_streams} -s ${marketreplay_streams} --recompute -o ${base_stream_dir}/plots
