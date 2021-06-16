#!/usr/bin/env bash

# dir to hold all streams

RETURN_STREAMS_DIR=return_streams

SYNTH_NAME=ABM
HIST_NAME=hist
MARKETREPLAY_NAME=marketreplay

# subdirs for synth, hist and marketreplay
SYNTH_DIR=${RETURN_STREAMS_DIR}/${SYNTH_NAME}
HIST_DIR=${RETURN_STREAMS_DIR}/${HIST_NAME}
MARKETREPLAY_DIR=${RETURN_STREAMS_DIR}/${MARKETREPLAY_NAME}

log_home_dir="/home/ec2-user/efs/_abides/dev/mm/abides-icaif/abides/log"
num_parallel_runs=16

num_random_samples=20

RANDOM_SAMPLE_DIR=return_streams_rand_${num_random_samples}
RANDOM_SYNTH_DIR=${RANDOM_SAMPLE_DIR}/${SYNTH_NAME}
RANDOM_HIST_DIR=${RANDOM_SAMPLE_DIR}/${HIST_NAME}
MARKETREPLAY_DIR=${RANDOM_SAMPLE_DIR}/${MARKETREPLAY_NAME}

mkdir -p ${RANDOM_SYNTH_DIR}
mkdir -p ${RANDOM_HIST_DIR}
mkdir -p ${MARKETREPLAY_DIR}

ls ${SYNTH_DIR}|sort -R |tail -${num_random_samples} | while read file; do
    # Something involving $file, or you can leave
    # off the while to just get the filenames
    cp -r ${SYNTH_DIR}/${file} ${RANDOM_SYNTH_DIR}/${file}
done

ls ${HIST_DIR}|sort -R |tail -${num_random_samples} | while read file; do
    # Something involving $file, or you can leave
    # off the while to just get the filenames
    cp -r ${HIST_DIR}/${file} ${RANDOM_HIST_DIR}/${file}
done

ls ${MARKETREPLAY_DIR}|sort -R |tail -${num_random_samples} | while read file; do
    # Something involving $file, or you can leave
    # off the while to just get the filenames
    cp -r ${MARKETREPLAY_DIR}/${file} ${RANDOM_MARKETREPLAY_DIR}/${file}
done