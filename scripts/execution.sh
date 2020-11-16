#!/bin/bash

seed=123456789
ticker=ABM
date=20200101

log=execution_${seed}

# (1) Sparse Mean Reverting Oracle Fundamental
python -u abides.py -c execution -t ${ticker} -d ${date} -l ${log} -s ${seed} -e

# (2) External Fundamental file
#fundamental_path=${PWD}/data/synthetic_fundamental.bz2
#python -u abides.py -c execution -t ${ticker} -d ${date} -f ${fundamental_path} -l ${log} -s ${seed} -e