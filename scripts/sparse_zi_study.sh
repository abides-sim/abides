#!/bin/bash

if [ $# -eq 0 ]; then
  echo $0: 'usage: sparse_zi_study.sh <simulation count>'
  exit 1
fi

if [ $# -ge 2 ]; then
  echo $0: 'usage: sparse_zi_study.sh <simulation count>'
  exit 1
fi

if [ $# -eq 1 ]; then
  count=$1
  dt=`date +%s`
  for i in `seq 1 ${count}`;
  do
    echo "Launching simulation $i"
    python -u abides.py -c sparse_zi -l sparse_zi_${i} -s ${i} > ./batch_output/sparse_zi_${i} &
    sleep 0.5
  done
fi
