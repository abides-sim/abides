#!/bin/bash

if [ $# -eq 0 ]; then
  echo $0: 'usage: impact_baseline.sh <simulation count>'
  exit 1
fi

if [ $# -ge 2 ]; then
  echo $0: 'usage: impact_baseline.sh <simulation count>'
  exit 1
fi

if [ $# -eq 1 ]; then
  count=$1
  dt=`date +%s`
  for i in `seq 1 ${count}`;
  do
    echo "Launching simulation $i"
    python -u abides.py -c impact -i -l impact_${i}_baseline -s ${i} > ./batch_output/impact_${i}_baseline &
    sleep 0.5
  done
fi
