#!/bin/bash

seed=123456789
config=rmsc01
log=rmsc01

if [ $# -eq 0 ]; then
    python -u abides.py -c $config -l $log -s $seed
else
  agent=$1
  python -u abides.py -c $config -l $log -s $seed -a $agent 
fi

