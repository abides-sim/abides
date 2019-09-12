#!/bin/bash

if [ $# -eq 0 ]; then
    echo $0: usage: dump.sh agent_filename
    exit 1
fi

if [ $# -eq 1 ]; then
  file=$1
  python cli/dump.py "$(ls -at log/15*/${file}* | head -n 1)"
fi

if [ $# -ge 2 ]; then
  file=$1
  python cli/dump.py "$(ls -at log/15*/${file}* | head -n 1)" "${@:2}"
fi
