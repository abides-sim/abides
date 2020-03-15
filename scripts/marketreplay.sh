#!/bin/bash

seed=123456789
#ticker=IBM
#dates=(20190628 20190627 20190626 20190625 20190624
#       20190621 20190620 20190619 20190618 20190617
#       20190614 20190613 20190612 20190611 20190610
#       20190607 20190606 20190605 20190604 20190603)

ticker=MSFT
dates=(20190604)
for d in ${dates[*]}
  do
    nohup python -u abides.py -c marketreplay -t $ticker -d $d -s $seed -l marketreplay_${ticker}_${d} &
    sleep 0.5
  done