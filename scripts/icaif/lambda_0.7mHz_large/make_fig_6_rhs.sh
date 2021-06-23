#!/usr/bin/env bash

### RUN THIS SCRIPT TO GENERATE ICAIF GetReal Fig 6 RHS plot
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
${SCRIPT_DIR}/execution_iabs_seed_search.sh
${SCRIPT_DIR}/rename.sh
${SCRIPT_DIR}/plot.sh