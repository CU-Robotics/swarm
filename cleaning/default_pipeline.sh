#!/bin/bash
if [ "$#" -le 0 ]; then
    echo "Usage: \$0 <full-path-to-project> [-l <log-level>]"
    exit 1
fi

PROJ_PATH="$1"
shift

LOG_ARG=()
if [ "$1" == "-l" ]; then
    if [ -n "$2" ]; then
        LOG_ARG=("-l" "$2")
    else
        echo "Error: -l requires a value (log-level)"
        exit 1
    fi
fi

# destroy existing pipeline data and run new pipeline
./destroy_pipeline_state.py
./pipeline.py -m "${LOG_ARG[@]}" -w "${PROJ_PATH}" \
    kill_green.py \
    undistort.py \
    plate_detector.py \
    label_bbox.py \ 
    crop.py
