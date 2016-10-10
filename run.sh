#!/bin/bash

if [ $# -lt 2 ]; then 
    echo "$0 data_dir"
    exit
fi

DATADIR="$1"
