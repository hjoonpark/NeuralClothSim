#!/bin/bash

set -e 
w
GPU_IDs=0
CONFIG_FILE="configs/smpl.json"
echo "Using configuration file: $CONFIG_FILE"

echo "========================================"
echo "Training starts "
echo "========================================"
python train.py --config "$CONFIG_FILE" --gpu_id "$GPU_IDs"

echo "========================================"
echo "All done"
echo "========================================"