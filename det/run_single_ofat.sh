#!/bin/bash
# Run Single OFAT Experiment
# Usage: ./run_single_ofat.sh <config_name>
# Example: ./run_single_ofat.sh baseline_lr0.001_bs8_adamw_ep100

set -e

if [ -z "$1" ]; then
    echo "Usage: ./run_single_ofat.sh <config_name>"
    echo "Available configs:"
    ls projects/ViTDet/configs/OFAT/*.py | xargs -n 1 basename
    exit 1
fi

CONFIG_NAME=$1
CONFIG_PATH="projects/ViTDet/configs/OFAT/${CONFIG_NAME}.py"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found: $CONFIG_PATH"
    exit 1
fi

echo "========================================="
echo "Running OFAT Experiment: $CONFIG_NAME"
echo "Config: $CONFIG_PATH"
echo "========================================="

# Run training
python tools/lazyconfig_train_net.py \
    --config-file "$CONFIG_PATH" \
    --num-gpus 1

echo "========================================="
echo "Experiment completed: $CONFIG_NAME"
echo "========================================="
