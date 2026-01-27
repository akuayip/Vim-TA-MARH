#!/bin/bash
# Run All OFAT Experiments Sequentially
# This will run all OFAT experiments one by one
# Total: 1 baseline + 8 variations = 9 experiments

set -e

echo "============================================"
echo "OFAT Fine-tuning Experiments for ViM-Tiny"
echo "Starting all experiments..."
echo "============================================"

# Define all experiments
experiments=(
    # BASELINE
    "baseline_lr0.001_bs8_adamw_ep100"
    
    # LR Variations (change LR, keep others baseline)
    "ofat_lr0.01_bs8_adamw_ep100"
    "ofat_lr0.005_bs8_adamw_ep100"
    
    # Batch Size Variations (change BS, keep others baseline)
    "ofat_lr0.001_bs16_adamw_ep100"
    "ofat_lr0.001_bs32_adamw_ep100"
    
    # Optimizer Variations (change optimizer, keep others baseline)
    "ofat_lr0.001_bs8_sgd_ep100"
    "ofat_lr0.001_bs8_rmsprop_ep100"
    
    # Epoch Variations (change epochs, keep others baseline)
    "ofat_lr0.001_bs8_adamw_ep150"
    "ofat_lr0.001_bs8_adamw_ep200"
)

total=${#experiments[@]}
current=0

for exp in "${experiments[@]}"; do
    current=$((current + 1))
    
    echo ""
    echo "============================================"
    echo "Experiment $current/$total: $exp"
    echo "============================================"
    
    # Run experiment
    bash run_single_ofat.sh "$exp"
    
    # Brief pause between experiments
    sleep 5
done

echo ""
echo "============================================"
echo "All OFAT Experiments Completed!"
echo "Results saved in ./work_dirs/ofat_*/"
echo "============================================"
