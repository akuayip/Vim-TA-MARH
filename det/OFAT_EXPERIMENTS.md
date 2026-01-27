# OFAT Experiments for Vision Mamba (ViM-Tiny) Fine-Tuning

## Overview
One Factor At a Time (OFAT) experimental methodology for fine-tuning pretrained Vision Mamba on head detection task with frozen backbone.

**Objective**: Systematically evaluate the impact of different hyperparameters on fine-tuning performance while keeping the backbone frozen.

## Dataset Information
- **Name**: Custom Head Detection (COCO format)
- **Location**: `datasets/coco/`
- **Training Images**: 1,271
- **Validation Images**: 159
- **Test Images**: 159
- **Classes**: 1 (head)
- **Annotation Type**: Bounding boxes only (no masks)

## Model Architecture
- **Backbone**: Vision Mamba Tiny (ViM-tiny)
  - Embed Dim: 192
  - Depth: 24 layers
  - Pretrained Weights: `ckpts/vim_tiny_pretrained.pth`
  - **Status**: FROZEN (not trainable)
  
- **Feature Pyramid**: SimpleFeaturePyramid
  - Multi-scale feature extraction (P2-P6)
  
- **Detection Head**: Cascade R-CNN
  - 3-stage cascade
  - Fully trainable

## Baseline Configuration
- **Learning Rate**: 0.001 (1e-3)
- **Batch Size**: 2 (reduced from 8 to fit RTX 3070 8GB with Cascade R-CNN)
- **Optimizer**: AdamW
- **Epochs**: 100
- **Iterations**: 63,600 (636 iters/epoch × 100 epochs)
- **LR Schedule**: MultiStep decay at 70% and 90% of training
- **Warmup**: 500 iterations
- **GPU Memory**: ~4.5 GB (safe for 8GB GPU)
- **Training Time**: ~5-6 hours per 100 epochs

## OFAT Experimental Design

### Factor 1: Learning Rate (2 variations)
| Experiment | LR | Batch Size | Optimizer | Epochs | Config File |
|-----------|----|-----------|-----------| -------|-------------|
| Baseline  | 0.001 | 8 | AdamW | 100 | `baseline_lr0.001_bs8_adamw_ep100.py` |
| Variation 1 | **0.01** | 8 | AdamW | 100 | `ofat_lr0.01_bs8_adamw_ep100.py` |
| Variation 2 | **0.005** | 8 | AdamW | 100 | `ofat_lr0.005_bs8_adamw_ep100.py` |

**Hypothesis**: Higher LR (0.01) might converge faster but risk instability. Medium LR (0.005) might balance convergence speed and stability.

---

### Factor 2: Batch Size (2 variations)
| Experiment | LR | Batch Size | Optimizer | Epochs | Iterations | Config File |
|-----------|----|-----------|-----------| -------|------------|-------------|
| Baseline  | 0.001 | 8 | AdamW | 100 | 15,900 | `baseline_lr0.001_bs8_adamw_ep100.py` |
| Variation 1 | 0.001 | **16** | AdamW | 100 | 8,000 | `ofat_lr0.001_bs16_adamw_ep100.py` |
| Variation 2 | 0.001 | **32** | AdamW | 100 | 4,000 | `ofat_lr0.001_bs32_adamw_ep100.py` |

**Hypothesis**: Larger batch sizes might provide more stable gradients but require more GPU memory. BS=32 may be challenging for RTX 3070 8GB.

---

### Factor 3: Optimizer (2 variations)
| Experiment | LR | Batch Size | Optimizer | Epochs | Config File |
|-----------|----|-----------|-----------| -------|-------------|
| Baseline  | 0.001 | 8 | AdamW | 100 | `baseline_lr0.001_bs8_adamw_ep100.py` |
| Variation 1 | 0.001 | 8 | **SGD** | 100 | `ofat_lr0.001_bs8_sgd_ep100.py` |
| Variation 2 | 0.001 | 8 | **RMSProp** | 100 | `ofat_lr0.001_bs8_rmsprop_ep100.py` |

**Optimizer Configurations**:
- **AdamW**: lr=0.001, betas=(0.9, 0.999), weight_decay=0.1
- **SGD**: lr=0.001, momentum=0.9, weight_decay=1e-4
- **RMSProp**: lr=0.001, alpha=0.99, eps=1e-8, weight_decay=1e-4

**Hypothesis**: AdamW is adaptive and might converge faster. SGD might find better minima but require more epochs. RMSProp balances both.

---

### Factor 4: Training Epochs (2 variations)
| Experiment | LR | Batch Size | Optimizer | Epochs | Iterations | Config File |
|-----------|----|-----------|-----------| -------|------------|-------------|
| Baseline  | 0.001 | 8 | AdamW | 100 | 15,900 | `baseline_lr0.001_bs8_adamw_ep100.py` |
| Variation 1 | 0.001 | 8 | AdamW | **150** | 23,850 | `ofat_lr0.001_bs8_adamw_ep150.py` |
| Variation 2 | 0.001 | 8 | AdamW | **200** | 31,800 | `ofat_lr0.001_bs8_adamw_ep200.py` |

**Hypothesis**: More epochs allow better fine-tuning of detection head, but may lead to overfitting on small dataset (1271 images).

---

## Experiment Summary Table

| # | Experiment Name | LR | BS | Optimizer | Epochs | Iters | Output Dir |
|---|----------------|----|----|-----------|--------|-------|-----------|
| 1 | **BASELINE** | 0.001 | 8 | AdamW | 100 | 15,900 | `work_dirs/ofat_baseline_lr0.001_bs8_adamw_ep100/` |
| 2 | LR Variation 1 | **0.01** | 8 | AdamW | 100 | 15,900 | `work_dirs/ofat_lr0.01_bs8_adamw_ep100/` |
| 3 | LR Variation 2 | **0.005** | 8 | AdamW | 100 | 15,900 | `work_dirs/ofat_lr0.005_bs8_adamw_ep100/` |
| 4 | BS Variation 1 | 0.001 | **16** | AdamW | 100 | 8,000 | `work_dirs/ofat_lr0.001_bs16_adamw_ep100/` |
| 5 | BS Variation 2 | 0.001 | **32** | AdamW | 100 | 4,000 | `work_dirs/ofat_lr0.001_bs32_adamw_ep100/` |
| 6 | Optim Variation 1 | 0.001 | 8 | **SGD** | 100 | 15,900 | `work_dirs/ofat_lr0.001_bs8_sgd_ep100/` |
| 7 | Optim Variation 2 | 0.001 | 8 | **RMSProp** | 100 | 15,900 | `work_dirs/ofat_lr0.001_bs8_rmsprop_ep100/` |
| 8 | Epoch Variation 1 | 0.001 | 8 | AdamW | **150** | 23,850 | `work_dirs/ofat_lr0.001_bs8_adamw_ep150/` |
| 9 | Epoch Variation 2 | 0.001 | 8 | AdamW | **200** | 31,800 | `work_dirs/ofat_lr0.001_bs8_adamw_ep200/` |

**Total Experiments**: 9 (1 baseline + 8 variations)

---

## Running Experiments

### Prerequisites
1. Ensure pretrained weights exist: `ckpts/vim_tiny_pretrained.pth`
2. Dataset registered: Run will auto-register via `register_custom_coco.py`
3. GPU available (CUDA-enabled)

### Clean and Restart Training

If you need to stop current training and clean all outputs:

```bash
cd det/

# Stop all training processes
pkill -9 -f lazyconfig_train_net.py

# Clean all outputs and logs
rm -rf work_dirs/ofat_* && \
rm -f /tmp/train*.log nohup.out && \
rm -f datasets/coco/train/_annotations.coco_filtered.json && \
rm -f datasets/coco/valid/_annotations.coco_filtered.json && \
rm -f datasets/coco/test/_annotations.coco_filtered.json

# Verify cleanup
echo "✓ Cleaned!" && ls work_dirs/
```

### Run Single Experiment

**Option 1: Run in foreground (see output live)**
```bash
cd det/
./run_single_ofat.sh baseline_lr0.001_bs8_adamw_ep100
```

**Option 2: Run in background**
```bash
cd det/
nohup ./run_single_ofat.sh baseline_lr0.001_bs8_adamw_ep100 > /tmp/train_baseline.log 2>&1 &

# Monitor progress
tail -f work_dirs/ofat_baseline_lr0.001_bs8_adamw_ep100/log.txt
```

**Check if training is running:**
```bash
# Check process
ps aux | grep lazyconfig_train_net | grep -v grep

# Check GPU usage
nvidia-smi

# View latest iterations
tail -20 work_dirs/ofat_baseline_lr0.001_bs8_adamw_ep100/log.txt
```

### Run All Experiments Sequentially
```bash
cd det/
./run_all_ofat.sh
```

This will run all 9 experiments one by one. **Estimated total time**: ~24-48 hours depending on GPU.

---

## Results Tracking

### Metrics to Track
For each experiment, monitor:
- **Training Loss**: Should decrease consistently
- **Validation mAP@0.5**: Primary metric
- **Validation mAP@0.5:0.95**: Secondary metric
- **Training Time**: Compare efficiency
- **GPU Memory Usage**: Important for batch size experiments

### Expected Outputs
Each experiment will generate:
```
work_dirs/ofat_<experiment_name>/
├── events.out.tfevents.*      # TensorBoard logs
├── log.txt                      # Training logs
├── model_final.pth              # Final trained model
├── model_0001000.pth            # Checkpoints every epoch
├── model_0002000.pth
├── ...
└── metrics.json                 # Evaluation metrics
```

### Results Comparison Template

| Experiment | mAP@0.5 | mAP@0.5:0.95 | Training Time | Best Epoch | Notes |
|------------|---------|--------------|---------------|------------|-------|
| Baseline (LR=0.001, BS=8, AdamW, 100ep) | | | | | |
| LR=0.01 | | | | | |
| LR=0.005 | | | | | |
| BS=16 | | | | | |
| BS=32 | | | | | |
| SGD | | | | | |
| RMSProp | | | | | |
| Epochs=150 | | | | | |
| Epochs=200 | | | | | |

---

## Analysis Guidelines

### 1. Learning Rate Analysis
- Compare convergence speed and stability
- Check if higher LR causes oscillation
- Identify optimal LR for detection head fine-tuning

### 2. Batch Size Analysis
- Compare GPU memory usage
- Evaluate gradient stability vs. training speed
- Check if larger batches improve generalization

### 3. Optimizer Analysis
- Compare convergence patterns
- Evaluate final performance
- Consider training stability

### 4. Epoch Analysis
- Identify overfitting point
- Determine optimal training duration
- Check validation performance plateau

---

## GPU Memory Considerations
**IMPORTANT**: Due to Cascade R-CNN memory requirements with 640x640 images, actual batch sizes differ from planned:

Actual Memory Usage (with frozen backbone):
- **BS=2**: ~4.5 GB ✅ (SAFE - use for baseline)
- **BS=4**: ~7.5 GB ✅ (may work with gradient checkpointing)
- **BS=8**: ~10+ GB ❌ (OOM - needs larger GPU)
- **BS=16**: ~15+ GB ❌ (requires cloud GPU with 16GB+)
- **BS=32**: ~25+ GB ❌ (requires A100 40GB or similar)

**Recommendation**: 
- Run all experiments with BS=2 on RTX 3070 8GB
- For BS=4, 8, 16, 32 experiments, use cloud platform (AWS p3.2xlarge, Lambda Labs, etc.)
- Alternative: Use gradient accumulation to simulate larger batch sizes

**Training Configuration Fixes Applied**:
1. ✅ Safe dictionary pop for rope keys (vim.py)
2. ✅ Image size matched to dataset (640x640)
3. ✅ Position interpolation for checkpoint loading (utils.py)
4. ✅ Category mapping filtered to single class (register_custom_coco.py)
5. ✅ num_classes properly configured in roi_heads
6. ✅ Batch size reduced to BS=2 for baseline

**Recommendation**: Start with BS=8 and BS=16 locally. Run BS=32 on cloud platform with larger GPU.

---

## Next Steps After OFAT
1. **Identify Best Hyperparameters**: Select optimal LR, BS, optimizer, epochs
2. **Combined Optimization**: Run final experiment with best combination
3. **Error Analysis**: Analyze failure cases
4. **Model Deployment**: Export best model for inference

---

## Files Structure
```
det/
├── register_custom_coco.py                    # Dataset registration
├── run_single_ofat.sh                         # Run one experiment
├── run_all_ofat.sh                            # Run all experiments
├── projects/ViTDet/configs/OFAT/
│   ├── baseline_lr0.001_bs8_adamw_ep100.py   # Baseline
│   ├── ofat_lr0.01_bs8_adamw_ep100.py        # LR variations
│   ├── ofat_lr0.005_bs8_adamw_ep100.py
│   ├── ofat_lr0.001_bs16_adamw_ep100.py      # Batch size variations
│   ├── ofat_lr0.001_bs32_adamw_ep100.py
│   ├── ofat_lr0.001_bs8_sgd_ep100.py         # Optimizer variations
│   ├── ofat_lr0.001_bs8_rmsprop_ep100.py
│   ├── ofat_lr0.001_bs8_adamw_ep150.py       # Epoch variations
│   └── ofat_lr0.001_bs8_adamw_ep200.py
├── tools/lazyconfig_train_net.py             # Modified training script
└── work_dirs/                                 # Output directory
```

---

## Contact & Support
For questions or issues, please refer to:
- Vision Mamba paper: https://arxiv.org/abs/2401.09417
- Detectron2 docs: https://detectron2.readthedocs.io/
- Project README: `../README.md`

---

**Document Version**: 1.0  
**Last Updated**: 2025-02-02  
**Author**: MARH Research Team
