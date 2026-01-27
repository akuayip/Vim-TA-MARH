# Vision Mamba Detection - Environment Setup Guide

## System Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (minimum 8GB VRAM, 16GB+ recommended)
  - Tested on: RTX 3070 Laptop (8GB), T4 (16GB), V100 (16GB)
- **RAM**: Minimum 16GB recommended
- **Storage**: ~20GB for environment and checkpoints

### Software
- **OS**: Ubuntu 20.04/22.04 or similar Linux distribution
- **NVIDIA Driver**: ≥ 520.61.05 (compatible with CUDA 11.8)
- **CUDA**: 11.8 (compilation tools required)
- **Python**: 3.9.x

## Tested Environment Specifications

```
Python: 3.9.19
CUDA: 11.8 (nvcc V11.8.89)
cuDNN: 8700
PyTorch: 2.1.1+cu118
NumPy: 1.26.4
Detectron2: 0.6
NVIDIA Driver: ≥ 520.61.05
```

**Note**: This setup works on various NVIDIA GPUs including RTX series, Tesla T4, V100, A10, A100, etc.

---

## Installation

### 1. CUDA 11.8

```bash
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run --toolkit --silent --override

# Add to ~/.bashrc
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
source ~/.bashrc

nvcc --version  # verify
```

### 2. Environment Setup

**Option A: Conda (Recommended)**
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
conda create -n vimdet python=3.9 -y
conda activate vimdet
```

**Option B: Python venv (No Conda)**
```bash
sudo apt install python3.9 python3.9-venv python3.9-dev -y
cd ~/Vim-TA-MARH
python3.9 -m venv venv
source venv/bin/activate
```

### 3. Install Packages

```bash
# Clone repo (if needed)
git clone <your-repo-url> Vim-TA-MARH && cd Vim-TA-MARH

# PyTorch + CUDA 11.8
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118

# Core dependencies (NumPy <2.0 critical!)
pip install "numpy<2.0" opencv-python timm einops fvcore pycocotools matplotlib scipy Pillow packaging ninja

# Mamba SSM (needs --no-build-isolation for CUDA compilation)
cd mamba-1p1p1 && pip install -e . --no-build-isolation && cd ..

# Causal Conv1D (needs --no-build-isolation for CUDA compilation)
cd causal-conv1d && pip install -e . --no-build-isolation && cd ..

# Detectron2 (needs --no-build-isolation)
cd det && pip install -e . --no-build-isolation && cd ..
```

### 4. Verify Installation

```bash
python -c "import torch, detectron2, mamba_ssm, causal_conv1d; print(f'PyTorch {torch.__version__} | CUDA {torch.cuda.is_available()} | Detectron2 {detectron2.__version__}')"
```

Expected: `PyTorch 2.1.1+cu118 | CUDA True | Detectron2 0.6`

---

## Dataset & Weights

```bash
# Organize dataset (COCO format)
det/datasets/coco/
├── annotations/instances_{train,val}.json
└── images/{train,val}/

# Download pretrained weights to det/ckpts/vim_tiny_pretrained.pth
```

## Training

```bash
# Activate environment
conda activate vimdet  # or: source venv/bin/activate

# Run training
cd det
python tools/train_net.py \
    --config-file configs/COCO-Detection/cascade_rcnn_vim_fpn.yaml \
    --num-gpus 1 \
    OUTPUT_DIR work_dirs/experiment_name \
    SOLVER.IMS_PER_BATCH 2 \
    SOLVER.BASE_LR 0.001 \
    SOLVER.MAX_ITER 63600 \
    MODEL.WEIGHTS ckpts/vim_tiny_pretrained.pth

# Monitor
tail -f work_dirs/experiment_name/log.txt
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| NumPy 2.0 error | `pip install "numpy<2.0"` |
| CUDA OOM | Reduce `SOLVER.IMS_PER_BATCH` to 1 |
| Mamba build fails | Check `nvcc --version`, ensure CUDA in PATH |
| Import errors | Verify all 3 source packages installed: mamba_ssm, causal_conv1d, detectron2 |

---

## Cloud Setup (Quick)

```bash
# On fresh Ubuntu GPU instance
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

# Conda
conda create -n vimdet python=3.9 -y && conda activate vimdet

# OR venv
python3.9 -m venv ~/venv && source ~/venv/bin/activate

# Install all
pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu118
pip install "numpy<2.0" opencv-python timm einops fvcore pycocotools ninja packaging
cd ~/Vim-TA-MARH && cd mamba-1p1p1 && pip install -e . --no-build-isolation && cd ../causal-conv1d && pip install -e . --no-build-isolation && cd ../det && pip install -e . --no-build-isolation

# Run in tmux
tmux new -s train
python tools/train_net.py --config-file ... 
# Detach: Ctrl+B then D
```

---

**Key Notes**: NumPy must be <2.0 | CUDA 11.8 required | Install mamba/causal-conv1d/detectron2 from source
