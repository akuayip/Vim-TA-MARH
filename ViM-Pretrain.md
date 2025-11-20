# 📘 Panduan Lengkap: ViM Pre-trained untuk Object Detection

> **Vision Mamba (ViM)** - Efficient Visual Representation Learning with Bidirectional State Space Model untuk Object Detection menggunakan Detectron2

---

## 📋 Daftar Isi

1. [Persyaratan Sistem](#1-persyaratan-sistem)
2. [Git Clone Repository](#2-git-clone-repository)
3. [Persiapan Environment](#3-persiapan-environment)
4. [Download Pre-trained Model](#4-download-pre-trained-model)
5. [Persiapan Dataset](#5-persiapan-dataset)
6. [Konfigurasi dan Hyperparameter](#6-konfigurasi-dan-hyperparameter)
7. [Training (Fine-tuning)](#7-training-fine-tuning)
8. [Evaluation](#8-evaluation)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Persyaratan Sistem

### Hardware Requirements

- **GPU**: NVIDIA GPU dengan minimal 8GB VRAM (Recommended: 16GB+)
  - Untuk training: RTX 3090, A100, V100, atau lebih tinggi
  - Untuk inference: RTX 2060 atau lebih tinggi
- **RAM**: Minimal 16GB (Recommended: 32GB+)
- **Storage**: Minimal 100GB ruang kosong

### Software Requirements

- **OS**: Linux (Ubuntu 18.04/20.04/22.04) atau macOS
  - Windows dengan WSL2 juga dapat digunakan
- **Python**: 3.9 - 3.10.13
- **CUDA**: 11.1 / 11.3 / 11.8 (sesuaikan dengan GPU)
- **PyTorch**: ≥ 1.8 (Recommended: 2.1.1)
- **gcc & g++**: ≥ 5.4

---

## 2. Git Clone Repository

### Clone Repository Utama

```bash
# Clone repository ViM
git clone https://github.com/hustvl/Vim.git
cd Vim

# Atau jika menggunakan repository custom Anda
git clone https://github.com/akuayip/Vim-TA-MARH.git
cd Vim-TA-MARH
```

### Struktur Folder Penting

```
Vim-TA-MARH/
├── vim/                    # Image classification & backbone training
│   ├── main.py            # Main training script
│   ├── models_mamba.py    # ViM model definition
│   └── scripts/           # Training scripts
├── det/                    # Object detection (Detectron2)
│   ├── detectron2/        # Detectron2 library
│   ├── projects/ViTDet/   # ViMDet configurations
│   ├── scripts/           # Training & evaluation scripts
│   └── tools/             # Training utilities
├── seg/                    # Semantic segmentation
├── mamba-1p1p1/           # Mamba SSM core
└── causal-conv1d/         # Causal convolution module
```

---

## 3. Persiapan Environment

### Step 1: Buat Conda Environment

```bash
# Buat environment baru dengan Python 3.9.19
conda create -n vimdet python=3.9.19
conda activate vimdet
```

### Step 2: Install PyTorch

**Untuk NVIDIA GPU (CUDA 11.8):**

```bash
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
```

**Untuk CUDA 11.3:**

```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

**Verifikasi instalasi PyTorch:**

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 3: Install Dependencies

**Install package dasar:**

```bash
# Navigate ke folder det
cd det

# Install dependencies dasar
pip install numpy==1.23.4
pip install iopath==0.1.9
pip install cloudpickle==2.2.1
pip install omegaconf==2.3.0
pip install Pillow==8.2.0
pip install requests==2.28.1
pip install sympy==12.1
pip install pycocotools==2.0.6
pip install Shapely==1.8.1
```

**Atau gunakan requirements file:**

```bash
pip install -r det-requirements.txt
```

### Step 4: Install Mamba & Causal Conv1D

```bash
# Kembali ke root directory
cd ..

# Install causal-conv1d
cd causal-conv1d
pip install -e .

# Install mamba
cd ../mamba-1p1p1
pip install -e .

# Verifikasi instalasi
python -c "import mamba_ssm; print('Mamba installed successfully')"
python -c "import causal_conv1d; print('Causal Conv1D installed successfully')"
```

### Step 5: Install Detectron2

```bash
cd ../det

# Install Detectron2 dari source
pip install -e .

# Atau install pre-built (untuk Linux saja)
# python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.1/index.html

# Verifikasi instalasi Detectron2
python -c "import detectron2; from detectron2 import model_zoo; print(f'Detectron2 version: {detectron2.__version__}')"
```

---

## 4. Download Pre-trained Model

### ViM Backbone Weights (ImageNet Pre-trained)

Pre-trained weights tersedia di **Hugging Face**:

| Model          | Parameters | Top-1 Acc | Hugging Face Link                                                               |
| -------------- | ---------- | --------- | ------------------------------------------------------------------------------- |
| **Vim-Tiny**   | 7M         | 76.1%     | [hustvl/Vim-tiny-midclstok](https://huggingface.co/hustvl/Vim-tiny-midclstok)   |
| **Vim-Tiny+**  | 7M         | 78.3%     | [hustvl/Vim-tiny-midclstok](https://huggingface.co/hustvl/Vim-tiny-midclstok)   |
| **Vim-Small**  | 26M        | 80.5%     | [hustvl/Vim-small-midclstok](https://huggingface.co/hustvl/Vim-small-midclstok) |
| **Vim-Small+** | 26M        | 81.6%     | [hustvl/Vim-small-midclstok](https://huggingface.co/hustvl/Vim-small-midclstok) |

### Download Script

```bash
# Buat folder untuk checkpoint
mkdir -p det/ckpts
cd det/ckpts

# Download Vim-Tiny (recommended untuk mulai)
wget https://huggingface.co/hustvl/Vim-tiny-midclstok/resolve/main/vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2.pth

# Atau download Vim-Small untuk akurasi lebih tinggi
wget https://huggingface.co/hustvl/Vim-small-midclstok/resolve/main/vim_small_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2.pth

# Rename untuk kemudahan
mv vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2.pth vim_tiny_pretrained.pth
```

**Atau download manual:**

1. Kunjungi: https://huggingface.co/hustvl/Vim-tiny-midclstok
2. Download file `.pth`
3. Simpan ke folder `det/ckpts/`

---

## 5. Persiapan Dataset

### Format Dataset: COCO Format

ViMDet menggunakan **COCO dataset format**. Untuk dataset custom, konversi ke format COCO.

### Struktur Folder Dataset

```bash
det/datasets/
└── coco/
    ├── annotations/
    │   ├── instances_train2017.json
    │   └── instances_val2017.json
    └── train2017/
        ├── 000000000001.jpg
        ├── 000000000002.jpg
        └── ...
    └── val2017/
        ├── 000000000001.jpg
        └── ...
```

### Option 1: Gunakan COCO Dataset (Standard)

```bash
# Buat folder datasets
cd det
mkdir -p datasets
cd datasets

# Download COCO 2017
# Train images (~18GB)
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip

# Val images (~1GB)
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip

# Annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip

# Cleanup
rm *.zip

# Struktur akhir:
# datasets/
# └── coco/
#     ├── annotations/
#     ├── train2017/
#     └── val2017/
```

### Option 2: Gunakan Dataset Custom

**Konversi dataset Anda ke format COCO:**

```python
# convert_to_coco.py
import json
from pathlib import Path

def create_coco_format(image_dir, output_json):
    """
    Buat annotation file dalam format COCO
    """
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    # Definisikan kategori (sesuaikan dengan dataset Anda)
    categories = [
        {"id": 1, "name": "person"},
        {"id": 2, "name": "car"},
        {"id": 3, "name": "bicycle"},
        # Tambahkan kategori lainnya
    ]
    coco_format["categories"] = categories

    # Tambahkan images dan annotations
    # ... (implementasi sesuai dataset Anda)

    # Save to JSON
    with open(output_json, 'w') as f:
        json.dump(coco_format, f, indent=2)

# Gunakan script
create_coco_format("path/to/images", "annotations/instances_train_custom.json")
```

**Atau gunakan symlink untuk dataset existing:**

```bash
cd det/datasets
mkdir coco
cd coco

# Link ke dataset yang sudah ada
ln -s /path/to/your/coco/dataset/annotations ./annotations
ln -s /path/to/your/coco/dataset/train2017 ./train2017
ln -s /path/to/your/coco/dataset/val2017 ./val2017
```

### Verifikasi Dataset

```python
# verify_dataset.py
from pycocotools.coco import COCO

# Load annotation file
coco = COCO('datasets/coco/annotations/instances_train2017.json')

# Print dataset info
print(f"Number of images: {len(coco.imgs)}")
print(f"Number of annotations: {len(coco.anns)}")
print(f"Categories: {coco.loadCats(coco.getCatIds())}")

# Visualize sample
import matplotlib.pyplot as plt
import cv2

img_id = list(coco.imgs.keys())[0]
img_info = coco.loadImgs(img_id)[0]
img_path = f"datasets/coco/train2017/{img_info['file_name']}"
img = cv2.imread(img_path)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
```

---

## 6. Konfigurasi dan Hyperparameter

### File Konfigurasi Utama

**Lokasi:** `det/projects/ViTDet/configs/COCO/cascade_mask_rcnn_vimdet_t_100ep_adj1.py`

### Parameter Penting untuk Fine-tuning

#### 1. **Model Configuration**

```python
# cascade_mask_rcnn_vimdet_t_100ep_adj1.py

# Path ke pre-trained weights
model.backbone.net.pretrained = "det/ckpts/vim_tiny_pretrained.pth"

# Model architecture
model.backbone.net.embed_dim = 192      # ViM-Tiny: 192, ViM-Small: 384
model.backbone.net.depth = 24           # Number of Mamba blocks
model.backbone.net.patch_size = 16      # Patch size untuk input
```

#### 2. **Input Resolution**

```python
# Dari Base-RCNN-FPN.yaml
INPUT:
  # Multi-scale training (short side)
  MIN_SIZE_TRAIN: (640, 672, 704, 736, 768, 800)
  MAX_SIZE_TRAIN: 1333

  # Test/inference resolution
  MIN_SIZE_TEST: 800
  MAX_SIZE_TEST: 1333

  # Untuk GPU terbatas, gunakan resolusi lebih kecil:
  # MIN_SIZE_TRAIN: (480, 512, 544, 576, 608, 640)
  # MAX_SIZE_TRAIN: 1066
```

#### 3. **Training Parameters**

```python
# Training configuration
train.max_iter = 90000              # Total iterations (~100 epochs untuk COCO)
train.eval_period = 5000            # Evaluate setiap N iterations
train.checkpointer.period = 5000    # Save checkpoint setiap N iterations
train.init_checkpoint = ""          # Kosongkan untuk fine-tuning dari backbone

# Learning rate schedule
lr_multiplier.scheduler.milestones = [80000, 88000]  # LR decay milestones
lr_multiplier.scheduler.gamma = 0.1                   # LR decay factor
lr_multiplier.warmup_length = 1000 / train.max_iter   # Warmup iterations
```

#### 4. **Optimizer Settings**

```python
# Optimizer configuration
optimizer.lr = 0.0001                    # Base learning rate
optimizer.weight_decay = 0.05            # Weight decay untuk regularization

# Layer-wise learning rate decay
from functools import partial
optimizer.params.lr_factor_func = partial(
    get_vim_lr_decay_rate,
    num_layers=24,              # Sesuaikan dengan model depth
    lr_decay_rate=0.837         # Decay rate per layer
)
```

#### 5. **Data Augmentation**

```python
# Data loading
dataloader.train.total_batch_size = 16   # Batch size (adjust untuk GPU)
dataloader.train.num_workers = 4         # Jumlah worker untuk data loading

# Augmentation (defined in base config)
# - ResizeShortestEdge: Multi-scale resize
# - RandomFlip: Horizontal flip
# - RandomCrop: Random cropping (optional)
```

### Hyperparameter Tuning Guide

#### **Untuk GPU dengan VRAM Terbatas (<12GB)**

```python
# Reduce batch size
dataloader.train.total_batch_size = 8  # atau 4

# Reduce input resolution
INPUT.MIN_SIZE_TRAIN = (480, 512, 544, 576, 608, 640)
INPUT.MAX_SIZE_TRAIN = 1066

# Enable gradient checkpointing (jika tersedia)
# model.backbone.net.use_checkpoint = True
```

#### **Untuk Dataset Kecil (<10K images)**

```python
# Reduce iterations
train.max_iter = 30000  # ~50 epochs

# Increase evaluation frequency
train.eval_period = 1000

# Stronger regularization
optimizer.weight_decay = 0.1
optimizer.lr = 0.00005  # Lower learning rate

# Add more augmentation
# - MixUp
# - CutOut
# - ColorJitter
```

#### **Untuk Dataset Besar (>100K images)**

```python
# Extend training
train.max_iter = 180000  # ~200 epochs

# Standard settings work well
optimizer.lr = 0.0001
dataloader.train.total_batch_size = 16
```

### Contoh Konfigurasi Custom

**Buat file:** `det/projects/ViTDet/configs/COCO/cascade_mask_rcnn_vimdet_t_custom.py`

```python
from functools import partial
from .cascade_mask_rcnn_vimdet_b_100ep import (
    dataloader,
    lr_multiplier,
    model,
    train,
    optimizer,
    get_vim_lr_decay_rate,
)

# ========== Model Configuration ==========
model.backbone.net.embed_dim = 192
model.backbone.net.depth = 24
model.backbone.net.pretrained = "det/ckpts/vim_tiny_pretrained.pth"

# ========== Dataset Configuration ==========
# Gunakan dataset custom Anda
dataloader.train.dataset.names = "your_custom_train"
dataloader.test.dataset.names = "your_custom_val"

# ========== Training Configuration ==========
train.init_checkpoint = ""
train.max_iter = 30000          # Sesuaikan dengan dataset size
train.eval_period = 2000
train.checkpointer.period = 2000
train.output_dir = "./output/vimdet_tiny_custom"

# ========== Optimizer Configuration ==========
optimizer.lr = 0.00008          # Slightly lower LR untuk fine-tuning
optimizer.weight_decay = 0.05
optimizer.params.lr_factor_func = partial(
    get_vim_lr_decay_rate,
    num_layers=24,
    lr_decay_rate=0.837
)

# ========== Learning Rate Schedule ==========
lr_multiplier.scheduler.milestones = [24000, 28000]
lr_multiplier.scheduler.gamma = 0.1
lr_multiplier.warmup_length = 500 / train.max_iter

# ========== Data Loading ==========
dataloader.train.total_batch_size = 8   # Adjust untuk GPU
dataloader.train.num_workers = 4
```

---

## 7. Training (Fine-tuning)

### Method 1: Gunakan Script Yang Tersedia

**Edit script:** `det/scripts/ft_vim_tiny_vimdet.sh`

```bash
#!/bin/bash

# Pastikan sudah berada di folder det
cd det

# Fine-tune ViMDet-Tiny dengan pre-trained weights
python tools/lazyconfig_train_net.py \
  --config-file projects/ViTDet/configs/COCO/cascade_mask_rcnn_vimdet_t_100ep_adj1.py \
  --num-gpus 1 \
  train.init_checkpoint="ckpts/vim_tiny_pretrained.pth" \
  train.output_dir="./output/vimdet_tiny_ft"
```

**Jalankan training:**

```bash
cd det
bash scripts/ft_vim_tiny_vimdet.sh
```

### Method 2: Training Command Manual

**Single GPU:**

```bash
cd det

python tools/lazyconfig_train_net.py \
  --config-file projects/ViTDet/configs/COCO/cascade_mask_rcnn_vimdet_t_100ep_adj1.py \
  --num-gpus 1 \
  train.init_checkpoint="ckpts/vim_tiny_pretrained.pth" \
  train.output_dir="./output/vimdet_tiny_coco" \
  train.max_iter=90000
```

**Multi-GPU (4 GPUs):**

```bash
python tools/lazyconfig_train_net.py \
  --config-file projects/ViTDet/configs/COCO/cascade_mask_rcnn_vimdet_t_100ep_adj1.py \
  --num-gpus 4 \
  train.init_checkpoint="ckpts/vim_tiny_pretrained.pth" \
  train.output_dir="./output/vimdet_tiny_coco_4gpu"
```

**Dengan Custom Dataset:**

```bash
python tools/lazyconfig_train_net.py \
  --config-file projects/ViTDet/configs/COCO/cascade_mask_rcnn_vimdet_t_custom.py \
  --num-gpus 1 \
  dataloader.train.dataset.names="my_custom_train" \
  dataloader.test.dataset.names="my_custom_val" \
  train.output_dir="./output/vimdet_custom"
```

### Monitor Training

**TensorBoard (jika tersedia):**

```bash
# Install tensorboard
pip install tensorboard

# Jalankan tensorboard
tensorboard --logdir det/output/vimdet_tiny_ft

# Buka browser: http://localhost:6006
```

**Check Log File:**

```bash
# Lihat log real-time
tail -f det/output/vimdet_tiny_ft/log.txt

# Atau gunakan less
less det/output/vimdet_tiny_ft/log.txt
```

### Training Output Structure

```
det/output/vimdet_tiny_ft/
├── config.yaml              # Saved configuration
├── log.txt                  # Training logs
├── metrics.json             # Evaluation metrics
├── events.out.tfevents.*    # TensorBoard events
├── model_0004999.pth        # Checkpoint @5k iterations
├── model_0009999.pth        # Checkpoint @10k iterations
├── model_final.pth          # Final model
└── last_checkpoint          # Pointer to last checkpoint
```

### Resume Training

```bash
# Resume dari checkpoint terakhir
python tools/lazyconfig_train_net.py \
  --config-file projects/ViTDet/configs/COCO/cascade_mask_rcnn_vimdet_t_100ep_adj1.py \
  --num-gpus 1 \
  --resume \
  train.output_dir="./output/vimdet_tiny_ft"
```

---

## 8. Evaluation

### Method 1: Gunakan Script Evaluation

**Edit script:** `det/scripts/eval_vim_tiny_vimdet.sh`

```bash
#!/bin/bash

cd det

python tools/lazyconfig_train_net.py \
  --config-file projects/ViTDet/configs/COCO/cascade_mask_rcnn_vimdet_t_100ep_adj1.py \
  --eval-only \
  train.init_checkpoint="output/vimdet_tiny_ft/model_final.pth"
```

**Jalankan evaluation:**

```bash
cd det
bash scripts/eval_vim_tiny_vimdet.sh
```

### Method 2: Evaluation Manual

```bash
cd det

# Evaluate model final
python tools/lazyconfig_train_net.py \
  --config-file projects/ViTDet/configs/COCO/cascade_mask_rcnn_vimdet_t_100ep_adj1.py \
  --eval-only \
  train.init_checkpoint="output/vimdet_tiny_ft/model_final.pth"

# Evaluate checkpoint tertentu
python tools/lazyconfig_train_net.py \
  --config-file projects/ViTDet/configs/COCO/cascade_mask_rcnn_vimdet_t_100ep_adj1.py \
  --eval-only \
  train.init_checkpoint="output/vimdet_tiny_ft/model_0089999.pth"
```

### Evaluation Metrics

Output akan menampilkan:

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.XXX
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.XXX
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.XXX
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.XXX
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.XXX
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.XXX
```

### Inference Demo

**Jalankan inference pada gambar:**

```bash
cd det

python demo/demo.py \
  --config-file projects/ViTDet/configs/COCO/cascade_mask_rcnn_vimdet_t_100ep_adj1.py \
  --input /path/to/image.jpg \
  --output ./output/demo_results \
  --opts train.init_checkpoint="output/vimdet_tiny_ft/model_final.pth"
```

**Inference pada video:**

```bash
python demo/demo.py \
  --config-file projects/ViTDet/configs/COCO/cascade_mask_rcnn_vimdet_t_100ep_adj1.py \
  --video-input /path/to/video.mp4 \
  --output ./output/demo_video.mp4 \
  --opts train.init_checkpoint="output/vimdet_tiny_ft/model_final.pth"
```

**Inference dengan webcam:**

```bash
python demo/demo.py \
  --config-file projects/ViTDet/configs/COCO/cascade_mask_rcnn_vimdet_t_100ep_adj1.py \
  --webcam \
  --opts train.init_checkpoint="output/vimdet_tiny_ft/model_final.pth"
```

---

## 9. Troubleshooting

### Common Issues & Solutions

#### 1. **CUDA Out of Memory**

**Error:**

```
RuntimeError: CUDA out of memory
```

**Solutions:**

```python
# 1. Reduce batch size
dataloader.train.total_batch_size = 4  # atau 2

# 2. Reduce input resolution
INPUT.MIN_SIZE_TRAIN = (480, 512, 544)
INPUT.MAX_SIZE_TRAIN = 800

# 3. Use gradient accumulation
train.gradient_accumulation_steps = 2
```

#### 2. **Mamba Installation Failed**

**Error:**

```
error: command 'gcc' failed with exit status 1
```

**Solutions:**

```bash
# Install build tools
sudo apt-get install build-essential

# Update gcc/g++
sudo apt-get install gcc-9 g++-9
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 90

# Install with verbose
pip install -e mamba-1p1p1 -v
```

#### 3. **Detectron2 Import Error**

**Error:**

```
ModuleNotFoundError: No module named 'detectron2'
```

**Solutions:**

```bash
# Reinstall detectron2
cd det
pip uninstall detectron2
pip install -e .

# Or use pre-built
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.1/index.html
```

#### 4. **Dataset Not Found**

**Error:**

```
FileNotFoundError: Dataset 'coco_2017_train' not found
```

**Solutions:**

```bash
# Register dataset di detectron2
# Edit detectron2/data/datasets/builtin.py atau buat file baru

from detectron2.data.datasets import register_coco_instances

register_coco_instances(
    "my_dataset_train",
    {},
    "det/datasets/coco/annotations/instances_train2017.json",
    "det/datasets/coco/train2017"
)

register_coco_instances(
    "my_dataset_val",
    {},
    "det/datasets/coco/annotations/instances_val2017.json",
    "det/datasets/coco/val2017"
)
```

#### 5. **Pre-trained Weights Mismatch**

**Error:**

```
RuntimeError: Error(s) in loading state_dict
```

**Solutions:**

```python
# Option 1: Ignore missing keys (untuk backbone saja)
train.init_checkpoint = "ckpts/vim_tiny_pretrained.pth"
# Pastikan hanya load backbone weights

# Option 2: Manual loading dengan strict=False
import torch
checkpoint = torch.load("ckpts/vim_tiny_pretrained.pth")
model.load_state_dict(checkpoint, strict=False)
```

#### 6. **Slow Training Speed**

**Solutions:**

```python
# 1. Increase num_workers
dataloader.train.num_workers = 8

# 2. Use mixed precision training
train.amp.enabled = True

# 3. Optimize data loading
dataloader.train.prefetch_factor = 2
```

### Performance Tips

1. **Multi-GPU Training:** Gunakan `--num-gpus N` untuk training lebih cepat
2. **Mixed Precision:** Enable AMP untuk speedup ~2x
3. **Data Loading:** Increase `num_workers` sesuai CPU cores
4. **Checkpoint:** Save checkpoint lebih jarang untuk mengurangi I/O
5. **Evaluation:** Reduce `eval_period` saat debugging

---

## 📊 Expected Results

### ViM-Tiny pada COCO val2017

```
AP (IoU=0.50:0.95): ~48.0
AP50:                ~66.5
AP75:                ~52.0
APs (small):         ~31.0
APm (medium):        ~50.5
APl (large):         ~62.0
```

### Training Time Estimation

| Configuration | Hardware     | Time per Epoch | Total Time (100 epochs) |
| ------------- | ------------ | -------------- | ----------------------- |
| Single V100   | 1x V100 32GB | ~4 hours       | ~400 hours              |
| 4x V100       | 4x V100 32GB | ~1 hour        | ~100 hours              |
| Single A100   | 1x A100 40GB | ~2.5 hours     | ~250 hours              |
| 8x A100       | 8x A100 40GB | ~30 min        | ~50 hours               |

---

## 📚 Additional Resources

### Official Documentation

- **ViM Paper:** https://arxiv.org/abs/2401.09417
- **Detectron2 Docs:** https://detectron2.readthedocs.io/
- **Mamba Paper:** https://arxiv.org/abs/2312.00752

### Useful Links

- **Hugging Face Models:** https://huggingface.co/hustvl
- **GitHub Issues:** https://github.com/hustvl/Vim/issues
- **COCO Dataset:** https://cocodataset.org/

### Community Support

- Open issue di GitHub repository
- Join Discord/Slack community (jika ada)
- Stack Overflow dengan tag `detectron2` dan `vision-mamba`

---

## 📝 Quick Start Checklist

- [ ] Install CUDA & PyTorch
- [ ] Clone repository
- [ ] Create conda environment
- [ ] Install dependencies (Mamba, Detectron2)
- [ ] Download pre-trained weights
- [ ] Prepare dataset (COCO format)
- [ ] Verify dataset dengan script
- [ ] Edit konfigurasi sesuai kebutuhan
- [ ] Run training script
- [ ] Monitor training progress
- [ ] Evaluate trained model
- [ ] Run inference demo

---

## 🎯 Next Steps

1. **Fine-tune dengan dataset custom Anda**
2. **Experiment dengan hyperparameter berbeda**
3. **Try different ViM variants (Tiny, Small, Base)**
4. **Deploy model untuk production**
5. **Optimize untuk inference speed**

---

## 📄 License

Vision Mamba (ViM) dilisensikan under Apache License 2.0.
Detectron2 dilisensikan under Apache License 2.0.

---

## 🙏 Citation

Jika menggunakan ViM dalam research, please cite:

```bibtex
@inproceedings{vim,
  title={Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model},
  author={Zhu, Lianghui and Liao, Bencheng and Zhang, Qian and Wang, Xinlong and Liu, Wenyu and Wang, Xinggang},
  booktitle={Forty-first International Conference on Machine Learning},
  year={2024}
}
```

---

**Good luck with your ViM fine-tuning! 🚀**

_Last updated: November 2025_
