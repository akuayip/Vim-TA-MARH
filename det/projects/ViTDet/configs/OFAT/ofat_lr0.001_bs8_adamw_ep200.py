"""
OFAT Epoch Variation: Fine-tuning ViM-Tiny for Head Detection
Parameters:
- Learning Rate: 0.001 (baseline)
- Batch Size: 8 (baseline)
- Optimizer: AdamW (baseline)
- Epochs: 200 (VARIED)
- Backbone: FROZEN
- Detection Head: Cascade R-CNN
- Dataset: Custom COCO (1271 train images, 159 val images)
"""

from functools import partial
from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from detectron2.modeling.backbone.vim import get_vim_lr_decay_rate

# ============================================================================
# DATA CONFIGURATION  
# ============================================================================
from detectron2.data import (
    DatasetMapper,
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.data import transforms as T
from detectron2.evaluation import COCOEvaluator
from omegaconf import OmegaConf

# Register custom dataset
import sys
sys.path.append('.')
import register_custom_coco  # This will register the datasets

# Data Augmentation
image_size = 640  # Match dataset image size (640x640)
dataloader = OmegaConf.create()
dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="custom_coco_train"),
    mapper=L(DatasetMapper)(
        is_train=True,
        augmentations=[
            L(T.RandomFlip)(horizontal=True),
            L(T.ResizeShortestEdge)(
                short_edge_length=(480, 560, 640),
                max_size=640,
                sample_style="choice",
            ),
        ],
        image_format="RGB",
        use_instance_mask=False,  # No masks, only bounding boxes
    ),
    total_batch_size=8,  # BASELINE BATCH SIZE
    num_workers=4,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="custom_coco_val", filter_empty=False),
    mapper=L(DatasetMapper)(
        is_train=False,
        augmentations=[
            L(T.ResizeShortestEdge)(short_edge_length=640, max_size=640),
        ],
        image_format="RGB",
    ),
    num_workers=4,
)

dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
)

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
# Use Cascade R-CNN for detection
from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.roi_heads import (
    FastRCNNOutputLayers,
    FastRCNNConvFCHead,
    CascadeROIHeads,
)

model = model_zoo.get_config("common/models/mask_rcnn_vimdet.py").model


# Switch to Cascade R-CNN (remove mask head completely)
model.roi_heads.pop("mask_in_features", None)
model.roi_heads.pop("mask_pooler", None)
model.roi_heads.pop("mask_head", None)

[model.roi_heads.pop(k) for k in ["box_head", "box_predictor", "proposal_matcher"]]

model.roi_heads.update(
    num_classes=1,  # Single foreground class: head
    _target_=CascadeROIHeads,
    box_heads=[
        L(FastRCNNConvFCHead)(
            input_shape=ShapeSpec(channels=256, height=7, width=7),
            conv_dims=[256, 256, 256, 256],
            fc_dims=[1024],
            conv_norm="LN",
        )
        for _ in range(3)
    ],
    box_predictors=[
        L(FastRCNNOutputLayers)(
            input_shape=ShapeSpec(channels=1024),
            test_score_thresh=0.05,
            box2box_transform=L(Box2BoxTransform)(weights=(w1, w1, w2, w2)),
            cls_agnostic_bbox_reg=True,
            num_classes=1,  # Reference model.num_classes
        )
        for (w1, w2) in [(10, 5), (20, 10), (30, 15)]
    ],
    proposal_matchers=[
        L(Matcher)(thresholds=[th], labels=[0, 1], allow_low_quality_matches=False)
        for th in [0.5, 0.6, 0.7]
    ],
)

# ViM-Tiny backbone configuration
model.backbone.net.img_size = 640  # Match dataset image size
model.backbone.net.embed_dim = 192
model.backbone.net.depth = 24
model.backbone.net.pretrained = "ckpts/vim_tiny_pretrained.pth"  # PRETRAINED WEIGHTS
model.backbone.net.freeze_backbone = True  # FREEZE BACKBONE for fine-tuning
model.backbone.square_pad = 640  # Match img_size

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
train = model_zoo.get_config("common/train.py").train
train.amp.enabled = True
train.ddp.fp16_compression = True
train.init_checkpoint = ""  # We load pretrained weights via model.backbone.net.pretrained

# Calculate iterations for 200 epochs
# Dataset: 1271 training images
# Batch size: 8
# Iterations per epoch = 1271 / 8 ≈ 159
# Total iterations = 159 * 200 = 31800

ITERS_PER_EPOCH = 159
EPOCHS = 200  # VARIED EPOCHS
train.max_iter = ITERS_PER_EPOCH * EPOCHS  # 31800 iterations

train.checkpointer.period = ITERS_PER_EPOCH  # Save checkpoint every epoch
train.eval_period = ITERS_PER_EPOCH  # Evaluate every epoch
train.log_period = 50

# Output directory
train.output_dir = "./work_dirs/ofat_lr0.001_bs8_adamw_ep200"

# ============================================================================
# LEARNING RATE SCHEDULE
# ============================================================================
# MultiStep LR: Decay at 70% and 90% of training
milestone_1 = int(0.7 * train.max_iter)  # At 70% (epoch 70)
milestone_2 = int(0.9 * train.max_iter)  # At 90% (epoch 90)

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[milestone_1, milestone_2],
        num_updates=train.max_iter,
    ),
    warmup_length=500 / train.max_iter,  # 500 iterations warmup
    warmup_factor=0.001,
)

# ============================================================================
# OPTIMIZER
# ============================================================================
optimizer = model_zoo.get_config("common/optim.py").AdamW  # BASELINE OPTIMIZER
optimizer.lr = 0.001  # BASELINE LEARNING RATE (1e-3)

# Layer-wise LR decay for ViM backbone (only applies to non-frozen layers)
optimizer.params.lr_factor_func = partial(
    get_vim_lr_decay_rate, 
    num_layers=24, 
    lr_decay_rate=0.7
)
optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}
