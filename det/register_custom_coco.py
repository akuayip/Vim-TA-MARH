"""
Register Custom COCO Dataset for Head Detection
Dataset structure: datasets/coco/{train,valid,test}/_annotations.coco.json
Single class: head (category_id=1 in JSON, mapped to class_id=0)
"""

import os
import json
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog

# Dataset paths
DATASET_ROOT = "datasets/coco"

def filter_coco_categories(json_file):
    """Filter COCO JSON to only include category_id=1 (head class)"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Keep only category with id=1 (head)
    data['categories'] = [c for c in data['categories'] if c['id'] == 1]
    # Remap category_id from 1 to 0
    for cat in data['categories']:
        cat['id'] = 0
        cat['name'] = 'head'  # Simplify name
    for ann in data['annotations']:
        if ann['category_id'] == 1:
            ann['category_id'] = 0
    
    # Save to temp file
    temp_file = json_file.replace('.json', '_filtered.json')
    with open(temp_file, 'w') as f:
        json.dump(data, f)
    return temp_file

# Filter and register training set
train_json = filter_coco_categories(os.path.join(DATASET_ROOT, "train/_annotations.coco.json"))
register_coco_instances(
    "custom_coco_train",
    {},
    train_json,
    os.path.join(DATASET_ROOT, "train")
)

# Filter and register validation set
val_json = filter_coco_categories(os.path.join(DATASET_ROOT, "valid/_annotations.coco.json"))
register_coco_instances(
    "custom_coco_val",
    {},
    val_json,
    os.path.join(DATASET_ROOT, "valid")
)

# Filter and register test set
test_json = filter_coco_categories(os.path.join(DATASET_ROOT, "test/_annotations.coco.json"))
register_coco_instances(
    "custom_coco_test",
    {},
    test_json,
    os.path.join(DATASET_ROOT, "test")
)

print("✓ Registered custom_coco_train, custom_coco_val, custom_coco_test (1 class: head)")
