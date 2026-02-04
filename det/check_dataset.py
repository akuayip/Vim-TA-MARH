"""
Quick dataset validation script
Checks:
1. Dataset registration
2. Annotation format
3. Bounding box validity
4. Image-annotation consistency
5. Class distribution
"""

import sys
sys.path.append('.')
import register_custom_coco

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
import cv2
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

def check_dataset():
    print("=" * 80)
    print("DATASET VALIDATION CHECK")
    print("=" * 80)
    
    # 1. Check registration
    print("\n[1] Checking dataset registration...")
    try:
        train_dicts = DatasetCatalog.get("custom_coco_train")
        val_dicts = DatasetCatalog.get("custom_coco_val")
        print(f"✅ Train dataset: {len(train_dicts)} images")
        print(f"✅ Val dataset: {len(val_dicts)} images")
    except Exception as e:
        print(f"❌ Registration failed: {e}")
        return None, None
    
    # 2. Check metadata
    print("\n[2] Checking metadata...")
    train_meta = MetadataCatalog.get("custom_coco_train")
    print(f"✅ Classes: {train_meta.thing_classes}")
    print(f"✅ Class count: {len(train_meta.thing_classes)}")
    
    # 3. Check annotations
    print("\n[3] Checking annotations...")
    total_boxes = 0
    invalid_boxes = 0
    empty_images = 0
    box_areas = []
    
    for idx, d in enumerate(train_dicts):
        annos = d.get("annotations", [])
        
        if len(annos) == 0:
            empty_images += 1
            if empty_images <= 5:
                print(f"⚠️  Empty image: {d['file_name']}")
        
        for anno in annos:
            total_boxes += 1
            bbox = anno["bbox"]
            x, y, w, h = bbox
            
            # Check validity
            if w <= 0 or h <= 0:
                invalid_boxes += 1
                print(f"❌ Invalid box (w={w}, h={h}): {d['file_name']}")
            
            # Check if box is inside image
            img_h, img_w = d["height"], d["width"]
            if x < 0 or y < 0 or x+w > img_w or y+h > img_h:
                print(f"⚠️  Box outside image: {d['file_name']} - bbox: {bbox}, img_size: ({img_w}, {img_h})")
            
            # Collect box areas
            box_areas.append(w * h)
    
    print(f"\n✅ Total boxes: {total_boxes}")
    if len(train_dicts) > 0:
        print(f"✅ Average boxes per image: {total_boxes / len(train_dicts):.2f}")
        print(f"⚠️  Empty images: {empty_images} ({empty_images/len(train_dicts)*100:.1f}%)")
    print(f"❌ Invalid boxes: {invalid_boxes}")
    
    # 4. Box size statistics
    if len(box_areas) > 0:
        print("\n[4] Box size statistics...")
        box_areas = np.array(box_areas)
        print(f"✅ Min area: {box_areas.min():.0f} px²")
        print(f"✅ Max area: {box_areas.max():.0f} px²")
        print(f"✅ Mean area: {box_areas.mean():.0f} px²")
        print(f"✅ Median area: {np.median(box_areas):.0f} px²")
        
        # COCO size categories (area in pixels²)
        small = (box_areas < 32**2).sum()
        medium = ((box_areas >= 32**2) & (box_areas < 96**2)).sum()
        large = (box_areas >= 96**2).sum()
        
        print(f"\n✅ COCO Size Distribution:")
        print(f"   - Small (<32²):   {small:4d} boxes ({small/total_boxes*100:5.1f}%)")
        print(f"   - Medium (32²-96²): {medium:4d} boxes ({medium/total_boxes*100:5.1f}%)")
        print(f"   - Large (>96²):    {large:4d} boxes ({large/total_boxes*100:5.1f}%)")
    else:
        print("\n[4] ❌ No boxes found in dataset!")
    
    # 5. Visualize sample
    print("\n[5] Visualizing samples...")
    
    # Create work_dirs if not exists
    Path("work_dirs").mkdir(exist_ok=True)
    
    if len(train_dicts) > 0:
        sample_indices = [0, len(train_dicts)//4, len(train_dicts)//2, len(train_dicts)-1]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        for i, idx in enumerate(sample_indices):
            d = train_dicts[idx]
            img = cv2.imread(d["file_name"])
            
            if img is None:
                print(f"❌ Cannot read image: {d['file_name']}")
                axes[i].text(0.5, 0.5, f"Image not found\n{Path(d['file_name']).name}", 
                           ha='center', va='center')
                axes[i].axis('off')
                continue
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            visualizer = Visualizer(img, metadata=train_meta, scale=0.5)
            vis = visualizer.draw_dataset_dict(d)
            
            axes[i].imshow(vis.get_image())
            axes[i].set_title(f"Image {idx}: {len(d['annotations'])} boxes\n{Path(d['file_name']).name}")
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('work_dirs/dataset_validation.png', dpi=100, bbox_inches='tight')
        print(f"✅ Saved visualization: work_dirs/dataset_validation.png")
    else:
        print("❌ No images to visualize!")
    
    # 6. Check JSON files
    print("\n[6] Checking raw JSON files...")
    json_files = [
        "datasets/coco/train/_annotations.coco.json",
        "datasets/coco/valid/_annotations.coco.json"
    ]
    
    for json_file in json_files:
        if Path(json_file).exists():
            with open(json_file) as f:
                data = json.load(f)
            print(f"\n✅ {json_file}:")
            print(f"   - Images: {len(data['images'])}")
            print(f"   - Annotations: {len(data['annotations'])}")
            print(f"   - Categories: {data['categories']}")
            
            # Check for empty annotations
            img_ids_with_annos = set(a['image_id'] for a in data['annotations'])
            img_ids_all = set(img['id'] for img in data['images'])
            empty_imgs = img_ids_all - img_ids_with_annos
            print(f"   - Images without annotations: {len(empty_imgs)}")
        else:
            print(f"❌ File not found: {json_file}")
    
    # 7. Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    issues = []
    if len(train_dicts) > 0 and empty_images > len(train_dicts) * 0.1:
        issues.append(f"⚠️  Too many empty images: {empty_images} ({empty_images/len(train_dicts)*100:.1f}%)")
    if invalid_boxes > 0:
        issues.append(f"❌ Invalid boxes found: {invalid_boxes}")
    if len(box_areas) > 0 and large == 0:
        issues.append(f"ℹ️  No large objects (expected for head detection)")
    
    if not issues:
        print("✅ Dataset looks GOOD! No major issues found.")
    else:
        print("Issues found:")
        for issue in issues:
            print(issue)
    
    print("\n" + "=" * 80)
    
    return train_dicts, val_dicts

if __name__ == "__main__":
    train_dicts, val_dicts = check_dataset()
    
    # Extra: Show first annotation
    if train_dicts and len(train_dicts) > 0:
        print("\n[SAMPLE] First training annotation:")
        print(json.dumps(train_dicts[0], indent=2, default=str))