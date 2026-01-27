"""Debug script to check dataset loading and category mapping"""
import sys
sys.path.append('.')
import register_custom_coco

from detectron2.data import DatasetCatalog, MetadataCatalog

# Get dataset
dataset_dicts = DatasetCatalog.get("custom_coco_train")
metadata = MetadataCatalog.get("custom_coco_train")

print("Dataset metadata:")
print(f"  thing_classes: {metadata.thing_classes}")
print(f"  thing_dataset_id_to_contiguous_id: {metadata.thing_dataset_id_to_contiguous_id}")
print(f"  Number of thing classes: {len(metadata.thing_classes)}")

print("\nFirst 3 annotations:")
for i, record in enumerate(dataset_dicts[:3]):
    print(f"\nImage {i}: {record['file_name']}")
    for ann in record['annotations'][:2]:  # First 2 annotations per image
        print(f"  Category ID: {ann['category_id']}, BBox: {ann['bbox']}")
