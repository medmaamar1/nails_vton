import json
import os

mapped_json_path = r"c:\Users\OrdiOne\Desktop\douccana marketplace - Copy\nails_vton\_annotations_mapped_train.coco.json"
orientation_json_path = r"c:\Users\OrdiOne\Desktop\douccana marketplace - Copy\nails_vton\mp_orientations_v1.json"

def verify_compatibility():
    with open(mapped_json_path, 'r') as f:
        mapped_data = json.load(f)
    
    with open(orientation_json_path, 'r') as f:
        orient_data = json.load(f)

    mapped_images = {str(img['id']): img['file_name'] for img in mapped_data['images']}
    orient_image_ids = set(orient_data.keys())

    # 1. Image ID coverage
    mapped_image_ids = set(mapped_images.keys())
    common_images = mapped_image_ids.intersection(orient_image_ids)
    
    print(f"Mapped JSON Images: {len(mapped_image_ids)}")
    print(f"Orientation JSON Images: {len(orient_image_ids)}")
    print(f"Common Image IDs: {len(common_images)}")

    if not common_images:
        print("CRITICAL: No common Image IDs found between the two JSONs.")
        return

    # 2. Annotation ID coverage for a few samples
    sample_ids = list(common_images)[:5]
    for img_id in sample_ids:
        mapped_ann_ids = {str(ann['id']) for ann in mapped_data['annotations'] if str(ann['image_id']) == img_id}
        orient_ann_ids = set(orient_data[img_id].keys())
        
        common_anns = mapped_ann_ids.intersection(orient_ann_ids)
        print(f"\nImage ID {img_id} ({mapped_images[img_id]}):")
        print(f"  Mapped Annotations: {mapped_ann_ids}")
        print(f"  Orient Annotations: {orient_ann_ids}")
        print(f"  Matches: {len(common_anns)}/{len(mapped_ann_ids)}")

if __name__ == "__main__":
    verify_compatibility()
