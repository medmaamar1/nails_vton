import os
import json
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import random

# Category lookup for v2
CATEGORIES = {
    0:  ("Unknown", "gray"),
    1:  ("L_Thumb", "red"),
    2:  ("L_Index", "orange"),
    3:  ("L_Middle", "yellow"),
    4:  ("L_Ring", "green"),
    5:  ("L_Pinky", "blue"),
    6:  ("R_Thumb", "purple"),
    7:  ("R_Index", "cyan"),
    8:  ("R_Middle", "magenta"),
    9:  ("R_Ring", "lime"),
    10: ("R_Pinky", "pink"),
}

def visualize_v2(json_path, train_dir, num_samples=20, out_dir="debug_v2_plots"):
    with open(json_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)

    train_dir = Path(train_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Indexing
    id_to_file = {img['id']: img['file_name'] for img in coco['images']}
    id_to_anns = {}
    for ann in coco['annotations']:
        id_to_anns.setdefault(ann['image_id'], []).append(ann)

    # Filter for images that have at least one non-unknown match
    valid_ids = []
    for img_id, anns in id_to_anns.items():
        if any(ann['category_id'] > 0 for ann in anns):
            valid_ids.append(img_id)
    
    print(f"Total matched images: {len(valid_ids)}")
    samples = random.sample(valid_ids, min(num_samples, len(valid_ids)))

    for img_id in samples:
        fname = id_to_file[img_id]
        anns = id_to_anns[img_id]
        
        # Robust path resolution (same as mapping script)
        img_path = train_dir / fname
        if not img_path.exists():
            # Try subfolder images/
            img_path = train_dir / "images" / fname
            if not img_path.exists():
                # Try recursive search
                found = list(train_dir.rglob(fname))
                if found:
                    img_path = found[0]
                else:
                    print(f"Skipping {fname}, path not found.")
                    continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img)

        for ann in anns:
            cat_id = ann['category_id']
            name, color = CATEGORIES.get(cat_id, ("Error", "black"))
            
            # BBox
            x, y, w, h = ann['bbox']
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none', alpha=0.8)
            ax.add_patch(rect)

            # Segmentation
            if 'segmentation' in ann and ann['segmentation']:
                for seg in ann['segmentation']:
                    if len(seg) >= 6:
                        poly = patches.Polygon(list(zip(seg[::2], seg[1::2])), 
                                              linewidth=1, edgecolor=color, facecolor=color, alpha=0.3)
                        ax.add_patch(poly)

            # Label
            plt.text(x, y-10, name, color="white", weight="bold", 
                     bbox=dict(facecolor=color, alpha=0.8, edgecolor="none", pad=2))

        plt.title(f"v2 Mapping: {fname}")
        plt.axis('off')
        
        save_path = out_dir / f"v2_check_{fname}.png"
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"  Saved visualization: {save_path.name}")

if __name__ == "__main__":
    JSON_PATH = r"C:\Users\OrdiOne\Desktop\douccana marketplace - Copy\nails_vton\_annotations_mapped_v2.coco.json"
    TRAIN_DIR = r"C:\Users\OrdiOne\Desktop\douccana marketplace - Copy\nails_segmentation_coco\train"
    visualize_v2(JSON_PATH, TRAIN_DIR)
