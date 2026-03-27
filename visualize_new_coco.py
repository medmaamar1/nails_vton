import os
import sys
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import assign_finger_ids, FINGER_UNUSED, FINGER_THUMB, FINGER_INDEX, FINGER_MIDDLE, FINGER_RING, FINGER_PINKY
from test_finger_mapping import visualize_mapping, LBL, COLOR_MAP

def visualize_new_dataset_mapping(coco_dir, num_samples=5):
    json_path = os.path.join(coco_dir, "_annotations.coco.json")
    with open(json_path, 'r') as f:
        coco = json.load(f)

    # Group annotations by image_id
    id_to_file = {}
    for img in coco['images']:
        id_to_file[img['id']] = img['file_name']

    id_to_anns = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        id_to_anns.setdefault(img_id, []).append(ann)

    # Take first num_samples images
    image_ids = list(id_to_anns.keys())[:num_samples]

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_plots")
    os.makedirs(out_dir, exist_ok=True)
    
    for img_id in image_ids:
        anns = id_to_anns[img_id]
        bboxes = [ann['bbox'] for ann in anns]
        segmentations = [ann.get('segmentation', []) for ann in anns]
        labels = assign_finger_ids(bboxes)
        
        filename = id_to_file[img_id]
        print(f"Assigning labels for image: {filename}")
        
        # Load the actual image for visualization if available
        visualize_mapping_with_image(
            os.path.join(coco_dir, filename),
            bboxes,
            segmentations,
            labels,
            f"Image ID: {img_id}",
            f"mapped_{filename}"
        )

def visualize_mapping_with_image(img_path, bboxes, segmentations, labels, title, filename):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    try:
        img = plt.imread(img_path)
        ax.imshow(img)
    except Exception as e:
        print(f"Could not load image {img_path}: {e}")
        ax.set_facecolor("#f5d5c5")
        
        all_x = [b[0] for b in bboxes] + [b[0] + b[2] for b in bboxes]
        all_y = [b[1] for b in bboxes] + [b[1] + b[3] for b in bboxes]
        if all_x and all_y:
            padx, pady = 50, 50
            ax.set_xlim(min(all_x) - padx, max(all_x) + padx)
            ax.set_ylim(max(all_y) + pady, min(all_y) - pady)

    for i, (bbox, seg, label) in enumerate(zip(bboxes, segmentations, labels)):
        x, y, w, h = bbox
        color = COLOR_MAP.get(label, "black")
        name = LBL.get(label, f"UNK({label})")
        
        # Draw the precise curved polygon mask
        if seg and len(seg) > 0 and len(seg[0]) >= 6:
            poly = seg[0]
            xs = poly[0::2]
            ys = poly[1::2]
            polygon = patches.Polygon(list(zip(xs, ys)), linewidth=2, edgecolor=color, facecolor=color, alpha=0.4)
            ax.add_patch(polygon)
        else:
            # Fallback to bbox if no segmentation exists
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor=color, alpha=0.4)
            ax.add_patch(rect)
        
        # Draw bounding box outline for reference
        rect_outline = patches.Rectangle((x, y), w, h, linewidth=1, linestyle='--', edgecolor=color, facecolor='none')
        ax.add_patch(rect_outline)
        
        plt.text(x, y-5, f"{i}: {name}", color="white", weight="bold", 
                 bbox=dict(facecolor=color, alpha=0.9, edgecolor="none", pad=2))
        
        cx = x + w/2
        cy = y + h/2
        plt.plot(cx, cy, 'o', color=color, markersize=5)

    plt.title(title)
    plt.tight_layout()
    
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_plots", filename)
    plt.savefig(save_path)
    plt.close()
    print(f"  -> Visual confirmation saved to: {save_path}")

if __name__ == "__main__":
    test_dir = "/kaggle/input/datasets/maamarmohamed12/nails-vton/train"
    print("Testing mapping on new dataset...")
    visualize_new_dataset_mapping(test_dir, num_samples=10)
