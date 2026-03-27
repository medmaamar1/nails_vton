import os
import json
import cv2
import numpy as np
from tqdm import tqdm

try:
    import mediapipe as mp
    from mediapipe.python.solutions import hands as mp_hands_module
except ImportError:
    print("Please install mediapipe first: pip install mediapipe")
    import sys
    sys.exit(1)

# Import the old heuristic as a fallback if MediaPipe can't see the hand
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import assign_finger_ids, FINGER_THUMB, FINGER_INDEX, FINGER_MIDDLE, FINGER_RING, FINGER_PINKY

def map_fingers_with_mediapipe(train_dir, output_path):
    print(f"Loading annotations from {train_dir}...")
    json_path = os.path.join(train_dir, "_annotations.coco.json")
    
    with open(json_path, 'r') as f:
        coco = json.load(f)

    # 1. Update Categories in COCO
    # Instead of just "Nail", we now have specific finger classes
    coco['categories'] = [
        {"id": 1, "name": "Thumb", "supercategory": "nails"},
        {"id": 2, "name": "Index", "supercategory": "nails"},
        {"id": 3, "name": "Middle", "supercategory": "nails"},
        {"id": 4, "name": "Ring", "supercategory": "nails"},
        {"id": 5, "name": "Pinky", "supercategory": "nails"}
    ]

    # MediaPipe Setup
    hands = mp_hands_module.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.3  # Generous confidence to catch tricky hands
    )

    # Landmark indices for fingertips
    TIP_IDS = {
        4: FINGER_THUMB,
        8: FINGER_INDEX,
        12: FINGER_MIDDLE,
        16: FINGER_RING,
        20: FINGER_PINKY
    }

    # Group annotations by image
    id_to_file = {img['id']: img['file_name'] for img in coco['images']}
    id_to_anns = {}
    for ann in coco['annotations']:
        id_to_anns.setdefault(ann['image_id'], []).append(ann)

    success_count = 0
    fallback_count = 0

    print("Mapping fingers using MediaPipe 3D Landmarks...")
    
    for img_id, anns in tqdm(id_to_anns.items()):
        file_name = id_to_file[img_id]
        img_path = os.path.join(train_dir, file_name)
        
        image = cv2.imread(img_path)
        if image is None:
            continue
            
        h, w, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            success_count += 1
            # Extract all fingertips from all detected hands
            tips = []
            for hand_landmarks in results.multi_hand_landmarks:
                for tip_idx, finger_label in TIP_IDS.items():
                    lm = hand_landmarks.landmark[tip_idx]
                    px, py = int(lm.x * w), int(lm.y * h)
                    tips.append((px, py, finger_label))
            
            # Map each bbox to the closest fingertip
            for ann in anns:
                bx, by, bw, bh = ann['bbox']
                cx, cy = bx + bw/2.0, by + bh/2.0
                
                best_dist = float('inf')
                best_label = FINGER_INDEX # fallback
                
                for (px, py, label) in tips:
                    dist = (cx - px)**2 + (cy - py)**2
                    if dist < best_dist:
                        best_dist = dist
                        best_label = label
                        
                # Update the annotation with the robust specific finger class!
                ann['category_id'] = best_label
                
        else:
            # Fallback to the old geometric logic if MediaPipe couldn't find the hand
            fallback_count += 1
            bboxes = [ann['bbox'] for ann in anns]
            labels = assign_finger_ids(bboxes)
            for ann, label in zip(anns, labels):
                # If old logic says 0 (UNUSED), default to Index to avoid crashing
                ann['category_id'] = label if label != 0 else FINGER_INDEX

    hands.close()

    # Save to the new, safe JSON file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(coco, f)

    print("\nMapping Complete!")
    print(f"MediaPipe successfully 3D-mapped {success_count} images.")
    print(f"Fell back to old heuristic for {fallback_count} tricky images.")
    print(f"Saved robust mapped COCO JSON to: {output_path}")

if __name__ == "__main__":
    train_dir = "/kaggle/input/datasets/maamarmohamed12/nails-vton/train"
    out_path  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_annotations_mapped_train.coco.json")
    map_fingers_with_mediapipe(train_dir, out_path)
