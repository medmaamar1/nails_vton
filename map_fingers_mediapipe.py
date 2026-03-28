import os
import json
import cv2
import numpy as np
from tqdm import tqdm
import mediapipe as mp
import mediapipe.python.solutions.hands as mp_hands

FINGER_UNUSED = 0
FINGER_THUMB  = 1
FINGER_INDEX  = 2
FINGER_MIDDLE = 3
FINGER_RING   = 4
FINGER_PINKY  = 5

def assign_finger_ids(bboxes):
    n = len(bboxes)
    if n == 0: return []
    cx_list = [x + w / 2.0 for x, y, w, h in bboxes]
    cy_list = [y + h / 2.0 for x, y, w, h in bboxes]
    area_list = [w * h for x, y, w, h in bboxes]
    median_cy = sorted(cy_list)[len(cy_list) // 2]
    thumb_idx = None
    best_area = -1
    for i, (area, cy) in enumerate(zip(area_list, cy_list)):
        if cy > median_cy and area > best_area:
            best_area = area
            thumb_idx = i
    if thumb_idx is None: thumb_idx = int(np.argmax(area_list))
    remaining = [(i, cx_list[i]) for i in range(n) if i != thumb_idx]
    remaining.sort(key=lambda t: t[1])
    pos_to_label = [FINGER_PINKY, FINGER_RING, FINGER_MIDDLE, FINGER_INDEX]
    labels = [FINGER_UNUSED] * n
    labels[thumb_idx] = FINGER_THUMB
    for pos, (orig_idx, _) in enumerate(remaining):
        if pos < len(pos_to_label): labels[orig_idx] = pos_to_label[pos]
        else: labels[orig_idx] = FINGER_UNUSED
    return labels

def map_fingers_with_mediapipe(train_dir, output_path):
    print(f"Loading annotations from {train_dir}...")
    json_path = os.path.join(train_dir, "_annotations.coco.json")
    with open(json_path, 'r') as f: coco = json.load(f)

    coco['categories'] = [
        {"id": 1, "name": "Thumb", "supercategory": "nails"},
        {"id": 2, "name": "Index", "supercategory": "nails"},
        {"id": 3, "name": "Middle", "supercategory": "nails"},
        {"id": 4, "name": "Ring", "supercategory": "nails"},
        {"id": 5, "name": "Pinky", "supercategory": "nails"}
    ]

    hands = mp_hands.Hands(
        static_image_mode=True, max_num_hands=2, min_detection_confidence=0.1
    )
    TIP_IDS = {
        4: FINGER_THUMB, 8: FINGER_INDEX, 12: FINGER_MIDDLE, 16: FINGER_RING, 20: FINGER_PINKY
    }

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
        if image is None: continue
        h, w, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            success_count += 1
            tips = []
            for hand_landmarks in results.multi_hand_landmarks:
                for tip_idx, finger_label in TIP_IDS.items():
                    lm = hand_landmarks.landmark[tip_idx]
                    px, py = int(lm.x * w), int(lm.y * h)
                    tips.append((px, py, finger_label))
            
            for ann in anns:
                bx, by, bw, bh = ann['bbox']
                cx, cy = bx + bw/2.0, by + bh/2.0
                best_dist, best_label = float('inf'), FINGER_INDEX
                for (px, py, label) in tips:
                    dist = (cx - px)**2 + (cy - py)**2
                    if dist < best_dist:
                        best_dist = dist
                        best_label = label
                ann['category_id'] = best_label
        else:
            fallback_count += 1
            bboxes = [ann['bbox'] for ann in anns]
            labels = assign_finger_ids(bboxes)
            for ann, label in zip(anns, labels):
                ann['category_id'] = label if label != 0 else FINGER_INDEX

    hands.close()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f: json.dump(coco, f)

    print("\nMapping Complete!")
    print(f"MediaPipe successfully 3D-mapped {success_count} images.")
    print(f"Fell back to old heuristic for {fallback_count} trick images.")
    print(f"Saved robust mapped COCO JSON to: {output_path}")

if __name__ == "__main__":
    train_dir = r"C:\Users\OrdiOne\Desktop\douccana marketplace - Copy\nails_segmentation_coco\train"
    out_path  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_annotations_mapped_train.coco.json")
    map_fingers_with_mediapipe(train_dir, out_path)
