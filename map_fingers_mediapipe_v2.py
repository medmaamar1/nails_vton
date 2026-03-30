"""
Rock-Solid MediaPipe Finger Mapping v2
---------------------------------------
Handles all real-world edge cases:
  - One hand or two hands in the same image
  - Two hands close together (per-hand nail grouping)
  - Partial hands (not all fingers visible)
  - Hand rotation / overhead / egocentric shots
  - Confidence filtering — rejects weak MediaPipe detections

10-Class Category Schema (Left+Right × 5 Fingers):
  0  = Unknown  (no reliable match found — nail still goes to binary mask)
  1  = Left_Thumb   2  = Left_Index   3  = Left_Middle   4  = Left_Ring   5  = Left_Pinky
  6  = Right_Thumb  7  = Right_Index  8  = Right_Middle  9  = Right_Ring  10 = Right_Pinky

Note on chirality: MediaPipe labels from the CAMERA's perspective.
For egocentric (selfie-style) photos this is consistent across the whole dataset.

Output:
  _annotations_mapped_v2.coco.json   — enriched COCO JSON, category_id 0..10
  _annotations_mapped_v2_audit.json  — per-image stats for quality inspection

Usage:
  python map_fingers_mediapipe_v2.py
  python map_fingers_mediapipe_v2.py --train_dir /path/to/train --output /path/to/output.json
"""

import os
import sys
import json
import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

import mediapipe as mp
mp_hands = mp.solutions.hands

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    sys.exit("[ERROR] scipy not installed. Run: pip install scipy")


# ── Category schema ────────────────────────────────────────────────────────────

FINGER_UNKNOWN = 0
FINGER_THUMB, FINGER_INDEX, FINGER_MIDDLE, FINGER_RING, FINGER_PINKY = 1, 2, 3, 4, 5

CATEGORIES = [
    {"id": 0, "name": "Unknown", "supercategory": "nail"},
    {"id": 1, "name": "Thumb",   "supercategory": "nail"},
    {"id": 2, "name": "Index",   "supercategory": "nail"},
    {"id": 3, "name": "Middle",  "supercategory": "nail"},
    {"id": 4, "name": "Ring",    "supercategory": "nail"},
    {"id": 5, "name": "Pinky",   "supercategory": "nail"},
]

# Map both Left and Right hands to the same 5 semantic finger IDs
HAND_TIP_TO_CAT = {
    "Left":  {4: 1, 8: 2, 12: 3, 16: 4, 20: 5},
    "Right": {4: 1, 8: 2, 12: 3, 16: 4, 20: 5},
}

# How many nail "diameters" a fingertip can be away and still be matched
DIST_THRESHOLD_MULTIPLIER = 2.5

# MediaPipe confidence floor — below this the detection is too noisy
MIN_DETECTION_CONFIDENCE = 0.5


# ── Core logic ─────────────────────────────────────────────────────────────────

def _hand_centroid(hand_landmarks, img_w: int, img_h: int):
    """Pixel-space centroid of all 21 hand landmarks."""
    xs = [lm.x * img_w for lm in hand_landmarks.landmark]
    ys = [lm.y * img_h for lm in hand_landmarks.landmark]
    return float(np.mean(xs)), float(np.mean(ys))


def _tips_for_hand(hand_landmarks, chirality: str, img_w: int, img_h: int):
    """
    Return list of {px, py, cat_id} for the five fingertips of one hand.
    chirality: "Left" or "Right" (MediaPipe's camera-perspective label).
    """
    tip_map = HAND_TIP_TO_CAT[chirality]
    return [
        {
            "px": hand_landmarks.landmark[tip_idx].x * img_w,
            "py": hand_landmarks.landmark[tip_idx].y * img_h,
            "cat_id": cat_id,
        }
        for tip_idx, cat_id in tip_map.items()
    ]


def _assign_nails_to_hands(anns: list, hand_centroids: list):
    """
    For each nail annotation, find the closest hand centroid.
    Returns list of hand indices parallel to `anns`.
    """
    assignments = []
    for ann in anns:
        bx, by, bw, bh = ann["bbox"]
        cx, cy = bx + bw / 2.0, by + bh / 2.0
        best_hi, best_d = 0, float("inf")
        for hi, (hx, hy) in enumerate(hand_centroids):
            d = (cx - hx) ** 2 + (cy - hy) ** 2
            if d < best_d:
                best_d, best_hi = d, hi
        assignments.append(best_hi)
    return assignments


def _hungarian_match(ann_group: list, tips: list):
    """
    1-to-1 Hungarian matching between nail centroids and fingertip positions.

    Cost = Euclidean distance.
    Pairs where distance > DIST_THRESHOLD_MULTIPLIER * nail_diameter are REJECTED
    (penalty cost 1e9 ensures they are not matched).

    Returns dict: ann_id → cat_id  (only for accepted matches)
    """
    if not ann_group or not tips:
        return {}

    n_nails = len(ann_group)
    n_tips  = len(tips)

    cost = np.full((n_nails, n_tips), 1e9, dtype=np.float64)

    for i, ann in enumerate(ann_group):
        bx, by, bw, bh = ann["bbox"]
        cx, cy = bx + bw / 2.0, by + bh / 2.0
        radius = max(bw, bh) * DIST_THRESHOLD_MULTIPLIER

        for j, tip in enumerate(tips):
            dist = np.sqrt((cx - tip["px"]) ** 2 + (cy - tip["py"]) ** 2)
            if dist <= radius:
                cost[i, j] = dist

    row_ind, col_ind = linear_sum_assignment(cost)

    result = {}
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] < 1e8:          # real match (not penalty)
            result[ann_group[r]["id"]] = tips[c]["cat_id"]

    return result


# ── File resolution ────────────────────────────────────────────────────────────

def _build_file_index(train_dir: Path):
    """
    Build two lookup maps to resolve filenames robustly:
      hash_to_path : Roboflow rf-hash → absolute Path
      name_to_path : bare filename   → absolute Path
    Handles truncated filenames and nested sub-directories.
    """
    all_files = list(train_dir.rglob("*.jpg")) + list(train_dir.rglob("*.png"))
    hash_to_path, name_to_path = {}, {}
    for p in all_files:
        name_to_path[p.name] = p
        if ".rf." in p.name:
            h = p.name.split(".rf.")[-1].split(".")[0]
            hash_to_path[h] = p
    return hash_to_path, name_to_path


def _resolve_path(fname: str, train_dir: Path, hash_to_path, name_to_path):
    rf_hash = fname.split(".rf.")[-1].split(".")[0] if ".rf." in fname else None
    if rf_hash and rf_hash in hash_to_path:
        return hash_to_path[rf_hash]
    if fname in name_to_path:
        return name_to_path[fname]
    for candidate in [train_dir / fname, train_dir / "images" / fname]:
        try:
            if candidate.exists():
                return candidate
        except OSError:
            pass
    return None


# ── Main mapping pipeline ──────────────────────────────────────────────────────

def map_fingers_v2(train_dir: str, output_path: str, audit_path: str):
    train_dir = Path(train_dir)
    output_path = Path(output_path)
    audit_path  = Path(audit_path)

    json_path = train_dir / "_annotations.coco.json"
    if not json_path.exists():
        sys.exit(f"[ERROR] COCO JSON not found: {json_path}")

    print(f"[v2] Loading COCO JSON ... ", end="", flush=True)
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    print("done.")

    # ── Reset all category IDs to Unknown ─────────────────────────────────────
    coco["categories"] = CATEGORIES
    for ann in coco["annotations"]:
        ann["category_id"] = FINGER_UNKNOWN

    # ── Build lookup structures ────────────────────────────────────────────────
    id_to_file    = {img["id"]: img["file_name"] for img in coco["images"]}
    id_to_anns    = {}
    ann_by_id     = {}
    for ann in coco["annotations"]:
        id_to_anns.setdefault(ann["image_id"], []).append(ann)
        ann_by_id[ann["id"]] = ann

    hash_to_path, name_to_path = _build_file_index(train_dir)

    # ── MediaPipe setup ────────────────────────────────────────────────────────
    detector = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_DETECTION_CONFIDENCE,
    )

    # ── Per-image loop ─────────────────────────────────────────────────────────
    audit        = []
    n_mp_success = 0
    n_total_nails   = sum(len(v) for v in id_to_anns.values())
    n_matched_nails = 0

    for img_info in tqdm(coco["images"], desc="Mapping fingers v2", unit="img"):
        img_id = img_info["id"]
        anns   = id_to_anns.get(img_id, [])
        if not anns:
            continue

        entry = {
            "image_id":  img_id,
            "n_nails":   len(anns),
            "n_hands":   0,
            "n_matched": 0,
            "status":    "no_hands",
        }

        # Resolve image path
        img_path = _resolve_path(id_to_file[img_id], train_dir, hash_to_path, name_to_path)
        if img_path is None:
            entry["status"] = "file_not_found"
            audit.append(entry)
            continue

        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            entry["status"] = "load_failed"
            audit.append(entry)
            continue

        img_h, img_w = image_bgr.shape[:2]
        results = detector.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

        if not results.multi_hand_landmarks:
            audit.append(entry)
            continue

        n_hands = len(results.multi_hand_landmarks)
        entry["n_hands"] = n_hands
        entry["status"]  = "ok"
        n_mp_success    += 1

        # 1. Compute centroid for each detected hand
        hand_centroids = [
            _hand_centroid(lm, img_w, img_h)
            for lm in results.multi_hand_landmarks
        ]

        # 2. Group each nail annotation to its closest hand
        nail_to_hand_idx = _assign_nails_to_hands(anns, hand_centroids)

        hand_ann_groups = {i: [] for i in range(n_hands)}
        for ann, hi in zip(anns, nail_to_hand_idx):
            hand_ann_groups[hi].append(ann)

        # 3. Hungarian match within each hand group
        matched_this_img = 0
        for hi, (lm, handedness) in enumerate(
            zip(results.multi_hand_landmarks, results.multi_handedness)
        ):
            chirality  = handedness.classification[0].label   # "Left" or "Right"
            tips       = _tips_for_hand(lm, chirality, img_w, img_h)
            ann_group  = hand_ann_groups[hi]
            matches    = _hungarian_match(ann_group, tips)

            for ann_id, cat_id in matches.items():
                ann_by_id[ann_id]["category_id"] = cat_id
                matched_this_img += 1

        n_matched_nails         += matched_this_img
        entry["n_matched"]       = matched_this_img
        audit.append(entry)

    detector.close()

    # ── Save outputs ───────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(coco, f)

    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)

    # ── Summary ────────────────────────────────────────────────────────────────
    imgs_with_nails = len([e for e in audit if e.get("n_nails", 0) > 0])
    two_hand_imgs   = len([e for e in audit if e.get("n_hands", 0) == 2])
    zero_match_imgs = len([e for e in audit if e.get("status") == "ok" and e.get("n_matched", 0) == 0])

    img_match_pct  = n_mp_success   / imgs_with_nails   * 100 if imgs_with_nails   > 0 else 0
    nail_match_pct = n_matched_nails / n_total_nails     * 100 if n_total_nails     > 0 else 0

    print()
    print("=" * 55)
    print(" v2 Mapping Complete")
    print("=" * 55)
    print(f"  Images with nails           : {imgs_with_nails}")
    print(f"  MediaPipe succeeded on      : {n_mp_success}  ({img_match_pct:.1f}%)")
    print(f"  Images with 2 hands         : {two_hand_imgs}")
    print(f"  MediaPipe ok but 0 matched  : {zero_match_imgs}  (investigate!)")
    print(f"  Nails matched / total       : {n_matched_nails} / {n_total_nails}  ({nail_match_pct:.1f}%)")
    print(f"  Unknown nails (cat_id=0)    : {n_total_nails - n_matched_nails}")
    print(f"  Output JSON  : {output_path}")
    print(f"  Audit log    : {audit_path}")
    print()

    # Category distribution
    from collections import Counter
    cat_counts = Counter(ann["category_id"] for ann in coco["annotations"])
    print("  Category breakdown:")
    for cat in CATEGORIES:
        cid  = cat["id"]
        name = cat["name"]
        cnt  = cat_counts.get(cid, 0)
        pct  = cnt / n_total_nails * 100 if n_total_nails > 0 else 0
        bar  = "█" * int(pct / 2)
        print(f"    [{cid:2d}] {name:<14} : {cnt:5d}  ({pct:5.1f}%)  {bar}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rock-solid MediaPipe finger mapping v2")
    parser.add_argument(
        "--train_dir",
        default=r"C:\Users\OrdiOne\Desktop\douccana marketplace - Copy\nails_segmentation_coco\train",
        help="Path to the train folder containing _annotations.coco.json and images",
    )
    parser.add_argument(
        "--output",
        default=r"C:\Users\OrdiOne\Desktop\douccana marketplace - Copy\nails_vton\_annotations_mapped_v2.coco.json",
        help="Output path for the enriched COCO JSON",
    )
    parser.add_argument(
        "--audit",
        default=r"C:\Users\OrdiOne\Desktop\douccana marketplace - Copy\nails_vton\_annotations_mapped_v2_audit.json",
        help="Output path for the per-image audit log",
    )
    args = parser.parse_args()

    map_fingers_v2(args.train_dir, args.output, args.audit)
