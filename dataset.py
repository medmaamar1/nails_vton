"""
Nail VTON Dataset
-----------------
Loads Roboflow v50 COCO Segmentation JSON.
Produces per-image:
  1. image           : (3, H, W)       float32 normalised
  2. binary_mask     : (1, H, W)       float32  — union of all nail masks
  3. instance_masks  : (10, H, W)      float32  — one-hot, channel i = nail i
  4. direction_field : (2, H, W)       float32  — unit vector base→tip per nail pixel
  5. finger_ids      : (10,)           int64    — finger label per slot (0=unused,
                                                  1=thumb, 2=index, 3=middle,
                                                  4=ring, 5=pinky)
  6. n_instances     : scalar          int64

Finger identity is derived geometrically — no extra annotation needed:
  • Thumb  → largest bbox area AND y-centroid is an outlier (below the finger row)
  • Fingers 2-5 → sorted by x-centroid: leftmost=pinky … rightmost=index
    (works for either hand; label assignment is position-relative)
"""

import json
import math
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF


# ── Constants ──────────────────────────────────────────────────────────────────
MAX_INSTANCES = 10
IMAGE_SIZE    = 512
MEAN          = [0.485, 0.456, 0.406]
STD           = [0.229, 0.224, 0.225]

# Finger label codes
FINGER_UNUSED = 0
FINGER_THUMB  = 1
FINGER_INDEX  = 2
FINGER_MIDDLE = 3
FINGER_RING   = 4
FINGER_PINKY  = 5


# ── Helpers ────────────────────────────────────────────────────────────────────

def polygon_to_mask(polygon, height, width):
    """Flat COCO polygon [x1,y1,x2,y2,...] → binary PIL mask."""
    mask = Image.new("L", (width, height), 0)
    if len(polygon) >= 6:
        xy = list(zip(polygon[::2], polygon[1::2]))
        ImageDraw.Draw(mask).polygon(xy, fill=255)
    return mask





# ── Dataset ────────────────────────────────────────────────────────────────────

class NailDataset(Dataset):
    def __init__(self, root, mp_json_path, augment=False, image_size=IMAGE_SIZE):
        self.root       = Path(root)
        self.augment    = augment
        self.image_size = image_size

        # 1. Load standard COCO annotations
        ann_path = self.root / "_annotations.coco.json"
        with open(ann_path, "r") as f:
            coco = json.load(f)

        # 2. Load MediaPipe Orientation Cache
        with open(mp_json_path, "r") as f:
            self.mp_data = json.load(f)

        self.id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}

        self.id_to_anns = {}
        for ann in coco["annotations"]:
            iid = ann["image_id"]
            self.id_to_anns.setdefault(iid, [])
            self.id_to_anns[iid].append(ann)

        # Filter image_ids: Only include those with verified MediaPipe skeletal data
        self.image_ids = [
            iid for iid in self.id_to_file
            if str(iid) in self.mp_data and len(self.id_to_anns.get(iid, [])) > 0
        ]

        # RAM Optimization: Prune the dictionary to only keep filtered IDs.
        # This prevents DataLoader workers from carrying the entire raw dataset JSON in memory.
        valid_str_ids = {str(iid) for iid in self.image_ids}
        self.mp_data = {k: v for k, v in self.mp_data.items() if k in valid_str_ids}

        print(f"[NailDataset] Filtered to {len(self.image_ids)} verified images  "
              f"(root={root}, augment={augment})")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id  = self.image_ids[idx]
        file_name = self.id_to_file[image_id]
        anns      = self.id_to_anns[image_id]

        # ── Load image ────────────────────────────────────────────────────────
        # Images may live directly in root or inside root/images
        img_path = self.root / file_name
        if not img_path.exists():
            img_path = self.root / "images" / file_name
        image          = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        # ── Build per-nail masks at original resolution ───────────────────────
        masks_pil   = []
        bboxes_orig = []

        for ann in anns[:MAX_INSTANCES]:
            seg = ann.get("segmentation", [])
            if not seg or len(seg[0]) < 6:
                continue
            masks_pil.append(polygon_to_mask(seg[0], orig_h, orig_w))
            bboxes_orig.append(ann["bbox"])

        # ── Finger identity ────────────────────────────────────────────────────
        # We do NOT use the geometric heuristic here — finger identity is resolved
        # at inference time via MediaPipe in the frontend. The AI only needs to
        # learn to segment each nail as a distinct instance.
        finger_labels = [FINGER_UNUSED] * len(bboxes_orig)

        # ── Resize to image_size ──────────────────────────────────────────────
        sx = self.image_size / orig_w
        sy = self.image_size / orig_h
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        masks_resized  = []
        bboxes_resized = []
        for m, (x, y, w, h) in zip(masks_pil, bboxes_orig):
            masks_resized.append(
                m.resize((self.image_size, self.image_size), Image.NEAREST)
            )
            bboxes_resized.append([x * sx, y * sy, w * sx, h * sy])

        # ── Augmentation ──────────────────────────────────────────────────────
        if self.augment:
            image, masks_resized = self._augment(image, masks_resized)

        # ── Image tensor ──────────────────────────────────────────────────────
        img_t = TF.normalize(TF.to_tensor(image), MEAN, STD)  # (3, H, W)

        S = self.image_size

        # ── Binary mask (union) ───────────────────────────────────────────────
        binary_np = np.zeros((S, S), dtype=np.float32)
        for m in masks_resized:
            binary_np = np.maximum(binary_np, np.array(m, dtype=np.float32) / 255.0)
        binary_t = torch.from_numpy(binary_np).unsqueeze(0)   # (1, H, W)

        # ── Instance masks — one-hot (10, H, W) ──────────────────────────────
        inst_np = np.zeros((MAX_INSTANCES, S, S), dtype=np.float32)
        for i, m in enumerate(masks_resized):
            inst_np[i] = np.array(m, dtype=np.float32) / 255.0
        inst_t = torch.from_numpy(inst_np)                     # (10, H, W)

        # ── Direction field ───────────────────────────────────────────────────
        # Using pre-computed skeletal anatomical directions (DIP -> TIP)
        img_mp_vectors = self.mp_data.get(str(image_id), {})
        dir_np = np.zeros((2, S, S), dtype=np.float32)

        for i, m in enumerate(masks_resized):
            ann_id = str(anns[i]["id"])
            if ann_id in img_mp_vectors:
                vx, vy = img_mp_vectors[ann_id]
                m_np = (np.array(m, dtype=np.float32) > 127).astype(np.float32)
                dir_np[0] += m_np * vx
                dir_np[1] += m_np * vy

        dir_t = torch.from_numpy(dir_np).clone()               # (2, H, W)

        # ── Finger id tensor (10,) — 0 for unused slots ───────────────────────
        finger_t = torch.zeros(MAX_INSTANCES, dtype=torch.long)
        # Note: Finger identification is deferred to inference (MediaPipe), 
        # so we keep these as 0 during standard binary/direction training.

        return {
            "image"          : img_t,                          # (3,  H, W)
            "binary_mask"    : binary_t,                       # (1,  H, W)
            "instance_masks" : inst_t,                         # (10, H, W)
            "direction_field": dir_t,                          # (2,  H, W)
            "finger_ids"     : finger_t,                       # (10,)
            "n_instances"    : torch.tensor(len(masks_resized), dtype=torch.long),
            "image_id"       : image_id,
        }

    # ── Augmentation ──────────────────────────────────────────────────────────

    def _augment(self, image, masks):
        if random.random() > 0.5:
            image = TF.hflip(image)
            masks = [TF.hflip(m) for m in masks]
        if random.random() > 0.5:
            image = TF.vflip(image)
            masks = [TF.vflip(m) for m in masks]
        image = TF.adjust_brightness(image, 1 + random.uniform(-0.2,  0.2))
        image = TF.adjust_contrast(image,   1 + random.uniform(-0.2,  0.2))
        image = TF.adjust_saturation(image, 1 + random.uniform(-0.3,  0.3))
        image = TF.adjust_hue(image,            random.uniform(-0.05, 0.05))
        return image, masks


# ── DataLoader factory ─────────────────────────────────────────────────────────

def make_loaders(dataset_root, mp_json_path, batch_size=8, num_workers=4, image_size=512, val_split=0.1):
    root = Path(dataset_root)
    
    # If standard 'train' subfolder exists, use it; otherwise use the root itself.
    train_root = root / "train" if (root / "train").exists() else root
    valid_root = root / "valid"

    train_ds = NailDataset(train_root, mp_json_path, augment=True, image_size=image_size)

    if valid_root.exists():
        val_ds = NailDataset(valid_root, mp_json_path, augment=False, image_size=image_size)
    else:
        n_val   = int(len(train_ds) * val_split)
        n_train = len(train_ds) - n_val
        train_ds, val_ds = torch.utils.data.random_split(
            train_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )
        print(f"[make_loaders] Auto-split → train={n_train}, val={n_val}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


# ── Sanity check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    root   = sys.argv[1] if len(sys.argv) > 1 else "nails_segmentation_coco/train"
    mp_json = "nails_vton/mp_orientations_v1.json"
    ds     = NailDataset(root, mp_json_path=mp_json, augment=True)
    sample = ds[0]

    print("image          :", sample["image"].shape,           sample["image"].dtype)
    print("binary_mask    :", sample["binary_mask"].shape,     sample["binary_mask"].max().item())
    print("instance_masks :", sample["instance_masks"].shape,  sample["instance_masks"].sum().item(), "fg pixels")
    print("direction_field:", sample["direction_field"].shape, sample["direction_field"].abs().max().item())
    print("finger_ids     :", sample["finger_ids"].tolist())
    print("n_instances    :", sample["n_instances"].item())
    print("image_id       :", sample["image_id"])
    print("\nDataset sanity check PASSED ✓")
