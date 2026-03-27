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


def assign_finger_ids(bboxes):
    """
    Given a list of bboxes [[x,y,w,h], ...] for one image, return a list of
    finger label codes of the same length.

    Algorithm:
      1. Compute centroid (cx, cy) and area for each nail.
      2. Identify thumb: largest area AND cy > median_cy of all nails.
         If no nail satisfies both, fall back to largest area alone.
      3. Sort remaining nails by cx (left → right).
         Assign labels left-to-right: pinky(5) → ring(4) → middle(3) → index(2).
         If fewer than 4 remain, assign from pinky inward.
    """
    n = len(bboxes)
    if n == 0:
        return []

    cx_list = [x + w / 2.0 for x, y, w, h in bboxes]
    cy_list = [y + h / 2.0 for x, y, w, h in bboxes]
    area_list = [w * h for x, y, w, h in bboxes]

    median_cy = sorted(cy_list)[len(cy_list) // 2]

    # Find thumb index
    thumb_idx = None
    best_area = -1
    for i, (area, cy) in enumerate(zip(area_list, cy_list)):
        if cy > median_cy and area > best_area:
            best_area = area
            thumb_idx = i
    if thumb_idx is None:                          # fallback: just largest
        thumb_idx = int(np.argmax(area_list))

    # Remaining nails sorted by cx
    remaining = [(i, cx_list[i]) for i in range(n) if i != thumb_idx]
    remaining.sort(key=lambda t: t[1])             # left → right

    # Label map: position 0=leftmost → pinky, last=rightmost → index
    pos_to_label = [FINGER_PINKY, FINGER_RING, FINGER_MIDDLE, FINGER_INDEX]

    labels = [FINGER_UNUSED] * n
    labels[thumb_idx] = FINGER_THUMB
    for pos, (orig_idx, _) in enumerate(remaining):
        if pos < len(pos_to_label):
            labels[orig_idx] = pos_to_label[pos]
        else:
            labels[orig_idx] = FINGER_UNUSED       # >4 non-thumb nails (rare)

    return labels


def compute_direction_field(mask_np, bbox):
    """
    Single-nail direction field: each foreground pixel gets a unit vector
    pointing from bbox bottom-centre (base) to bbox top-centre (tip).

    Returns (2, H, W) float32.  All-zero outside the mask.
    """
    H, W = mask_np.shape
    x, y, w, h = bbox

    vx = 0.0                        # tip and base share the same x-centre
    vy = -(h)                       # tip is above base → negative y direction

    norm = math.sqrt(vx * vx + vy * vy)
    if norm < 1e-6:
        return np.zeros((2, H, W), dtype=np.float32)

    ux, uy = vx / norm, vy / norm   # unit vector (always (0, -1) for vertical nails)

    fg = (mask_np > 127).astype(np.float32)
    dx = fg * ux
    dy = fg * uy
    return np.stack([dx, dy], axis=0)              # (2, H, W)


# ── Dataset ────────────────────────────────────────────────────────────────────

class NailDataset(Dataset):
    def __init__(self, root, augment=False, image_size=IMAGE_SIZE):
        self.root       = Path(root)
        self.augment    = augment
        self.image_size = image_size

        ann_path = self.root / "_annotations.coco.json"
        with open(ann_path, "r") as f:
            coco = json.load(f)

        self.id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}

        self.id_to_anns = {}
        for ann in coco["annotations"]:
            iid = ann["image_id"]
            self.id_to_anns.setdefault(iid, [])
            self.id_to_anns[iid].append(ann)

        self.image_ids = [
            iid for iid in self.id_to_file
            if iid in self.id_to_anns and len(self.id_to_anns[iid]) > 0
        ]

        print(f"[NailDataset] {len(self.image_ids)} images  "
              f"(root={root}, augment={augment})")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id  = self.image_ids[idx]
        file_name = self.id_to_file[image_id]
        anns      = self.id_to_anns[image_id]

        # ── Load image ────────────────────────────────────────────────────────
        image          = Image.open(self.root / "images" / file_name).convert("RGB")
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

        # ── Finger identity (from original-scale bboxes) ──────────────────────
        finger_labels = assign_finger_ids(bboxes_orig)

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
        dir_np = np.zeros((2, S, S), dtype=np.float32)
        for m, bbox in zip(masks_resized, bboxes_resized):
            mask_np = np.array(m, dtype=np.uint8)
            dir_np += compute_direction_field(mask_np, bbox)

        # Re-normalise pixels touched by >1 nail (overlap edge case)
        norm  = np.sqrt(dir_np[0] ** 2 + dir_np[1] ** 2)
        valid = norm > 1e-6
        dir_np[0, valid] /= norm[valid]
        dir_np[1, valid] /= norm[valid]
        dir_t = torch.from_numpy(dir_np)                       # (2, H, W)

        # ── Finger id tensor (10,) — 0 for unused slots ───────────────────────
        finger_t = torch.zeros(MAX_INSTANCES, dtype=torch.long)
        for i, lbl in enumerate(finger_labels):
            finger_t[i] = lbl

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

def make_loaders(dataset_root, batch_size=8, num_workers=4, val_split=0.1):
    train_root = Path(dataset_root) / "train"
    valid_root = Path(dataset_root) / "valid"

    train_ds = NailDataset(train_root, augment=True)

    if valid_root.exists():
        val_ds = NailDataset(valid_root, augment=False)
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
    root   = sys.argv[1] if len(sys.argv) > 1 else "data/train"
    ds     = NailDataset(root, augment=True)
    sample = ds[0]

    print("image          :", sample["image"].shape,           sample["image"].dtype)
    print("binary_mask    :", sample["binary_mask"].shape,     sample["binary_mask"].max().item())
    print("instance_masks :", sample["instance_masks"].shape,  sample["instance_masks"].sum().item(), "fg pixels")
    print("direction_field:", sample["direction_field"].shape, sample["direction_field"].abs().max().item())
    print("finger_ids     :", sample["finger_ids"].tolist())
    print("n_instances    :", sample["n_instances"].item())
    print("image_id       :", sample["image_id"])
    print("\nDataset sanity check PASSED ✓")
