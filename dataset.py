"""
Nail VTON Dataset
-----------------
Loads Roboflow v50 COCO Segmentation JSON.
Produces per-image:
  1. image           : (3, H, W)       float32 normalised
  2. binary_mask     : (1, H, W)       float32  — union of all nail masks
  3. direction_field : (2, H, W)       float32  — unit vector base→tip per nail pixel
  4. n_instances     : scalar          int64
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
IMAGE_SIZE    = 448
MEAN          = [0.485, 0.456, 0.406]
STD           = [0.229, 0.224, 0.225]




# ── Helpers ────────────────────────────────────────────────────────────────────

def polygon_to_mask(polygon, height, width):
    """Flat COCO polygon [x1,y1,x2,y2,...] → binary PIL mask."""
    mask = Image.new("L", (width, height), 0)
    if len(polygon) >= 6:
        xy = list(zip(polygon[::2], polygon[1::2]))
        ImageDraw.Draw(mask).polygon(xy, fill=255)
    return mask





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
    def __init__(self, root, augment=False, image_size=IMAGE_SIZE, json_path=None):
        self.root       = Path(root)
        self.augment    = augment
        self.image_size = image_size

        # Zero-overhead path lookup instead of rglob scanning
        if json_path is not None:
            ann_path = str(Path(json_path))
            print(f"[NailDataset] Using custom JSON path: {ann_path}")
        else:
            ann_path = str(self.root / "_annotations.coco.json")

        import os
        with open(ann_path, "r", encoding='utf-8') as f:
            coco = json.load(f)

        self.id_to_anns = {}
        for ann in coco["annotations"]:
            aid = ann["image_id"]
            self.id_to_anns.setdefault(aid, [])
            # Forensic memory reduction: only keep segmentation points and bbox
            self.id_to_anns[aid].append({
                "segmentation": ann.get("segmentation", []),
                "bbox": ann.get("bbox", [0,0,0,0])
            })

        self.image_ids  = []
        self.id_to_path = {}

        # Look in the root and 'images' subfolder for each image in the JSON
        # This is surgically precise and avoids scanning thousands of unrelated files.
        root_str    = str(self.root)
        images_str  = str(self.root / "images")

        for img in coco["images"]:
            iid   = img["id"]
            fname = img["file_name"]
            
            # Skip if no annotations
            if iid not in self.id_to_anns or not self.id_to_anns[iid]:
                continue

            p1 = os.path.join(root_str, fname)
            p2 = os.path.join(images_str, fname)

            if os.path.exists(p1):
                self.id_to_path[iid] = p1
                self.image_ids.append(iid)
            elif os.path.exists(p2):
                self.id_to_path[iid] = p2
                self.image_ids.append(iid)
            else:
                # Final fallback for Roboflow hash-mismatches (rare but possible)
                continue

        print(f"[NailDataset] Loaded {len(self.image_ids)} images (root={root_str})")

        # Explicitly toast the coco object to free RAM
        del coco
        import gc
        gc.collect()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id  = self.image_ids[idx]
        img_path  = self.id_to_path[image_id]
        anns      = self.id_to_anns[image_id]

        # ── Load image ────────────────────────────────────────────────────────
        image     = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        # ── Build per-nail masks at original resolution ───────────────────────
        masks_pil   = []
        bboxes_orig = []

        for ann in anns:
            seg = ann.get("segmentation", [])
            if not seg or len(seg[0]) < 6:
                continue
            masks_pil.append(polygon_to_mask(seg[0], orig_h, orig_w))
            bboxes_orig.append(ann["bbox"])

        # ── Resize to image_size ──────────────────────────────────────────────
        sx = self.image_size / orig_w
        sy = self.image_size / orig_h
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        masks_resized  = []
        bboxes_resized = []
        for m, (x, y, w, h) in zip(masks_pil, bboxes_orig):
            m_resized = m.resize((self.image_size, self.image_size), Image.NEAREST)
            masks_resized.append(m_resized)
            bboxes_resized.append([x * sx, y * sy, w * sx, h * sy])
            # Explicitly close original to save RAM
            m.close()

        del masks_pil

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
        binary_t = torch.from_numpy(binary_np).clone().unsqueeze(0)   # (1, H, W)

        # ── Direction field ───────────────────────────────────────────────────
        dir_np = np.zeros((2, S, S), dtype=np.float32)
        
        # Get orientation mapping for this image if loaded
        img_orientations = self.orientations.get(str(image_id), {}) if hasattr(self, 'orientations') else {}

        for m, bbox, ann in zip(masks_resized, bboxes_resized, anns):
            mask_np = np.array(m, dtype=np.uint8)
            ann_id_str = str(ann.get("id", ""))
            
            if ann_id_str in img_orientations:
                # Use ground-truth orientation [dx, dy]
                dx, dy = img_orientations[ann_id_str]
                
                # Create a uniform directional vector for the foreground area
                vector_field = np.zeros((2, S, S), dtype=np.float32)
                vector_field[0, mask_np > 0] = dx
                vector_field[1, mask_np > 0] = dy
                dir_np += vector_field
            else:
                dir_np += compute_direction_field(mask_np, bbox)

        # Re-normalise pixels touched by >1 nail (overlap edge case)
        norm  = np.sqrt(dir_np[0] ** 2 + dir_np[1] ** 2)
        valid = norm > 1e-6
        dir_np[0, valid] /= norm[valid]
        dir_np[1, valid] /= norm[valid]
        dir_t = torch.from_numpy(dir_np).clone()                       # (2, H, W)

        # Cleanup explicitly to prevent multiprocess IPC leaks
        image.close()
        for m in masks_resized:
            m.close()

        # Phase 6 Lockdown: Minimal 3-tensor Tuple (img, bin, dir)
        # Bypasses all hidden collation caches.
        return (
            img_t.clone().detach(),
            binary_t.clone().detach(),
            dir_t.clone().detach()
        )

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

def make_loaders(dataset_root, batch_size=8, num_workers=4, val_split=0.1, json_path=None, orientation_path=None):
    root = Path(dataset_root)
    
    # If standard 'train' subfolder exists, use it; otherwise use the root itself.
    # This prevents path doubling like /kaggle/.../train/train/_annotations...
    train_root = root / "train" if (root / "train").exists() else root
    valid_root = root / "valid"

    train_ds = NailDataset(train_root, augment=True, json_path=json_path, orientation_path=orientation_path)

    if valid_root.exists():
        val_ds = NailDataset(valid_root, augment=False, json_path=json_path, orientation_path=orientation_path)
    else:
        n_val   = int(len(train_ds) * val_split)
        n_train = len(train_ds) - n_val
        train_ds, val_ds = torch.utils.data.random_split(
            train_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )
        print(f"[make_loaders] Auto-split → train={n_train}, val={n_val}")

    # num_workers=2 with persistent_workers=False provides the best memory isolation
    # as the process heap is destroyed more reliably than the main thread.
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=False,
                              drop_last=True, persistent_workers=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=False,
                              persistent_workers=False)
    return train_loader, val_loader


# ── Sanity check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    root   = sys.argv[1] if len(sys.argv) > 1 else "data/train"
    ds     = NailDataset(root, augment=True)
    sample = ds[0]

    print("image          :", sample["image"].shape,           sample["image"].dtype)
    print("binary_mask    :", sample["binary_mask"].shape,     sample["binary_mask"].max().item())
    print("direction_field:", sample["direction_field"].shape, sample["direction_field"].abs().max().item())
    print("n_instances    :", sample["n_instances"].item())
    print("image_id       :", sample["image_id"])
    print("\nDataset sanity check PASSED ✓")
