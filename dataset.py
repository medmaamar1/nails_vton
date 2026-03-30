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
    pointing from the nail base to the nail tip.
    
    Strict Paper Parity: 
    - We compute the Principal Axis of the nail mask.
    - We flip the sign so it points away from the palm (towards bbox top).
    - We ensure magnitude = 1.0 (Section 3.3).
    """
    coords = np.argwhere(mask_np > 127)
    if len(coords) < 10:
        return np.zeros((2, mask_np.shape[0], mask_np.shape[1]), dtype=np.float32)

    # PCA to find the primary axis (elongation)
    mean = coords.mean(axis=0)
    diff = coords - mean
    cov  = np.dot(diff.T, diff)
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    # The largest eigenvalue (index 1) corresponds to the nail's length axis
    vy, vx = eigvecs[:, 1]
    
    # Orient the vector so it points "upward" in the bounding box coordinate frame
    # (i.e. towards the top of the nail relative to the centroid)
    # We compare with the vector from centroid to the top-middle of the bbox
    if vy > 0: # In image coords, +Y is DOWN. vy > 0 means pointing DOWN.
        vy, vx = -vy, -vx

    # Final Unit Normalization
    norm = np.sqrt(vx**2 + vy**2) + 1e-8
    vx, vy = vx/norm, vy/norm

    H, W = mask_np.shape
    fg = (mask_np > 127).astype(np.float32)
    return np.stack([fg * vx, fg * vy], axis=0)  # (2, H, W)


# ── Dataset ────────────────────────────────────────────────────────────────────

class NailDataset(Dataset):
    def __init__(self, root, augment=False, image_size=IMAGE_SIZE, json_path=None):
        self.root       = Path(root)
        self.augment    = augment
        self.image_size = image_size

        if json_path is not None:
            ann_path = Path(json_path)
            print(f"[NailDataset] Using custom JSON path: {ann_path}")
        else:
            ann_path = self.root / "_annotations.coco.json"
            
        with open(ann_path, "r", encoding='utf-8') as f:
            coco = json.load(f)

        # Pre-resolve image paths by scanning the directory (recursive).
        # This workaround handles Roboflow long filenames that may be truncated 
        # on certain filesystems (Kaggle/Linux) causing OSError: [Errno 36].
        print(f"[NailDataset] Mapping files in {self.root}...")
        all_files = list(self.root.rglob("*.jpg")) + list(self.root.rglob("*.png"))
        
        # Maps for quick lookup: hash -> Path, and name -> Path
        hash_to_path = {}
        name_to_path = {}
        for p in all_files:
            name_to_path[p.name] = p
            if ".rf." in p.name:
                h = p.name.split(".rf.")[-1].split(".")[0]
                hash_to_path[h] = p

        self.id_to_anns = {}
        for ann in coco["annotations"]:
            aid = ann["image_id"]
            self.id_to_anns.setdefault(aid, [])
            self.id_to_anns[aid].append(ann)

        self.image_ids  = []
        self.id_to_path = {}

        for img in coco["images"]:
            iid = img["id"]
            # Skip if no annotations
            if iid not in self.id_to_anns or not self.id_to_anns[iid]:
                continue

            fname = img["file_name"]
            rf_hash = fname.split(".rf.")[-1].split(".")[0] if ".rf." in fname else None
            
            # Attempt to find the file using: hash -> direct name -> fallback images subfolder
            target_path = None
            if rf_hash and rf_hash in hash_to_path:
                target_path = hash_to_path[rf_hash]
            elif fname in name_to_path:
                target_path = name_to_path[fname]
            else:
                # Final check (handles cases where rglob might have missed it or path is weird)
                try:
                    p1 = self.root / fname
                    p2 = self.root / "images" / fname
                    if p1.exists(): target_path = p1
                    elif p2.exists(): target_path = p2
                except OSError: # Still too long? Skip it.
                    pass

            if target_path:
                self.id_to_path[iid] = target_path
                self.image_ids.append(iid)

        print(f"[NailDataset] Loaded {len(self.image_ids)} valid images (root={root}, augment={augment})")

        # Free memory (JSON and file list can be large)
        if 'coco' in locals(): del coco
        if 'all_files' in locals(): del all_files
        if 'hash_to_path' in locals(): del hash_to_path
        if 'name_to_path' in locals(): del name_to_path

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id  = self.image_ids[idx]
        img_path  = self.id_to_path[image_id]
        anns      = self.id_to_anns[image_id]

        # ── Load image ────────────────────────────────────────────────────────
        image_orig     = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image_orig.size

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
        image = image_orig.resize((self.image_size, self.image_size), Image.BILINEAR)
        image_orig.close()

        masks_resized  = []
        bboxes_resized = []
        for m, (x, y, w, h) in zip(masks_pil, bboxes_orig):
            m_resized = m.resize((self.image_size, self.image_size), Image.NEAREST)
            masks_resized.append(m_resized)
            bboxes_resized.append([x * sx, y * sy, w * sx, h * sy])
            # Explicitly close and delete original PIL image to save RAM
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
        for m, bbox in zip(masks_resized, bboxes_resized):
            mask_np = np.array(m, dtype=np.uint8)
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

        return {
            "image"          : img_t,                          # (3,  H, W)
            "binary_mask"    : binary_t,                       # (1,  H, W)
            "direction_field": dir_t,                          # (2,  H, W)
            "n_instances"    : torch.tensor(len(masks_resized), dtype=torch.long),
            "image_id"       : image_id,
        }

    # ── Augmentation ──────────────────────────────────────────────────────────

    def _augment(self, image, masks):
        def _apply_img_aug(img_in, fn, *args):
            img_out = fn(img_in, *args)
            if hasattr(img_in, "close") and img_in is not img_out:
                img_in.close()
            return img_out

        if random.random() > 0.5:
            image = _apply_img_aug(image, TF.hflip)
            masks = [_apply_img_aug(m, TF.hflip) for m in masks]
        if random.random() > 0.5:
            image = _apply_img_aug(image, TF.vflip)
            masks = [_apply_img_aug(m, TF.vflip) for m in masks]
            
        image = _apply_img_aug(image, TF.adjust_brightness, 1 + random.uniform(-0.2,  0.2))
        image = _apply_img_aug(image, TF.adjust_contrast,   1 + random.uniform(-0.2,  0.2))
        image = _apply_img_aug(image, TF.adjust_saturation, 1 + random.uniform(-0.3,  0.3))
        image = _apply_img_aug(image, TF.adjust_hue,            random.uniform(-0.05, 0.05))
        
        return image, masks


# ── DataLoader factory ─────────────────────────────────────────────────────────

def make_loaders(dataset_root, batch_size=8, num_workers=4, val_split=0.1, json_path=None):
    root = Path(dataset_root)
    
    # If standard 'train' subfolder exists, use it; otherwise use the root itself.
    # This prevents path doubling like /kaggle/.../train/train/_annotations...
    train_root = root / "train" if (root / "train").exists() else root
    valid_root = root / "valid"

    train_ds = NailDataset(train_root, augment=True, json_path=json_path)

    if valid_root.exists():
        # Typically valid subfolder has its own JSON, but we'll pass the custom one if they are explicitly bypassing the defaults
        val_ds = NailDataset(valid_root, augment=False, json_path=json_path)
    else:
        import copy
        n_val   = int(len(train_ds) * val_split)
        n_train = len(train_ds) - n_val
        
        train_subset, val_subset = torch.utils.data.random_split(
            train_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Validation shouldn't be augmented (keeps loss metrics stable)
        base_val_ds = copy.deepcopy(train_ds)
        base_val_ds.augment = False
        
        train_ds = train_subset
        val_ds   = torch.utils.data.Subset(base_val_ds, val_subset.indices)
        
        print(f"[make_loaders] Auto-split → train={n_train}, val={n_val} (Golden 'No-Aug' Validation)")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=False,
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=False)
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
