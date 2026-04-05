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
import os
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
DATA_ROOT     = "/kaggle/input/datasets/maamarmohamed/nail-segmentation/train"




# ── Helpers ────────────────────────────────────────────────────────────────────

# ── Helpers ────────────────────────────────────────────────────────────────────

def polygon_to_mask(polygon, max_dim, offset_x, offset_y, target_size=IMAGE_SIZE):
    """
    High-Precision Anti-Aliased Masking using 4x Super-Sampling.
    Prevents 'pixellated' jagged edges by drawing at 1792px then downsampling.
    """
    # Supersampling factor 4
    S = target_size * 4 
    mask_large = Image.new("L", (S, S), 0)
    
    if len(polygon) >= 6:
        # Scale factor from raw max_dim to 1792 target
        scale = S / max_dim
        xy = [((x + offset_x) * scale, (y + offset_y) * scale) for x, y in zip(polygon[::2], polygon[1::2])]
        ImageDraw.Draw(mask_large).polygon(xy, fill=255)
    
    # Bilinear downsampling creates a smooth probability edge at 448x448
    return mask_large.resize((target_size, target_size), Image.BILINEAR)





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
    def __init__(self, root, augment=False, image_size=IMAGE_SIZE, json_path=None, orientation_path=None, subset_ids=None):
        self.root       = Path(root)
        self.augment    = augment
        self.image_size = image_size

        self.orientations = {}
        if orientation_path and os.path.exists(orientation_path):
            with open(orientation_path, "r", encoding='utf-8') as f:
                self.orientations = json.load(f)
            print(f"[NailDataset] Loaded orientations from {orientation_path}")

        # Zero-overhead path lookup instead of rglob scanning
        if json_path is not None:
            ann_path = str(Path(json_path))
            print(f"[NailDataset] Using custom JSON path: {ann_path}")
        else:
            ann_path = str(self.root / "_annotations.coco.json")

        with open(ann_path, "r", encoding='utf-8') as f:
            coco = json.load(f)

        self.id_to_anns = {}
        orientation_hits = 0
        for ann in coco["annotations"]:
            aid = ann["image_id"]
            aid_str = str(aid)
            ann_id_str = str(ann.get("id", ""))
            
            # Record annotation even if orientation is missing (for segmentation-only)
            self.id_to_anns.setdefault(aid, [])
            
            has_orient = (aid_str in self.orientations and ann_id_str in self.orientations[aid_str])
            if has_orient:
                orientation_hits += 1

            self.id_to_anns[aid].append({
                "id": ann.get("id"),
                "segmentation": ann.get("segmentation", []),
                "bbox": ann.get("bbox", [0,0,0,0]),
                "has_orientation": has_orient
            })

        self.image_ids  = []
        self.id_to_path = {}

        root_str    = str(self.root)
        images_str  = str(self.root / "images")

        for img in coco["images"]:
            iid   = img["id"]
            fname = img["file_name"]
            
            # SUBSET FILTER: Skip if not in the requested subset
            if subset_ids is not None and iid not in subset_ids:
                continue

            # Skip if no annotations at all
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
        image_id, img_path, anns = self.image_ids[idx], self.id_to_path[self.image_ids[idx]], self.id_to_anns[self.image_ids[idx]]

        # ── Load and Letterbox Pad ──
        raw_img = Image.open(img_path).convert("RGB")
        raw_img = TF.to_grayscale(raw_img, num_output_channels=3) # Agnostic
        w, h = raw_img.size
        max_dim = max(w, h)
        
        # Calculate padding to center the image
        pad_x = (max_dim - w) // 2
        pad_y = (max_dim - h) // 2
        
        # Create square canvas
        canvas = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
        canvas.paste(raw_img, (pad_x, pad_y))
        
        # ── Masks on Square Canvas ──
        masks_resized = []
        for ann in anns:
            seg = ann.get("segmentation", [])
            if not seg or len(seg[0]) < 6: continue
            # Now uses 4x super-sampling for smooth edges
            m_anti_aliased = polygon_to_mask(seg[0], max_dim, pad_x, pad_y, self.image_size)
            masks_resized.append(m_anti_aliased)

        # Resize square canvas to 448x448
        image = canvas.resize((self.image_size, self.image_size), Image.BILINEAR)
        canvas.close(); raw_img.close()

        h_flip, v_flip = False, False
        if self.augment:
            image, masks_resized, h_flip, v_flip = self._augment(image, masks_resized)

        # ── Normalize & Tensors ──
        img_t = TF.normalize(TF.to_tensor(image), MEAN, STD)
        S = self.image_size
        binary_np = np.zeros((S, S), dtype=np.float32)
        dir_np = np.zeros((2, S, S), dtype=np.float32)
        img_orient = self.orientations.get(str(image_id), {})

        for m, ann in zip(masks_resized, anns):
            m_np = np.array(m, dtype=np.float32) / 255.0
            binary_np = np.maximum(binary_np, m_np)
            
            # Direction Field
            aid_str = str(ann.get("id", ""))
            if aid_str in img_orient:
                dx, dy = img_orient[aid_str]
                if h_flip: dx = -dx
                if v_flip: dy = -dy
                mask_fg = m_np > 0.5
                dir_np[0, mask_fg] += dx
                dir_np[1, mask_fg] += dy

        # Re-normalise
        norm = np.sqrt(dir_np[0]**2 + dir_np[1]**2)
        valid = norm > 1e-6
        dir_np[0, valid] /= norm[valid]
        dir_np[1, valid] /= norm[valid]
        
        for m in masks_resized: m.close()
        return (img_t.clone().detach(), torch.from_numpy(binary_np).unsqueeze(0), torch.from_numpy(dir_np))

    # ── Augmentation ──────────────────────────────────────────────────────────

    def _augment(self, image, masks):
        h_flipped = False
        v_flipped = False
        S = self.image_size

        # ── Spatial Flips ─────────────────────────────────────────────────────
        if random.random() > 0.5:
            image = TF.hflip(image)
            masks = [TF.hflip(m) for m in masks]
            h_flipped = True
        if random.random() > 0.5:
            image = TF.vflip(image)
            masks = [TF.vflip(m) for m in masks]
            v_flipped = True

        # ── Random Crop & Zoom ────────────────────────────────────────────────
        if random.random() > 0.5:
            scale = random.uniform(0.6, 1.0)
            crop_size = int(S * scale)
            top  = random.randint(0, S - crop_size)
            left = random.randint(0, S - crop_size)
            image = TF.resized_crop(image, top, left, crop_size, crop_size, (S, S), Image.BILINEAR)
            masks = [TF.resized_crop(m, top, left, crop_size, crop_size, (S, S), Image.NEAREST) for m in masks]

        # ── Intensity Jitter (Agnostic version) ───────────────────────────────
        # Brightness and Contrast are still vital for edge detection in grayscale.
        # Saturation and Hue are now irrelevant as input is already grayscale.
        image = TF.adjust_brightness(image, 1 + random.uniform(-0.35, 0.35))
        image = TF.adjust_contrast(image,   1 + random.uniform(-0.35, 0.35))
        
        return image, masks, h_flipped, v_flipped


# ── DataLoader factory ─────────────────────────────────────────────────────────

def make_loaders(dataset_root, batch_size=8, num_workers=4, val_split=0.1, json_path=None, orientation_path=None):
    root = Path(dataset_root)
    train_root = root / "train" if (root / "train").exists() else root
    valid_root = root / "valid"

    # 1. Subject-Aware Split logic
    ann_path = json_path if json_path else str(train_root / "_annotations.coco.json")
    with open(ann_path, "r", encoding='utf-8') as f:
        coco = json.load(f)

    # Group images by subject ID (filename prefix before roboflow hash)
    subject_to_ids = {}
    for img in coco["images"]:
        fname = img["file_name"]
        # Roboflow format: SubjectName_jpg.rf.hash.jpg or similar
        # We take the part before '.rf.' as the subject identifier
        subject_id = fname.split('.rf.')[0] if '.rf.' in fname else fname.rsplit('.', 1)[0]
        subject_to_ids.setdefault(subject_id, []).append(img["id"])

    subjects = sorted(list(subject_to_ids.keys()))
    random.Random(42).shuffle(subjects)
    
    n_val_subs = int(len(subjects) * val_split)
    val_subjects = set(subjects[:n_val_subs])
    train_subjects = set(subjects[n_val_subs:])

    train_ids = []
    for s in train_subjects: train_ids.extend(subject_to_ids[s])
    val_ids = []
    for s in val_subjects: val_ids.extend(subject_to_ids[s])

    print(f"[make_loaders] Subject-Aware Split: {len(train_subjects)} training subjects ({len(train_ids)} imgs), "
          f"{len(val_subjects)} validation subjects ({len(val_ids)} imgs)")

    train_ds = NailDataset(train_root, augment=True, json_path=json_path, 
                           orientation_path=orientation_path, subset_ids=set(train_ids))

    if valid_root.exists():
        # If a valid folder exists, we assume it's already pre-split or we split it too
        val_ds = NailDataset(valid_root, augment=False, json_path=json_path, 
                             orientation_path=orientation_path)
    else:
        val_ds = NailDataset(train_root, augment=False, json_path=json_path, 
                             orientation_path=orientation_path, subset_ids=set(val_ids))

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
