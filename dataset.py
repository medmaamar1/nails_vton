"""
Nail Segmentation Dataset V2
----------------------------
Loads Image/Mask pairs perfectly padded (Letterbox) and Grayscale (Color-Agnostic).
No more direction field, JSONs, or polygons. 100% focused on pixel-perfect binary masks.
"""

import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageOps

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

# ── Constants ──────────────────────────────────────────────────────────────────
IMAGE_SIZE    = 448
MEAN          = [0.485, 0.456, 0.406]
STD           = [0.229, 0.224, 0.225]
DATA_ROOT     = "/kaggle/input/datasets/muhammadhammad261/nail-segmentation-dataset/NailSegmentationDatasetV2"

# ── Dataset ────────────────────────────────────────────────────────────────────

class NailDataset(Dataset):
    def __init__(self, root, split="train", augment=False, image_size=IMAGE_SIZE):
        self.root       = Path(root)
        self.augment    = augment
        self.image_size = image_size
        self.split      = split
        
        # We load strictly from the CSV to ensure perfect Image-Mask alignment
        csv_path = self.root / "NailSegmentationV1.csv"
        df = pd.read_csv(csv_path)
        
        # Standardize split names (handle 'val' vs 'valid')
        if split == "val":
            mask = (df['split'] == "val") | (df['split'] == "valid")
        else:
            mask = df['split'] == split
            
        df = df[mask].reset_index(drop=True)
        
        self.image_paths = df['image_path'].tolist()
        self.mask_paths  = df['mask_path'].tolist()

        print(f"[NailDataset] Loaded {len(self.image_paths)} images for split: {split}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Full absolute paths
        img_path = str(self.root / self.image_paths[idx])
        msk_path = str(self.root / self.mask_paths[idx])

        # ── 1. Load, Auto-Rotate and Letterbox Pad ──
        raw_img = Image.open(img_path).convert("RGB")
        raw_img = ImageOps.exif_transpose(raw_img) # Fix phone-rotation shifts
        # raw_img = TF.to_grayscale(raw_img, num_output_channels=3) # Agnostic Feature (Disabled)
        
        # Load mask exactly the same way to stay 100% synchronized
        raw_msk = Image.open(msk_path).convert("L")
        raw_msk = ImageOps.exif_transpose(raw_msk)

        w, h = raw_img.size
        max_dim = max(w, h)
        
        # Exact floating-point padding coordinates
        pad_x = int((max_dim - w) / 2.0)
        pad_y = int((max_dim - h) / 2.0)
        
        # Create square canvases
        canvas_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
        canvas_msk = Image.new("L", (max_dim, max_dim), 0)
        
        canvas_img.paste(raw_img, (pad_x, pad_y))
        canvas_msk.paste(raw_msk, (pad_x, pad_y))
        
        # ── 2. Resize to 448x448 ──
        # Image gets Bilinear (smooth), Mask gets Nearest (keeps edges sharp before augment)
        image = canvas_img.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask  = canvas_msk.resize((self.image_size, self.image_size), Image.NEAREST)
        
        canvas_img.close(); canvas_msk.close(); raw_img.close(); raw_msk.close()

        # ── 3. Augmentation ──
        if self.augment:
            image, mask = self._augment(image, mask)

        # ── 4. Tensors ──
        img_t = TF.normalize(TF.to_tensor(image), MEAN, STD)
        
        # Mask tensor: Convert [0-255] to [0.0 - 1.0] float32
        mask_np = np.array(mask, dtype=np.float32) / 255.0
        mask_t  = torch.from_numpy(mask_np).unsqueeze(0)   # (1, H, W)
        
        image.close(); mask.close()

        # Phase 4 Simplification: Return ONLY Image and Mask (No Direction)
        return img_t.clone().detach(), mask_t.clone().detach()

    # ── Augmentation ──────────────────────────────────────────────────────────

    def _augment(self, image, mask):
        S = self.image_size

        # Spatial Flips
        if random.random() > 0.5:
            image = TF.hflip(image); mask = TF.hflip(mask)
        if random.random() > 0.5:
            image = TF.vflip(image); mask = TF.vflip(mask)

        # Random Crop & Zoom (vital for single-finger macro shots)
        if random.random() > 0.5:
            scale = random.uniform(0.6, 1.0)
            crop_size = int(S * scale)
            top  = random.randint(0, S - crop_size)
            left = random.randint(0, S - crop_size)
            image = TF.resized_crop(image, top, left, crop_size, crop_size, (S, S), Image.BILINEAR)
            mask  = TF.resized_crop(mask,  top, left, crop_size, crop_size, (S, S), Image.NEAREST)

        # Intensity Jitter (Agnostic version - just brightness and contrast)
        image = TF.adjust_brightness(image, 1 + random.uniform(-0.35, 0.35))
        image = TF.adjust_contrast(image,   1 + random.uniform(-0.35, 0.35))
        
        return image, mask

# ── DataLoader factory ─────────────────────────────────────────────────────────

def make_loaders(dataset_root, batch_size=8, num_workers=4):
    """
    Creates train and val loaders using the CSV splits.
    """
    train_ds = NailDataset(dataset_root, split="train", augment=True)
    val_ds   = NailDataset(dataset_root, split="val",   augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=False, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=2, pin_memory=False)
                              
    return train_loader, val_loader

if __name__ == "__main__":
    # Test script will only work if DATA_ROOT is valid
    try:
        ds = NailDataset(DATA_ROOT, split="train", augment=True)
        img, mask = ds[0]
        print(f"Image: {img.shape} | Mask: {mask.shape}")
        print("✓ SUCCESS: Phase 4 Dataset load complete.")
    except Exception as e:
        print(f"Test failed (Ensure DATA_ROOT exists, usually on Kaggle). Error: {e}")
