"""
Nail VTON Dataset
-----------------
Loads NailSegmentationDatasetV2 (Kaggle CSV-based).
Structure:
  base_path/train/images/nail_train_XXXXXX.jpg
  base_path/train/masks/nail_train_XXXXXX.png
  base_path/NailSegmentationV1.csv

Returns per-sample tuple:
  (image_tensor: (3,H,W) float32 normalised grayscale,
   mask_tensor:  (1,H,W) float32 binary {0,1})
"""

import gc
import random
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps


# ── Constants ──────────────────────────────────────────────────────────────────
IMAGE_SIZE = 448
MEAN       = [0.485, 0.456, 0.406]
STD        = [0.229, 0.224, 0.225]
DATA_ROOT  = "/kaggle/input/datasets/muhammadhammad261/nail-segmentation-dataset/NailSegmentationDatasetV2"


# ── Dataset ────────────────────────────────────────────────────────────────────

class NailDataset(Dataset):
    """
    Memory-safe image/mask pair loader.
    Stores only path strings in RAM — zero PIL or tensor caching.
    """
    def __init__(self, base_path, split="train", augment=False, image_size=IMAGE_SIZE):
        self.base_path  = Path(base_path)
        self.augment    = augment
        self.image_size = image_size

        csv_path = self.base_path / "NailSegmentationV1.csv"
        df = pd.read_csv(csv_path)
        df['split'] = df['split'].str.strip().str.lower()
        
        # Auto-match 'val', 'valid', or 'validation'
        if split == "val":
            mask = df['split'].isin(['val', 'valid', 'validation'])
        else:
            mask = df['split'] == split
            
        df = df[mask].reset_index(drop=True)

        if len(df) == 0:
            print(f"[Warning] No samples found for split '{split}'. Unique values in CSV: {list(pd.read_csv(csv_path)['split'].unique())}")

        # Store only strings — no PIL, no tensors in __init__
        def fix_path(p):
            # The CSV uses 'valid/' but the folder is named 'val/'
            if isinstance(p, str) and p.startswith("valid/"):
                return p.replace("valid/", "val/", 1)
            return p

        self.image_paths = [str(self.base_path / fix_path(row["image_path"])) for _, row in df.iterrows()]
        self.mask_paths  = [str(self.base_path / fix_path(row["mask_path"]))  for _, row in df.iterrows()]

        del df
        gc.collect()

        print(f"[NailDataset] split={split} | {len(self.image_paths)} samples loaded.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # ── Load image (grayscale agnostic) ───────────────────────────────────
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = ImageOps.exif_transpose(img)          # Fix phone rotation
        # img = TF.to_grayscale(img, num_output_channels=3)  # Removed for Full RGB Training

        # ── Load mask ─────────────────────────────────────────────────────────
        msk = Image.open(self.mask_paths[idx]).convert("L")  # 1-channel

        # ── Resize both to target size ─────────────────────────────────────────
        # Images are already 640x640, but we allow custom image_size
        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            msk = msk.resize((self.image_size, self.image_size), Image.NEAREST)

        # ── Augmentation ──────────────────────────────────────────────────────
        if self.augment:
            img, msk = self._augment(img, msk)

        # ── To Tensors ────────────────────────────────────────────────────────
        img_t  = TF.normalize(TF.to_tensor(img), MEAN, STD)       # (3, H, W)
        msk_np = np.array(msk, dtype=np.float32) / 255.0          # 0.0 or 1.0
        msk_t  = torch.from_numpy(msk_np).unsqueeze(0)            # (1, H, W)

        # Binarize: treat any pixel > 0.5 as nail
        msk_t = (msk_t > 0.5).float()

        # ── Explicit cleanup ──────────────────────────────────────────────────
        img.close()
        msk.close()

        return img_t.clone(), msk_t.clone()

    # ── Augmentation ──────────────────────────────────────────────────────────

    def _augment(self, img, msk):
        S = self.image_size

        # Horizontal flip
        if random.random() > 0.5:
            img = TF.hflip(img)
            msk = TF.hflip(msk)

        # Random crop & zoom (simulates close-ups)
        if random.random() > 0.5:
            scale     = random.uniform(0.65, 1.0)
            crop_size = int(S * scale)
            top       = random.randint(0, S - crop_size)
            left      = random.randint(0, S - crop_size)
            img = TF.resized_crop(img, top, left, crop_size, crop_size, (S, S), Image.BILINEAR)
            msk = TF.resized_crop(msk, top, left, crop_size, crop_size, (S, S), Image.NEAREST)

        # Brightness & contrast jitter (grayscale-safe — no hue/saturation)
        img = TF.adjust_brightness(img, 1 + random.uniform(-0.35, 0.35))
        img = TF.adjust_contrast(img,   1 + random.uniform(-0.35, 0.35))

        # Anatomy-First: Random Blur to smear manicures and ignore art
        if random.random() > 0.4:
            kernel = random.choice([3, 5])
            img = TF.gaussian_blur(img, kernel_size=(kernel, kernel))

        # Texture-Invariance: Vandalize the nail area with noise/patterns
        # 50% Sane (Clean) / 50% Randomized (Vandalized)
        if random.random() > 0.5:
            img = self._vandalize_texture(img, msk)

        # Background vandalization: force model to ignore noisy non-nail regions
        if random.random() > 0.6:
            img = self._vandalize_background(img, msk)

        # Global Gaussian noise: simulate camera noise and low-quality images
        if random.random() > 0.75:
            img_np = np.array(img).astype(np.float32)
            noise  = np.random.normal(0, random.uniform(5, 20), img_np.shape)
            img    = Image.fromarray(np.clip(img_np + noise, 0, 255).astype(np.uint8))

        return img, msk

    def _vandalize_texture(self, img, msk):
        """
        Sequential Layered Manicure Engine.
        Step 1: Paint Base Coat. 
        Step 2: Paint Art (on top of Base Coat).
        """
        img_np = np.array(img).astype(np.float32)
        msk_np = np.array(msk).astype(np.float32) / 255.0  # (H, W)
        h, w, _ = img_np.shape
        mask_3d = np.expand_dims(msk_np, axis=-1)
        
        # ── Step 1: Base Coat (80% chance) ───────────────────────────────────
        if random.random() > 0.2:
            color = np.random.randint(0, 256, (3,))
            a = random.uniform(0.4, 0.9)
            if random.random() > 0.4: # Solid
                img_np = img_np * (1 - mask_3d * a) + (color * mask_3d * a)
            else: # Gradient
                grad = np.linspace(0, 1, h).reshape(h, 1, 1)
                img_np = img_np * (1 - mask_3d * a * grad) + (color * mask_3d * a * grad)

        # ── Step 2: High-Density Art Overlays (Glitter, Dots, Rects, Stripes) ──
        if random.random() > 0.2:
            num_designs = random.randint(5, 15) # High density
            for _ in range(num_designs):
                dtype = random.choice(["rect", "star", "stripe", "circle"])
                color = np.random.randint(0, 256, (3,))
                a     = random.uniform(0.6, 0.95)
                
                s_mask = np.zeros((h, w, 1))
                
                if dtype == "rect":
                    rw, rh = random.randint(5, 15), random.randint(5, 15) # Smaller
                    ry, rx = random.randint(0, h-rh), random.randint(0, w-rw)
                    s_mask[ry:ry+rh, rx:rx+rw, :] = 1.0
                
                elif dtype == "circle":
                    radius = random.randint(3, 8)
                    ry, rx = random.randint(0, h), random.randint(0, w)
                    yy, xx = np.ogrid[:h, :w]
                    dist = (yy - ry)**2 + (xx - rx)**2
                    s_mask = (dist <= radius**2).astype(float)
                    s_mask = np.expand_dims(s_mask, -1)
                
                elif dtype == "stripe":
                    angle = random.uniform(0, np.pi)
                    cos_a, sin_a = np.cos(angle), np.sin(angle)
                    thick = random.randint(2, 4)
                    yy, xx = np.ogrid[:h, :w]
                    proj = xx * cos_a + yy * sin_a
                    s_mask = (np.abs(proj - random.randint(0, h+w)) < thick).astype(float)
                    s_mask = np.expand_dims(s_mask, -1)

                elif dtype == "star":
                    ry, rx = random.randint(5, h-5), random.randint(5, w-5)
                    size = random.randint(5, 10)
                    s_mask[ry-size:ry+size, rx-1:rx+1, :] = 1.0
                    s_mask[ry-1:ry+1, rx-size:rx+size, :] = 1.0

                # Bake ON TOP
                final_piece_mask = s_mask * mask_3d * a
                img_np = img_np * (1 - final_piece_mask) + (color * final_piece_mask)

        return Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))

    def _vandalize_background(self, img, msk):
        """Paint random color patches onto non-nail background regions.
        Prevents the model from using texture shortcuts (smooth skin = background)."""
        img_np  = np.array(img).astype(np.float32)
        msk_np  = np.array(msk).astype(np.float32) / 255.0
        h, w, _ = img_np.shape
        bg_mask = np.expand_dims(1.0 - msk_np, axis=-1)  # non-nail area

        num_patches = random.randint(3, 8)
        for _ in range(num_patches):
            color = np.random.randint(0, 256, (3,))
            a     = random.uniform(0.2, 0.55)
            dtype = random.choice(["rect", "circle"])
            s_mask = np.zeros((h, w, 1))

            if dtype == "rect":
                rw, rh = random.randint(10, 50), random.randint(10, 50)
                ry     = random.randint(0, max(1, h - rh))
                rx     = random.randint(0, max(1, w - rw))
                s_mask[ry:ry+rh, rx:rx+rw, :] = 1.0
            else:  # circle
                radius = random.randint(8, 30)
                ry, rx = random.randint(0, h), random.randint(0, w)
                yy, xx = np.ogrid[:h, :w]
                s_mask = ((yy - ry)**2 + (xx - rx)**2 <= radius**2).astype(float)
                s_mask = np.expand_dims(s_mask, -1)

            piece = s_mask * bg_mask * a
            img_np = img_np * (1 - piece) + (color * piece)

        return Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))


# ── DataLoader factory ─────────────────────────────────────────────────────────

def make_loaders(base_path, batch_size=16, num_workers=2, image_size=IMAGE_SIZE):
    train_ds = NailDataset(base_path, split="train", augment=True,  image_size=image_size)
    val_ds   = NailDataset(base_path, split="val",   augment=False, image_size=image_size)

    train_loader = DataLoader(
        train_ds,
        batch_size       = batch_size,
        shuffle          = True,
        num_workers      = num_workers,
        pin_memory       = False,
        drop_last        = True,
        persistent_workers = False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size       = batch_size,
        shuffle          = False,
        num_workers      = num_workers,
        pin_memory       = False,
        persistent_workers = False,
    )
    return train_loader, val_loader


# ── Sanity check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else DATA_ROOT
    ds   = NailDataset(root, split="train", augment=True)

    img_t, msk_t = ds[0]
    print(f"image : {img_t.shape}  dtype={img_t.dtype}")
    print(f"mask  : {msk_t.shape}  max={msk_t.max().item()}  unique={msk_t.unique()}")
    print("\nDataset sanity check PASSED ✓")
