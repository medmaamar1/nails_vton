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

        # Vertical flip — simulates hand photographed from the top (palm up vs down)
        if random.random() > 0.5:
            img = TF.vflip(img)
            msk = TF.vflip(msk)

        # Small rotation — hand held at a slight angle
        if random.random() > 0.4:
            angle = random.uniform(-20, 20)
            img = TF.rotate(img, angle, interpolation=Image.BILINEAR, fill=0)
            msk = TF.rotate(msk, angle, interpolation=Image.NEAREST,  fill=0)

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
        if random.random() > 0.5:
            img = self._vandalize_background(img, msk)

        # Hand/skin distortion: overlay semi-transparent shapes on the hand area
        if random.random() > 0.5:
            img = self._vandalize_hand(img, msk)

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

    def _pick_alpha(self):
        """100% chance fully opaque (no transparency)."""
        return 1.0

    def _vandalize_background(self, img, msk):
        """Paint random shapes, lines, and noise onto the background."""
        img_np  = np.array(img).astype(np.float32)
        msk_np  = np.array(msk).astype(np.float32) / 255.0
        h, w, _ = img_np.shape
        bg_mask = np.expand_dims(1.0 - msk_np, axis=-1)

        # Dense Gaussian noise over the entire background
        noise_strength = random.uniform(15, 45)
        noise = np.random.normal(0, noise_strength, img_np.shape)
        img_np = img_np + noise * (1.0 - np.expand_dims(msk_np, -1))

        # Heavily increased number of patches/shapes to destroy background
        num_patches = random.randint(30, 60)
        for _ in range(num_patches):
            color  = np.random.randint(0, 256, (3,))
            a      = self._pick_alpha()
            dtype  = random.choice(["rect", "circle", "line", "cross", "filled_rect"])
            s_mask = np.zeros((h, w, 1))

            if dtype in ("rect", "filled_rect"):
                rw, rh = random.randint(20, 100), random.randint(20, 100)
                ry     = random.randint(0, max(1, h - rh))
                rx     = random.randint(0, max(1, w - rw))
                s_mask[ry:ry+rh, rx:rx+rw, :] = 1.0

            elif dtype == "circle":
                radius = random.randint(15, 60)
                ry, rx = random.randint(0, h), random.randint(0, w)
                yy, xx = np.ogrid[:h, :w]
                s_mask = ((yy - ry)**2 + (xx - rx)**2 <= radius**2).astype(float)
                s_mask = np.expand_dims(s_mask, -1)

            elif dtype == "line":
                angle  = random.uniform(0, np.pi)
                thick  = random.randint(4, 16)
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                yy, xx = np.ogrid[:h, :w]
                proj   = xx * cos_a + yy * sin_a
                offset = random.randint(0, h + w)
                s_mask = (np.abs(proj - offset) < thick).astype(float)
                s_mask = np.expand_dims(s_mask, -1)

            elif dtype == "cross":
                cy, cx = random.randint(0, h), random.randint(0, w)
                arm    = random.randint(15, 50)
                thick  = random.randint(4, 12)
                s_mask[max(0,cy-thick):min(h,cy+thick), max(0,cx-arm):min(w,cx+arm), :] = 1.0
                s_mask[max(0,cy-arm):min(h,cy+arm), max(0,cx-thick):min(w,cx+thick), :] = 1.0

            piece  = s_mask * bg_mask * a
            img_np = img_np * (1 - piece) + (color * piece)

        return Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))

    def _vandalize_hand(self, img, msk):
        """Overlay shapes and noise on the hand/skin region to destroy texture cues."""
        img_np  = np.array(img).astype(np.float32)
        msk_np  = np.array(msk).astype(np.float32) / 255.0
        h, w, _ = img_np.shape
        hand_mask = np.expand_dims(1.0 - msk_np, axis=-1)

        # Gaussian noise scoped to the hand area
        noise_strength = random.uniform(10, 35)
        noise = np.random.normal(0, noise_strength, img_np.shape)
        img_np = img_np + noise * hand_mask

        # Heavily increased number of patches/shapes to destroy hand texture
        num_patches = random.randint(20, 40)
        for _ in range(num_patches):
            color  = np.random.randint(0, 256, (3,))
            a      = self._pick_alpha()
            dtype  = random.choice(["rect", "circle", "line", "stripe_thin", "filled_rect"])
            s_mask = np.zeros((h, w, 1))

            if dtype in ("rect", "filled_rect"):
                rw, rh = random.randint(15, 60), random.randint(15, 60)
                ry     = random.randint(0, max(1, h - rh))
                rx     = random.randint(0, max(1, w - rw))
                s_mask[ry:ry+rh, rx:rx+rw, :] = 1.0

            elif dtype == "circle":
                radius = random.randint(8, 35)
                ry, rx = random.randint(0, h), random.randint(0, w)
                yy, xx = np.ogrid[:h, :w]
                s_mask = ((yy - ry)**2 + (xx - rx)**2 <= radius**2).astype(float)
                s_mask = np.expand_dims(s_mask, -1)

            elif dtype == "line":
                angle  = random.uniform(0, np.pi)
                thick  = random.randint(3, 10)
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                yy, xx = np.ogrid[:h, :w]
                proj   = xx * cos_a + yy * sin_a
                offset = random.randint(0, h + w)
                s_mask = (np.abs(proj - offset) < thick).astype(float)
                s_mask = np.expand_dims(s_mask, -1)

            elif dtype == "stripe_thin":
                angle  = random.uniform(0, np.pi)
                thick  = random.randint(1, 4)
                gap    = random.randint(8, 25)
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                yy, xx = np.ogrid[:h, :w]
                proj   = xx * cos_a + yy * sin_a
                offset = random.randint(0, h + w)
                s_mask = (
                    (np.abs(proj - offset) < thick) |
                    (np.abs(proj - offset - gap) < thick)
                ).astype(float)
                s_mask = np.expand_dims(s_mask, -1)

            piece  = s_mask * hand_mask * a
            img_np = img_np * (1 - piece) + (color * piece)

        return Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))


# ── DataLoader factory ─────────────────────────────────────────────────────────

def _print_augmentation_stats(train_ds, val_ds):
    n_train = len(train_ds)
    n_val   = len(val_ds)
    n_total = n_train + n_val

    def count(p):
        return round(n_train * p)

    p_hflip   = 0.50
    p_vflip   = 0.50
    p_rotate  = 0.60
    p_crop    = 0.50
    p_bg      = 0.50
    p_hand    = 0.50
    p_texture = 0.50
    p_noise   = 0.25

    p_none = (1-p_hflip)*(1-p_vflip)*(1-p_rotate)*(1-p_crop)*(1-p_bg)*(1-p_hand)*(1-p_texture)*(1-p_noise)
    p_any  = 1.0 - p_none

    print(f"\n{'─'*60}")
    print(f"  Dataset            train={n_train}  val={n_val}  total={n_total}")
    print(f"  Val set            never augmented")
    print(f"")
    print(f"  Expected per epoch (train only):")
    print(f"    any augmentation       ~{count(p_any):>5} / {n_train}  ({p_any*100:.1f}%)")
    print(f"    horizontal flip        ~{count(p_hflip):>5} / {n_train}  ({p_hflip*100:.0f}%)")
    print(f"    vertical flip          ~{count(p_vflip):>5} / {n_train}  ({p_vflip*100:.0f}%)")
    print(f"    rotation ±20°          ~{count(p_rotate):>5} / {n_train}  ({p_rotate*100:.0f}%)")
    print(f"    random crop/zoom       ~{count(p_crop):>5} / {n_train}  ({p_crop*100:.0f}%)")
    print(f"    background vandalize   ~{count(p_bg):>5} / {n_train}  ({p_bg*100:.0f}%,  8–18 shapes + noise)")
    print(f"    hand vandalize         ~{count(p_hand):>5} / {n_train}  ({p_hand*100:.0f}%,  6–14 shapes + noise)")
    print(f"    nail texture/manicure  ~{count(p_texture):>5} / {n_train}  ({p_texture*100:.0f}%)")
    print(f"    global noise           ~{count(p_noise):>5} / {n_train}  ({p_noise*100:.0f}%)")
    print(f"    opaque shapes (α=1.0)  ~30% of every shape drawn")
    print(f"{'─'*60}\n")


def make_loaders(base_path, batch_size=16, num_workers=2, image_size=IMAGE_SIZE):
    train_ds = NailDataset(base_path, split="train", augment=True,  image_size=image_size)
    val_ds   = NailDataset(base_path, split="val",   augment=False, image_size=image_size)

    _print_augmentation_stats(train_ds, val_ds)

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
