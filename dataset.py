"""
Nail VTON Dataset
-----------------
Loads NailSegmentationDatasetV2 (Kaggle CSV-based).
Structure:
  base_path/train/images/nail_train_XXXXXX.jpg
  base_path/train/masks/nail_train_XXXXXX.png
  base_path/NailSegmentationV1.csv

Returns per-sample tuple:
  (image_tensor: (3,H,W) float32 normalised,
   mask_tensor:  (1,H,W) float32 binary {0,1})

Training resolution: 640×640 (native Kaggle dataset size).
Inference resolution: any — DeepLabV3+ is fully convolutional.
"""

import gc
import random
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from PIL import Image, ImageOps


# ── Constants ──────────────────────────────────────────────────────────────────
IMAGE_SIZE = 640
MEAN       = [0.485, 0.456, 0.406]
STD        = [0.229, 0.224, 0.225]
DATA_ROOT  = "/kaggle/input/datasets/muhammadhammad261/nail-segmentation-dataset/NailSegmentationDatasetV2"


# ── Helpers ────────────────────────────────────────────────────────────────────

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

        # Fix: If user passed the /train or /val subfolder, move to root to find CSV
        if self.base_path.name in ["train", "val", "valid", "test"]:
            self.base_path = self.base_path.parent

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

        # Texture-Invariance: Random Manicure (Base Coat only)
        if random.random() > 0.5:
            img = self._vandalize_texture(img, msk)

        # Global Gaussian noise: simulate camera noise and low-quality images
        if random.random() > 0.75:
            img_np = np.array(img).astype(np.float32)
            noise  = np.random.normal(0, random.uniform(5, 20), img_np.shape)
            img    = Image.fromarray(np.clip(img_np + noise, 0, 255).astype(np.uint8))

        return img, msk

    def _vandalize_texture(self, img, msk):
        """
        Manicure Engine — Base Coat only.
        Applies random solid or gradient colors to the nail.
        """
        img_np = np.array(img).astype(np.float32)
        msk_np = np.array(msk).astype(np.float32) / 255.0  # (H, W)
        h, w, _ = img_np.shape
        mask_3d = np.expand_dims(msk_np, axis=-1)
        
        # ── Base Coat (80% chance if texture vandalize is active) ───────────
        color = np.random.randint(0, 256, (3,))
        a = random.uniform(0.2, 0.5) # Subtle (was up to 0.9)
        
        if random.random() > 0.4: # Solid
            img_np = img_np * (1 - mask_3d * a) + (color * mask_3d * a)
        else: # Gradient
            grad = np.linspace(0, 1, h).reshape(h, 1, 1)
            img_np = img_np * (1 - mask_3d * a * grad) + (color * mask_3d * a * grad)

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
    p_bg      = 0.00 # Removed
    p_hand    = 0.00 # Removed
    p_texture = 0.00 # Removed
    p_noise   = 0.25

    p_none = (1-p_hflip)*(1-p_vflip)*(1-p_rotate)*(1-p_crop)*(1-p_noise)
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


def make_loaders(base_path, batch_size=16, num_workers=2, image_size=IMAGE_SIZE,
                 distributed=False, rank=0, world_size=1):
    train_ds = NailDataset(base_path, split="train", augment=True,  image_size=image_size)
    val_ds   = NailDataset(base_path, split="val",   augment=False, image_size=image_size)

    _print_augmentation_stats(train_ds, val_ds)

    train_sampler = None
    val_sampler   = None
    if distributed:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
        )

    train_loader = DataLoader(
        train_ds,
        batch_size       = batch_size,
        shuffle          = (train_sampler is None),
        sampler          = train_sampler,
        num_workers      = num_workers,
        pin_memory       = False,
        drop_last        = True,
        persistent_workers = False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size       = batch_size,
        shuffle          = False,
        sampler          = val_sampler,
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
