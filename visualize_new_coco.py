"""
visualize_new_coco.py
----------------------
Visualize the augmentation pipeline from the current NailDataset.

For each sampled image shows a 1-row grid:
  Col 0 : Original (no augmentation) + mask overlay (green)
  Col 1 : Augmented variant #1
  Col 2 : Augmented variant #2
  Col 3 : Augmented variant #3

The three augmented variants are independently sampled from the same index,
so you can see the full range of what each image can look like during training
(nail vandalization, background patches, gaussian noise, blur, crop/zoom, hflip).

Usage:
    python visualize_new_coco.py
    python visualize_new_coco.py --data_root /path/to/data --n_samples 8
    python visualize_new_coco.py --split val --n_samples 4
"""

import argparse
import random
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import NailDataset, DATA_ROOT, MEAN, STD


# ── Helpers ────────────────────────────────────────────────────────────────────

def denormalize(tensor):
    """Normalized (3,H,W) tensor → uint8 numpy (H,W,3)."""
    mean = np.array(MEAN).reshape(3, 1, 1)
    std  = np.array(STD).reshape(3, 1, 1)
    img  = tensor.numpy() * std + mean
    return np.clip(img * 255, 0, 255).astype(np.uint8).transpose(1, 2, 0)


def overlay_mask(img_np, mask_np, color=(50, 220, 80), alpha=0.45):
    """Blend a green mask overlay onto the image."""
    out = img_np.astype(np.float32).copy()
    nail_px = mask_np > 0.5
    for c, col in enumerate(color):
        out[:, :, c] = np.where(nail_px,
                                out[:, :, c] * (1 - alpha) + col * alpha,
                                out[:, :, c])
    return out.astype(np.uint8)


def nail_coverage(mask_np):
    """Return % of pixels that are nail."""
    return mask_np.mean() * 100


# ── Main ───────────────────────────────────────────────────────────────────────

def visualize_augmentation(data_root, split="train", n_samples=6, out_dir="debug_augmentation"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_clean = NailDataset(data_root, split=split, augment=False)
    ds_aug   = NailDataset(data_root, split=split, augment=True)

    if len(ds_clean) == 0:
        print(f"[ERROR] No samples found in split='{split}'. Check your data_root.")
        return

    n_samples = min(n_samples, len(ds_clean))
    indices   = random.sample(range(len(ds_clean)), n_samples)

    N_AUG = 3
    COLS  = 1 + N_AUG  # original + 3 augmented variants

    print(f"Visualizing {n_samples} samples from split='{split}'...")

    for sample_i, idx in enumerate(indices):
        fig, axes = plt.subplots(1, COLS, figsize=(5 * COLS, 5.5))
        fig.suptitle(
            f"Sample idx={idx}  |  Green = nail mask  |  "
            f"Augmentations: hflip · crop/zoom · brightness · blur · "
            f"nail vandalize (50%) · bg vandalize (40%) · noise (25%)",
            fontsize=8, color="#444"
        )

        # ── Col 0: original ──────────────────────────────────────────────────
        img_t, msk_t = ds_clean[idx]
        img_np  = denormalize(img_t)
        msk_np  = msk_t.squeeze(0).numpy()
        cov     = nail_coverage(msk_np)

        axes[0].imshow(overlay_mask(img_np, msk_np))
        axes[0].set_title(f"Original\ncoverage={cov:.1f}%", fontsize=9, weight="bold")
        axes[0].axis("off")

        # ── Col 1-3: augmented variants ──────────────────────────────────────
        aug_labels = ["Augmented #1", "Augmented #2", "Augmented #3"]
        for v in range(N_AUG):
            img_a, msk_a = ds_aug[idx]
            img_np_a = denormalize(img_a)
            msk_np_a = msk_a.squeeze(0).numpy()
            cov_a    = nail_coverage(msk_np_a)

            axes[v + 1].imshow(overlay_mask(img_np_a, msk_np_a))
            axes[v + 1].set_title(f"{aug_labels[v]}\ncoverage={cov_a:.1f}%", fontsize=9)
            axes[v + 1].axis("off")

        plt.tight_layout()
        save_path = out_dir / f"aug_{sample_i:03d}_idx{idx}.png"
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"  [{sample_i+1}/{n_samples}] Saved → {save_path.name}")

    print(f"\nDone. All outputs in: {out_dir.resolve()}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Nail VTON — Augmentation Visualizer")
    parser.add_argument("--data_root", default=DATA_ROOT,
                        help="Root of the NailSegmentationDatasetV2 folder")
    parser.add_argument("--split",     default="train",
                        help="Dataset split to sample from (train / val)")
    parser.add_argument("--n_samples", type=int, default=6,
                        help="Number of images to visualize")
    parser.add_argument("--out_dir",   default="debug_augmentation",
                        help="Folder to save output PNG files")
    args = parser.parse_args()

    visualize_augmentation(args.data_root, args.split, args.n_samples, args.out_dir)
