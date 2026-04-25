"""
Nail VTON Sanity Check
-----------------------
Validates the full pipeline (data → model → loss → metric) before
launching a full Kaggle training run.

Usage:
    python sanity_check.py --data_root /kaggle/input/.../NailSegmentationDatasetV2
"""

import sys
import argparse
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dataset import make_loaders
from model   import build_model, count_parameters
from losses  import BinarySegLoss, compute_miou


def run_sanity_check(data_root: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Nail VTON Sanity Check (DeepLabV3+ ResNet-101) ---")
    print(f"Device   : {device}")
    print(f"Data root: {data_root}")

    # ── 1. Data Loader ────────────────────────────────────────────────────────
    print("\n[1/4] Testing Data Loader...")
    train_loader, val_loader = make_loaders(data_root, batch_size=2, num_workers=0)

    img_cpu, msk_cpu = next(iter(train_loader))
    images = img_cpu.to(device)
    masks  = msk_cpu.to(device)

    print(f"     Image shape : {images.shape}")   # (B, 3, 640, 640)
    print(f"     Mask  shape : {masks.shape}")    # (B, 1, 640, 640)
    print(f"     Mask  unique: {masks.unique().tolist()}")

    assert images.shape[1] == 3,         "Expected 3-channel image"
    assert masks.shape[1]  == 1,         "Expected 1-channel mask"
    assert masks.max()     <= 1.0 + 1e-5, "Mask values must be in [0, 1]"
    print("     ✅ Data loader OK")

    # ── 2. Model Forward ──────────────────────────────────────────────────────
    print("\n[2/4] Testing Model Architecture (DeepLabV3+)...")
    model = build_model(image_size=images.shape[-1]).to(device)
    count_parameters(model)

    model.eval()
    with torch.no_grad():
        out = model(images)

    print(f"     Output shape: {out.shape}")      # Expected: (B, 1, H, W)
    assert out.shape == images.shape[:1] + torch.Size([1]) + images.shape[2:], \
        f"Output shape mismatch: {out.shape}"
    print("     ✅ Model output shape OK")

    # ── 3. Loss ───────────────────────────────────────────────────────────────
    print("\n[3/4] Testing Loss Functions...")
    model.train()
    criterion = BinarySegLoss().to(device)

    out      = model(images)
    loss_val = criterion(out, masks)

    print(f"     Loss value : {loss_val.item():.4f}")
    assert not torch.isnan(loss_val), "Loss is NaN — check your data or model!"
    assert not torch.isinf(loss_val), "Loss is Inf — check your data or model!"

    loss_val.backward()
    print("     ✅ Backward pass OK (no NaN/Inf)")

    # ── 4. Metric ─────────────────────────────────────────────────────────────
    print("\n[4/4] Testing mIoU Metric...")
    model.eval()
    with torch.no_grad():
        out  = model(images)
        miou = compute_miou(out, masks)

    print(f"     mIoU (untrained) : {miou:.4f}")
    # Untrained binary model should be near 0.5 (random chance baseline)
    print("     ✅ compute_miou returned a valid scalar")

    print("\n--- SANITY CHECK COMPLETE ---")
    print("Result: READY TO TRAIN 🚀")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Nail VTON Sanity Check")
    parser.add_argument("--data_root", default=None, help="Path to NailSegmentationDatasetV2")
    args = parser.parse_args()

    root = args.data_root
    if not root:
        common_paths = [
            "/kaggle/input/datasets/muhammadhammad261/nail-segmentation-dataset/NailSegmentationDatasetV2",
            "c:/Users/OrdiOne/Desktop/douccana marketplace - Copy/nails_vton/train",
        ]
        for p in common_paths:
            if Path(p).exists():
                root = p
                print(f"📂 Auto-detected data_root: {root}")
                break

    if not root or not Path(root).exists():
        print(f"\n🛑 ERROR: Data root '{root}' not found.")
        sys.exit(1)

    run_sanity_check(root)
