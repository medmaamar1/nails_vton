import torch
import numpy as np
from pathlib import Path
from dataset import make_loaders
from model   import NailVTONModel
from losses  import NailVTONLoss, compute_iou

def run_sanity_check(data_root):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Nail VTON Sanity Check (Stripped Architecture) ---")
    print(f"Device: {device}")

    # 1. Test Data Loading
    print("\n[1/4] Testing Data Loader...")
    try:
        train_loader, _ = make_loaders(data_root, batch_size=4, num_workers=0)
        batch = next(iter(train_loader))
    except Exception as e:
        print(f"     [ERROR] Data loading failed: {e}")
        return

    images   = batch["image"].to(device)
    binary_t = batch["binary_mask"].to(device)
    dir_t    = batch["direction_field"].to(device)
    
    print(f"     Image shape    : {images.shape}")         # (B, 3, 448, 448)
    print(f"     Binary mask    : {binary_t.shape}")       # (B, 1, 448, 448)
    print(f"     Direction field: {dir_t.shape}")         # (B, 2, 448, 448)
    
    # Check Direction Field Unit Normalization
    norm = torch.norm(dir_t, dim=1)
    valid_mask = (norm > 1e-6)
    if valid_mask.any():
        avg_norm = norm[valid_mask].mean().item()
        print(f"     Avg Target Vector Norm: {avg_norm:.4f}")
        if abs(avg_norm - 1.0) < 1e-2:
            print("     [OK] Direction field is unit-normalized.")
        else:
            print(f"     [WARNING] Direction field norm is {avg_norm:.4f}, expected ~1.0.")
    
    # 2. Test Model Output (Laplacian Pyramid)
    print("\n[2/4] Testing Model Architecture (Laplacian Pyramid)...")
    model = NailVTONModel(image_size=448, pretrained=False).to(device)
    multi_preds = model(images)  
    
    print(f"     Pyramid Levels: {len(multi_preds)}")
    for i, (p_bin, p_dir) in enumerate(multi_preds):
        print(f"     Level {i}: bin={p_bin.shape}, dir={p_dir.shape}")
    
    if len(multi_preds) == 3:
        print("     [OK] Model outputs 3 Laplacian levels as expected.")
    
    # 3. Test Loss Function
    print("\n[3/4] Testing Loss Functions...")
    criterion = NailVTONLoss()
    target_dict = {
        "binary_mask": binary_t,
        "direction_field": dir_t
    }
    loss, loss_dict = criterion(multi_preds, target_dict)
    
    print(f"     Total Loss      : {loss.item():.4f}")
    
    # 4. Test Metric Masking
    print("\n[4/4] Testing IoU Metrics...")
    final_bin, final_dir = multi_preds[-1]
    bin_iou = compute_iou(final_bin, binary_t)
    print(f"     Final Level Binary IoU  : {bin_iou:.4f}")
    
    print("\n--- SANITY CHECK COMPLETE ---")
    print("Result: READY TO TRAIN")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", help="Path to your COCO dataset")
    args = parser.parse_args()
    
    root = args.data_root
    if not root:
        common_paths = [
            "/kaggle/input/datasets/almohamed132/nails-vton/train",
            "c:/Users/OrdiOne/Desktop/douccana marketplace - Copy/nails_vton/train"
        ]
        for p in common_paths:
            if Path(p).exists():
                root = p
                print(f"Auto-detected data_root: {root}")
                break

    if not root or not Path(root).exists():
        print(f"\n[ERROR] Data root '{root}' not found. Please provide --data_root")
    else:
        run_sanity_check(root)
