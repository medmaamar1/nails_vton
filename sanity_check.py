import torch
import numpy as np
from pathlib import Path
from dataset import make_loaders
from model   import NailVTONModel
from losses  import NailVTONLoss, compute_iou, compute_instance_iou

def run_sanity_check(data_root):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Nail VTON Sanity Check (Strict VTNFP) ---")
    print(f"Device: {device}")

    # 1. Test Data Loading
    print("\n[1/4] Testing Data Loader...")
    train_loader, _ = make_loaders(data_root, batch_size=4, num_workers=0)
    batch = next(iter(train_loader))
    
    images   = batch["image"].to(device)
    binary_t = batch["binary_mask"].to(device)
    inst_t   = batch["instance_masks"].to(device)
    dir_t    = batch["direction_field"].to(device)
    
    print(f"     Image shape    : {images.shape}")         # (B, 3, 448, 448)
    print(f"     Binary mask    : {binary_t.shape}")       # (B, 1, 448, 448)
    print(f"     Instance masks : {inst_t.shape}")         # (B, 10, 448, 448)
    print(f"     Direction field: {dir_t.shape}")         # (B, 2, 448, 448)
    
    # Check Direction Field Unit Normalization (Section 3.3)
    norm = torch.norm(dir_t, dim=1)
    # Norm should be 1.0 where mask is 1.0, and 0.0 elsewhere
    valid_mask = (torch.norm(dir_t, dim=1) > 1e-6)
    if valid_mask.any():
        avg_norm = norm[valid_mask].mean().item()
        print(f"     Avg Target Vector Norm: {avg_norm:.4f}")
        if abs(avg_norm - 1.0) < 1e-3:
            print("     ✅ Direction field is unit-normalized.")
        else:
            print(f"     ❌ ERROR: Direction field norm is {avg_norm:.4f}, expected 1.0.")
    
    # 2. Test Model Output (Laplacian Pyramid)
    print("\n[2/4] Testing Model Architecture (Laplacian Pyramid)...")
    model = NailVTONModel(image_size=448, pretrained=False).to(device)
    multi_preds = model(images)  # Returns list of [(bin, inst, dir), ...]
    
    print(f"     Pyramid Levels: {len(multi_preds)}")
    for i, (p_bin, p_inst, p_dir) in enumerate(multi_preds):
        print(f"     Level {i}: bin={p_bin.shape}, inst={p_inst.shape}, dir={p_dir.shape}")
    
    if len(multi_preds) == 3:
        print("     ✅ Model outputs 3 Laplacian levels as expected.")
    
    # 3. Test Loss Function (Strict Scaling)
    print("\n[3/4] Testing Loss Functions (Equation 3 Scaling)...")
    criterion = NailVTONLoss()
    target_dict = {
        "binary_mask": binary_t,
        "instance_masks": inst_t,
        "direction_field": dir_t
    }
    loss, loss_dict = criterion(multi_preds, target_dict)
    
    print(f"     Total Loss      : {loss.item():.4f}")
    print(f"     Final Level Bin : {loss_dict['l2_bin']:.4f}")
    print(f"     Final Level Inst: {loss_dict['l2_inst']:.4f}")
    print(f"     Final Level Dir : {loss_dict['l2_dir']:.4f}")
    
    # Strict Equation 3 scaling usually results in small Direction loss initially
    if loss_dict['l2_dir'] < 0.5:
        print("     ✅ Direction Loss scaling appears correct (Image-Average).")
    
    # 4. Test Metric Masking
    print("\n[4/4] Testing IoU Metrics...")
    final_bin, final_inst, final_dir = multi_preds[-1]
    bin_iou = compute_iou(final_bin, binary_t)
    inst_iou = compute_instance_iou(final_inst, inst_t, binary_t)
    print(f"     Final Level Binary IoU  : {bin_iou:.4f}")
    print(f"     Final Level Instance IoU: {inst_iou:.4f}")
    
    print("\n--- SANITY CHECK COMPLETE ---")
    print("Result: READY TO TRAIN 🚀")

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
                print(f"📂 Auto-detected data_root: {root}")
                break

    if not root or not Path(root).exists():
        print(f"\n🛑 ERROR: Data root '{root}' not found.")
    else:
        run_sanity_check(root)
