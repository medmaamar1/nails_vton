import torch
import numpy as np
from pathlib import Path
from dataset import make_loaders
from model   import NailVTONModel
from losses  import NailVTONLoss, compute_iou, compute_instance_iou

def run_sanity_check(data_root):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Nail VTON Sanity Check ---")
    print(f"Device: {device}")

    # 1. Test Data Loading (11 Channels)
    print("\n[1/4] Testing Data Loader...")
    train_loader, _ = make_loaders(data_root, batch_size=4, num_workers=0)
    batch = next(iter(train_loader))
    
    images   = batch["image"].to(device)
    binary_t = batch["binary_mask"].to(device)
    inst_t   = batch["instance_masks"].to(device)
    dir_t    = batch["direction_field"].to(device)
    
    print(f"     Image shape    : {images.shape}")         # (B, 3, 512, 512)
    print(f"     Binary mask    : {binary_t.shape}")       # (B, 1, 512, 512)
    print(f"     Instance masks : {inst_t.shape}")         # (B, 11, 512, 512)
    print(f"     Direction field: {dir_t.shape}")         # (B, 2, 512, 512)
    
    # Check if background is at Channel 0
    bg_sum = inst_t[:, 0].sum().item()
    nail_sum = inst_t[:, 1:].sum().item()
    print(f"     Background Pixels: {bg_sum:.0f}")
    print(f"     Nail Pixels      : {nail_sum:.0f}")
    
    if bg_sum == 0:
        print("     ❌ ERROR: Background channel is empty!")
    else:
        print("     ✅ Background (Ch 0) vs Nails (Ch 1-10) separation verified.")

    # 2. Test Model Output
    print("\n[2/4] Testing Model Architecture...")
    model = NailVTONModel(image_size=512, pretrained=False).to(device)
    preds = model(images)
    
    print(f"     Binary Logits: {preds[0].shape}")
    print(f"     Inst Logits  : {preds[1].shape}") # Should be 11
    print(f"     Dir Vectors  : {preds[2].shape}")
    
    if preds[1].shape[1] == 11:
        print("     ✅ Model outputs 11 channels as expected.")
    else:
        print(f"     ❌ ERROR: Model output {preds[1].shape[1]} channels, expected 11.")

    # 3. Test Loss Function & Direction Gradient
    print("\n[3/4] Testing Loss Functions...")
    criterion = NailVTONLoss()
    loss, loss_dict = criterion(preds, {
        "binary_mask": binary_t,
        "instance_masks": inst_t,
        "direction_field": dir_t
    })
    
    print(f"     Total Loss     : {loss.item():.4f}")
    print(f"     Binary Loss    : {loss_dict['loss_binary']:.4f}")
    print(f"     Instance Loss  : {loss_dict['loss_instance']:.4f}")
    print(f"     Direction Loss : {loss_dict['loss_direction']:.4f}")
    
    if loss_dict['loss_direction'] > 0:
        print("     ✅ Direction Loss is ACTIVE (non-zero target vectors detected).")
    else:
        print("     ⚠️  WARNING: Direction Loss is 0. Check your annotations or target field.")

    # 4. Test Metric Masking
    print("\n[4/4] Testing IoU Masking...")
    inst_iou = compute_instance_iou(preds[1], inst_t)
    print(f"     Instance IoU (Excluding Background): {inst_iou:.4f}")
    
    print("\n--- SANITY CHECK COMPLETE ---")
    if bg_sum > 0 and preds[1].shape[1] == 11:
        print("Result: READY TO TRAIN 🚀")
    else:
        print("Result: FIX ERRORS BEFORE STARTING 🛑")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", help="Path to your COCO dataset (e.g. /kaggle/input/nails-dataset)")
    args = parser.parse_args()
    
    if not args.data_root or not Path(args.data_root).exists():
        print(f"\n🛑 ERROR: Data root '{args.data_root}' not found.")
        print("Please provide the correct path using --data_root")
        print("Example: python sanity_check.py --data_root /kaggle/input/datasets/maamarmohamed12/nails-vton/train")
    else:
        run_sanity_check(args.data_root)
