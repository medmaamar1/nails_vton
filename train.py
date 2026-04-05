"""
Nail VTON Training Script
--------------------------
Usage:
    python train.py --data_root /path/to/dataset --epochs 100 --batch_size 8

Dataset root should contain:
    train/_annotations.coco.json  + train/images/
    valid/_annotations.coco.json  + valid/images/   (optional)
"""

import sys
import os
import json
import psutil
import gc
import argparse
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast

sys.path.insert(0, str(Path(__file__).parent))
from dataset import make_loaders
from model   import NailVTONModel
from losses  import NailVTONLoss, compute_miou


# ── Args ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("Nail VTON Training")
    p.add_argument("--data_root",   default="/kaggle/input/datasets/maamarmohamed/nail-segmentation/train")
    p.add_argument("--json_path",   default=None, help="Explicit path to annotations (optional)")
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--image_size",  type=int,   default=448)
    p.add_argument("--num_workers", type=int,   default=0)
    p.add_argument("--ckpt_dir",    default="checkpoints")
    p.add_argument("--resume",      default=None)
    p.add_argument("--no_amp",      action="store_true")
    p.add_argument("--warmup_epochs", type=int, default=5,
                   help="Linear LR warmup before cosine decay kicks in")

    p.add_argument("--orientation_path", default="/kaggle/input/datasets/maamarmohamed/oriented-nails/mp_orientations_v1.json", 
                   help="Path to mp_orientations_v1.json for strict orientation filtering")
    
    p.add_argument("--limit_train_batches", type=int, default=None,
                   help="Limit number of training batches per epoch (for smoke testing)")
    p.add_argument("--limit_val_batches", type=int, default=None,
                   help="Limit number of validation batches per epoch (for smoke testing)")
    return p.parse_args()


# ── LR schedule ────────────────────────────────────────────────────────────────

def get_lr_scale(epoch, warmup_epochs, total_epochs):
    """Linear warmup then cosine decay. Returns a scale factor in (0, 1]."""
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    import math
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# ── One epoch ──────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, scaler, device, use_amp, limit=None):
    model.train()
    total_loss, total_miou, total_dir_loss = 0.0, 0.0, 0.0
    n_batches = len(loader) if limit is None else min(len(loader), limit)

    # ── Iron-Clad Step Function (Strict Scoping) ──
    def _train_step(batch_data):
        # image, bin, dir
        img = batch_data[0].to(device, non_blocking=True)
        tgt = (None, batch_data[1].to(device, non_blocking=True), batch_data[2].to(device, non_blocking=True))
        
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=use_amp):
            outputs = model(img) # list of tuples
            l_val, l_dict = criterion(outputs, tgt)

        if use_amp:
            scaler.scale(l_val).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            l_val.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        # Metrics
        m = float(compute_miou(outputs[-1][0].detach(), tgt[1]))
        lv = float(l_dict["loss_total"])
        dv = float(l_dict.get('l2_dir', 0.0))

        # ── NUCLEAR CLEANUP ──
        # Breaking nested tuple references
        for (b, d) in outputs:
            del b, d
        del outputs, img
        # Breaking targets tuple
        _, t1, t2 = tgt
        del t1, t2, tgt
        del l_val, l_dict
        return lv, m, dv

    for i, batch in enumerate(loader):
        cur_loss, cur_miou, cur_dir = _train_step(batch)
        del batch # Kill CPU batch

        total_loss += cur_loss
        total_miou += cur_miou
        total_dir_loss += cur_dir

        if (i + 1) % 10 == 0: gc.collect(1)

        if i < 40 or (i + 1) % 50 == 0:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect(2)
            mem = psutil.virtual_memory().used / (1024**3)
            print(f"  step {i+1}/{n_batches} | loss={cur_loss:.4f} miou={cur_miou:.4f} | RAM={mem:.1f}GB")

        if limit is not None and i + 1 >= limit: break

    return total_loss/n_batches, total_miou/n_batches, total_dir_loss/n_batches


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp, limit=None):
    model.eval()
    total_loss, total_miou, total_dir_loss = 0.0, 0.0, 0.0
    n_batches = len(loader) if limit is None else min(len(loader), limit)

    def _val_step(batch_data):
        img = batch_data[0].to(device, non_blocking=True)
        tgt = (None, batch_data[1].to(device, non_blocking=True), batch_data[2].to(device, non_blocking=True))
        
        with autocast("cuda", enabled=use_amp):
            outputs = model(img)
            _, l_dict = criterion(outputs, tgt)

        m = float(compute_miou(outputs[-1][0].detach(), tgt[1]))
        lv = float(l_dict["loss_total"])
        dv = float(l_dict.get('l2_dir', 0.0))

        for (b, d) in outputs: del b, d
        del outputs, img
        _, t1, t2 = tgt
        del t1, t2, tgt
        del l_dict
        return lv, m, dv

    for i, batch in enumerate(loader):
        lv, m, dv = _val_step(batch)
        del batch
        total_loss += lv
        total_miou += m
        total_dir_loss += dv
        if limit is not None and i + 1 >= limit: break

    return total_loss/n_batches, total_miou/n_batches, total_dir_loss/n_batches


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    print(f"Device: {device}  |  AMP: {use_amp}")

    # ── Path Verification & Auto-Discovery ──────────────────────────────────────
    if not os.path.exists(args.data_root):
        print(f"\n[ERROR] DATA ROOT NOT FOUND: {args.data_root}")
        parent = os.path.dirname(args.data_root)
        if os.path.exists(parent):
            print(f"  But parent directory exists! Contents of {parent}:")
            print(f"  {os.listdir(parent)}\n")
        raise FileNotFoundError(f"Missing data_root: {args.data_root}")

    if args.orientation_path and not os.path.exists(args.orientation_path):
        print(f"\n[ERROR] ORIENTATION JSON NOT FOUND: {args.orientation_path}")
        # The user's new dataset folder might have a different JSON filename.
        parent = os.path.dirname(args.orientation_path)
        if os.path.exists(parent):
            jsons = [f for f in os.listdir(parent) if f.endswith('.json')]
            print(f"  Parent directory '{parent}' exists!")
            print(f"  Found these JSON files inside: {jsons}\n")
            if len(jsons) == 1:
                # Auto-correction
                new_path = os.path.join(parent, jsons[0])
                print(f"  -> AUTO-CORRECTING orientation_path to: {new_path}")
                args.orientation_path = new_path
            else:
                raise FileNotFoundError(f"Missing orientation_path. Found JSONs: {jsons}")
        else:
            raise FileNotFoundError(f"Missing orientation_path AND parent directory: {parent}")

    # ── Data ───────────────────────────────────────────────────────────────────
    train_loader, val_loader = make_loaders(
        args.data_root,
        batch_size       = args.batch_size,
        num_workers      = args.num_workers,
        json_path        = args.json_path,
        orientation_path = args.orientation_path
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    model = NailVTONModel(image_size=args.image_size, pretrained=True).to(device)
    
    # Enable DataParallel for Kaggle 2x GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
        model = torch.nn.DataParallel(model)
    else:
        print(f"Using {device}")
        
    if hasattr(model, 'module'):
        model.module.count_parameters()
    else:
        model.count_parameters()

    # ── Loss ───────────────────────────────────────────────────────────────────
    criterion = NailVTONLoss()

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # Encoder (pretrained) gets 10× lower LR than decoder (random init)
    base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    
    encoder_params = list(base_model.encoder_low.parameters()) + \
                     list(base_model.encoder_high.parameters())
    encoder_ids    = {id(p) for p in encoder_params}
    decoder_params = [p for p in model.parameters() if id(p) not in encoder_ids]

    optimizer = optim.AdamW([
        {"params": encoder_params, "lr": args.lr * 0.1},
        {"params": decoder_params, "lr": args.lr},
    ], weight_decay=1e-4)

    # ── Scheduler: linear warmup + cosine decay (manual per-epoch) ─────────────
    # We handle LR manually so warmup and cosine work across param groups.
    base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def set_lr(epoch):
        scale = get_lr_scale(epoch, args.warmup_epochs, args.epochs)
        for pg, base in zip(optimizer.param_groups, base_lrs):
            pg["lr"] = base * scale

    scaler = GradScaler("cuda", enabled=use_amp)

    # ── Resume ─────────────────────────────────────────────────────────────────
    start_epoch      = 0
    best_val_miou    = 0.0
    history          = []

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        
        # Guard against mismatch between multi/single GPU checkpoints
        state_dict = ckpt["model"]
        is_multi_gpu_ckpt = any(k.startswith('module.') for k in state_dict.keys())
        is_currently_multi = isinstance(model, torch.nn.DataParallel)
        
        if is_multi_gpu_ckpt and not is_currently_multi:
            # Strip 'module.' prefix
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        elif not is_multi_gpu_ckpt and is_currently_multi:
            # Add 'module.' prefix
            state_dict = {f'module.{k}': v for k, v in state_dict.items()}
            
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch      = ckpt["epoch"] + 1
        best_val_miou    = ckpt.get("best_val_miou", 0.0)
        history          = ckpt.get("history", [])
        print(f"Resumed from epoch {start_epoch}  "
              f"(best mIoU={best_val_miou:.4f})")

    # ── Checkpoint dir ─────────────────────────────────────────────────────────
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ──────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        set_lr(epoch)
        current_lrs = [f"{pg['lr']:.2e}" for pg in optimizer.param_groups]

        t0 = time.time()
        print(f"\n{'='*65}")
        print(f"Epoch {epoch+1}/{args.epochs}  "
              f"LR=[enc={current_lrs[0]}, dec={current_lrs[1]}]")

        train_loss, train_miou, train_dir_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, use_amp, limit=args.limit_train_batches
        )
        val_loss, val_miou, val_dir_loss = validate(
            model, val_loader, criterion, device, use_amp, limit=args.limit_val_batches
        )

        elapsed = time.time() - t0
        print(f"Epoch {epoch+1} — {elapsed:.0f}s | "
              f"train loss={train_loss:.4f}  "
              f"miou={train_miou:.4f}  "
              f"dir_loss={train_dir_loss:.4f} | "
              f"val loss={val_loss:.4f}  "
              f"val_miou={val_miou:.4f}  "
              f"val_dir_loss={val_dir_loss:.4f}")

        # ── Checkpointing ──────────────────────────────────────────────────────
        record = {
            "epoch"          : epoch,
            "train_loss"     : train_loss,
            "train_miou"     : train_miou,
            "train_dir_loss" : train_dir_loss,
            "val_loss"       : val_loss,
            "val_miou"       : val_miou,
            "val_dir_loss"   : val_dir_loss,
        }
        history.append(record)

        ckpt = {
            "epoch"           : epoch,
            "model"           : model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
            "optimizer"       : optimizer.state_dict(),
            "best_val_miou": best_val_miou,
            "history"         : history,
            "args"            : vars(args),
        }

        torch.save(ckpt, ckpt_dir / "latest.pt")

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            ckpt["best_val_miou"] = best_val_miou
            torch.save(ckpt, ckpt_dir / "best.pt")
            print(f"  ✓ New best val mIoU: {best_val_miou:.4f} — saved best.pt")

        with open(ckpt_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nTraining complete.")
    print(f"Best val mIoU : {best_val_miou:.4f}")
    print(f"Best checkpoint     : {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()