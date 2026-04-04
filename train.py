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
import json
import argparse
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast

sys.path.insert(0, str(Path(__file__).parent))
from dataset import make_loaders
from model   import NailVTONModel
from losses  import NailVTONLoss, compute_iou


# ── Args ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("Nail VTON Training")
    p.add_argument("--data_root",   default="/kaggle/input/datasets/almohamed132/nails-vton/train")
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

def train_one_epoch(model, loader, optimizer, criterion, scaler, device, use_amp):
    model.train()
    total_loss     = 0.0
    total_bin_iou  = 0.0
    total_dir_loss = 0.0
    n_batches      = len(loader)

    for i, batch in enumerate(loader):
        image   = batch["image"].to(device, non_blocking=True)
        targets = {
            "binary_mask"    : batch["binary_mask"].to(device,     non_blocking=True),
            "direction_field": batch["direction_field"].to(device, non_blocking=True),
        }
        del batch  # Release unused CPU tensors immediately

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=use_amp):
            preds = model(image)
            loss, loss_dict = criterion(preds, targets)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        # Pull final level for metrics
        final_preds = preds[-1]
        bin_iou  = compute_iou(final_preds[0].detach(), targets["binary_mask"])
        
        total_loss     += loss_dict["loss_total"]
        total_bin_iou  += bin_iou
        total_dir_loss += loss_dict.get('l2_dir', 0.0)

        if (i + 1) % 50 == 0:
            mem = psutil.virtual_memory().used / (1024**3)
            l_dir_now = loss_dict.get('l2_dir', 0.0)
            print(f"  step {i+1}/{n_batches} | "
                  f"loss={loss_dict['loss_total']:.4f}  "
                  f"bin_iou={bin_iou:.4f}  "
                  f"dir_loss={l_dir_now:.4f} | "
                  f"RAM={mem:.1f}GB")
            
            # Frequent small flush to prevent pile-up
            torch.cuda.empty_cache()
            import gc
            gc.collect()

        # Aggressively delete everything from the GPU/RAM
        del image, targets, preds, loss

    return (total_loss     / n_batches,
            total_bin_iou  / n_batches,
            total_dir_loss / n_batches)


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp):
    model.eval()
    total_loss     = 0.0
    total_bin_iou  = 0.0
    total_dir_loss = 0.0
    n_batches      = len(loader)

    for batch in loader:
        image   = batch["image"].to(device, non_blocking=True)
        targets = {
            "binary_mask"    : batch["binary_mask"].to(device,     non_blocking=True),
            "direction_field": batch["direction_field"].to(device, non_blocking=True),
        }
        del batch

        with autocast("cuda", enabled=use_amp):
            preds = model(image)
            _, loss_dict = criterion(preds, targets)

        final_preds = preds[-1]
        total_loss     += loss_dict["loss_total"]
        total_bin_iou  += compute_iou(final_preds[0], targets["binary_mask"])
        total_dir_loss += loss_dict.get('l2_dir', 0.0)

        del image, targets, preds

    return (total_loss     / n_batches,
            total_bin_iou  / n_batches,
            total_dir_loss / n_batches)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    print(f"Device: {device}  |  AMP: {use_amp}")

    # ── Data ───────────────────────────────────────────────────────────────────
    train_loader, val_loader = make_loaders(
        args.data_root,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
    )

    # ── Model ──────────────────────────────────────────────────────────────────
    model = NailVTONModel(image_size=args.image_size, pretrained=True).to(device)
    
    # Enable DataParallel for Kaggle 2x GPUs
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)
        
    model.module.count_parameters() if isinstance(model, torch.nn.DataParallel) else model.count_parameters()

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
    best_val_bin_iou = 0.0
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
        best_val_bin_iou = ckpt.get("best_val_bin_iou", 0.0)
        history          = ckpt.get("history", [])
        print(f"Resumed from epoch {start_epoch}  "
              f"(best binary IoU={best_val_bin_iou:.4f})")

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

        train_loss, train_bin_iou, train_dir_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, use_amp
        )
        val_loss, val_bin_iou, val_dir_loss = validate(
            model, val_loader, criterion, device, use_amp
        )

        elapsed = time.time() - t0
        print(f"Epoch {epoch+1} — {elapsed:.0f}s | "
              f"train loss={train_loss:.4f}  "
              f"bin_iou={train_bin_iou:.4f}  "
              f"dir_loss={train_dir_loss:.4f} | "
              f"val loss={val_loss:.4f}  "
              f"val_bin_iou={val_bin_iou:.4f}  "
              f"val_dir_loss={val_dir_loss:.4f}")

        # ── Checkpointing ──────────────────────────────────────────────────────
        record = {
            "epoch"          : epoch,
            "train_loss"     : train_loss,
            "train_bin_iou"  : train_bin_iou,
            "train_dir_loss" : train_dir_loss,
            "val_loss"       : val_loss,
            "val_bin_iou"    : val_bin_iou,
            "val_dir_loss"   : val_dir_loss,
        }
        history.append(record)

        ckpt = {
            "epoch"           : epoch,
            "model"           : model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
            "optimizer"       : optimizer.state_dict(),
            "best_val_bin_iou": best_val_bin_iou,
            "history"         : history,
            "args"            : vars(args),
        }

        torch.save(ckpt, ckpt_dir / "latest.pt")

        if val_bin_iou > best_val_bin_iou:
            best_val_bin_iou = val_bin_iou
            ckpt["best_val_bin_iou"] = best_val_bin_iou
            torch.save(ckpt, ckpt_dir / "best.pt")
            print(f"  ✓ New best val binary IoU: {best_val_bin_iou:.4f} — saved best.pt")

        with open(ckpt_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nTraining complete.")
    print(f"Best val binary IoU : {best_val_bin_iou:.4f}")
    print(f"Best checkpoint     : {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()