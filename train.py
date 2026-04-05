"""
Nail VTON Training Script
--------------------------
Usage:
    python train.py --epochs 100 --batch_size 16

Dataset: NailSegmentationDatasetV2 (CSV-based, pre-split).
Model  : NailVTONModel — binary segmentation only.
"""

import gc
import sys
import json
import psutil
import argparse
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast

sys.path.insert(0, str(Path(__file__).parent))
from dataset import make_loaders, DATA_ROOT
from model   import NailVTONModel
from losses  import BinarySegLoss, compute_miou


# ── Args ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("Nail VTON Training")
    p.add_argument("--data_root",          default=DATA_ROOT)
    p.add_argument("--epochs",             type=int,   default=100)
    p.add_argument("--batch_size",         type=int,   default=16)
    p.add_argument("--lr",                 type=float, default=1e-3)
    p.add_argument("--image_size",         type=int,   default=448)
    p.add_argument("--num_workers",        type=int,   default=2)
    p.add_argument("--ckpt_dir",           default="checkpoints")
    p.add_argument("--resume",             default=None)
    p.add_argument("--no_amp",             action="store_true")
    p.add_argument("--warmup_epochs",      type=int,   default=5)
    p.add_argument("--limit_train_batches",type=int,   default=None)
    p.add_argument("--limit_val_batches",  type=int,   default=None)
    return p.parse_args()


# ── LR schedule ────────────────────────────────────────────────────────────────

def get_lr_scale(epoch, warmup_epochs, total_epochs):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    import math
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# ── Training epoch ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, scaler, device, use_amp, limit=None):
    model.train()
    total_loss  = 0.0
    total_miou  = 0.0
    n_batches   = len(loader) if limit is None else min(len(loader), limit)
    loader_iter = iter(loader)

    for i in range(n_batches):
        # Periodic GC to prevent accumulation
        if i > 0 and i % 100 == 0:
            gc.collect(2)
            torch.cuda.empty_cache()

        try:
            batch = next(loader_iter)
        except StopIteration:
            break

        # Minimal GPU transfer
        image  = batch[0].to(device, non_blocking=True)  # (B, 3, H, W)
        target = batch[1].to(device, non_blocking=True)   # (B, 1, H, W)
        batch  = None  # Kill CPU tensor immediately

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=use_amp):
            logits = model(image)                    # (B, 2, H, W)
            loss   = criterion(logits, target)

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

        cur_loss = loss.item()
        cur_miou = compute_miou(logits.detach(), target)

        total_loss += cur_loss
        total_miou += cur_miou

        if i < 20 or (i + 1) % 50 == 0:
            mem = psutil.virtual_memory().used / (1024**3)
            print(f"  step {i+1:04d}/{n_batches} | loss={cur_loss:.4f}  miou={cur_miou:.4f} | RAM={mem:.1f}GB")
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect(2)

        # Surgical nullification
        image  = None
        target = None
        logits = None
        loss   = None

        if limit is not None and i + 1 >= limit:
            break

    return total_loss / n_batches, total_miou / n_batches


# ── Validation ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, criterion, device, use_amp, limit=None):
    model.eval()
    total_loss  = 0.0
    total_miou  = 0.0
    n_batches   = len(loader) if limit is None else min(len(loader), limit)
    loader_iter = iter(loader)

    for i in range(n_batches):
        try:
            batch = next(loader_iter)
        except StopIteration:
            break

        image  = batch[0].to(device, non_blocking=True)
        target = batch[1].to(device, non_blocking=True)
        batch  = None

        with autocast("cuda", enabled=use_amp):
            logits = model(image)
            loss   = criterion(logits, target)

        total_loss += loss.item()
        total_miou += compute_miou(logits, target)

        # Cleanup
        image  = None
        target = None
        logits = None
        loss   = None

    if n_batches == 0:
        return 0.0, 0.0
    return total_loss / n_batches, total_miou / n_batches


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    print(f"Device: {device}  |  AMP: {use_amp}")

    # ── Data ────────────────────────────────────────────────────────────────────
    train_loader, val_loader = make_loaders(
        args.data_root,
        batch_size   = args.batch_size,
        num_workers  = args.num_workers,
        image_size   = args.image_size,
    )

    # ── Model ───────────────────────────────────────────────────────────────────
    model = NailVTONModel(image_size=args.image_size, pretrained=True).to(device)
    model.count_parameters()

    # ── Loss ───────────────────────────────────────────────────────────────────
    criterion = BinarySegLoss().to(device)

    # ── Optimizer (encoder 10× lower LR) ───────────────────────────────────────
    encoder_params = list(model.encoder_low.parameters()) + list(model.encoder_high.parameters())
    encoder_ids    = {id(p) for p in encoder_params}
    decoder_params = [p for p in model.parameters() if id(p) not in encoder_ids]

    optimizer = optim.AdamW([
        {"params": encoder_params, "lr": args.lr * 0.1},
        {"params": decoder_params, "lr": args.lr},
    ], weight_decay=1e-4)

    base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def set_lr(epoch):
        scale = get_lr_scale(epoch, args.warmup_epochs, args.epochs)
        for pg, base in zip(optimizer.param_groups, base_lrs):
            pg["lr"] = base * scale

    scaler       = GradScaler("cuda", enabled=use_amp)
    start_epoch  = 0
    best_val_miou = 0.0
    history      = []

    # ── Resume ──────────────────────────────────────────────────────────────────
    if args.resume and Path(args.resume).exists():
        ckpt       = torch.load(args.resume, map_location=device)
        state_dict = ckpt["model"]
        # Strip 'module.' prefix if saved from DataParallel
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch  = ckpt["epoch"] + 1
        best_val_miou = ckpt.get("best_val_miou", 0.0)
        history      = ckpt.get("history", [])
        print(f"Resumed from epoch {start_epoch}  (best mIoU={best_val_miou:.4f})")

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ───────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        set_lr(epoch)
        lrs = [f"{pg['lr']:.2e}" for pg in optimizer.param_groups]
        t0  = time.time()

        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}  LR=[enc={lrs[0]}, dec={lrs[1]}]")

        train_loss, train_miou = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler,
            device, use_amp, limit=args.limit_train_batches
        )
        val_loss, val_miou = validate(
            model, val_loader, criterion,
            device, use_amp, limit=args.limit_val_batches
        )

        elapsed = time.time() - t0
        print(f"Epoch {epoch+1} — {elapsed:.0f}s | "
              f"train loss={train_loss:.4f}  train_miou={train_miou:.4f} | "
              f"val loss={val_loss:.4f}  val_miou={val_miou:.4f}")

        record = {
            "epoch"      : epoch,
            "train_loss" : train_loss,
            "train_miou" : train_miou,
            "val_loss"   : val_loss,
            "val_miou"   : val_miou,
        }
        history.append(record)

        ckpt_payload = {
            "epoch"        : epoch,
            "model"        : model.state_dict(),
            "optimizer"    : optimizer.state_dict(),
            "best_val_miou": best_val_miou,
            "history"      : history,
            "args"         : vars(args),
        }

        torch.save(ckpt_payload, ckpt_dir / "latest.pt")

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            ckpt_payload["best_val_miou"] = best_val_miou
            torch.save(ckpt_payload, ckpt_dir / "best.pt")
            print(f"  ✓ New best val mIoU: {best_val_miou:.4f}  — saved best.pt")

        with open(ckpt_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val mIoU: {best_val_miou:.4f}")
    print(f"Best checkpoint: {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    main()