"""
Nail VTON Training Script
--------------------------
Usage:
    python train.py --epochs 200 --batch_size 16

Dataset: NailSegmentationDatasetV2 (CSV-based, pre-split).
Model  : NailVTONModel — binary segmentation only.
"""

import gc
import os
import sys
import json
import psutil
import argparse
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).parent))
from dataset import make_loaders, DATA_ROOT
from model   import NailVTONModel
from losses  import BinarySegLoss, compute_miou


# ── Args ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("Nail VTON Training")
    p.add_argument("--data_root",          default=DATA_ROOT)
    p.add_argument("--epochs",             type=int,   default=200)
    p.add_argument("--patience",           type=int,   default=20)
    p.add_argument("--batch_size",         type=int,   default=16)
    p.add_argument("--lr",                 type=float, default=5e-4) # Fine-Tuning LR
    p.add_argument("--image_size",         type=int,   default=448)
    p.add_argument("--num_workers",        type=int,   default=2)
    p.add_argument("--ckpt_dir",           default="checkpoints")
    p.add_argument("--resume",             default=None)
    p.add_argument("--no_amp",             action="store_true")
    p.add_argument("--warmup_epochs",      type=int,   default=10) # Longer warmup for stability
    p.add_argument("--limit_train_batches",type=int,   default=None)
    p.add_argument("--limit_val_batches",  type=int,   default=None)
    p.add_argument("--hard_neg_prob",      type=float, default=0.2)
    p.add_argument("--history_json",       default=None)
    return p.parse_args()


# ── LR schedule ────────────────────────────────────────────────────────────────

def get_lr_scale(epoch, warmup_epochs, total_epochs):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    import math
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    else:
        local_rank = 0
    return distributed, local_rank, world_size


def reduce_scalar(value, device, distributed):
    t = torch.tensor(float(value), device=device)
    if distributed:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= dist.get_world_size()
    return t.item()


# ── Training epoch ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, scaler, device, use_amp,
                    limit=None, is_main_process=True):
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
            img_cpu, tgt_cpu = next(loader_iter)
        except StopIteration:
            break

        image  = img_cpu.to(device, non_blocking=True)
        target = tgt_cpu.to(device, non_blocking=True)
        del img_cpu, tgt_cpu

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=use_amp, cache_enabled=False):
            out = model(image)  # Tuple: (logits, logits_inter, gate)
            loss_val = criterion(out, target)

        if use_amp:
            scaler.scale(loss_val).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        cur_loss = loss_val.item()
        cur_miou = compute_miou(out[0].detach(), target)

        total_loss += cur_loss
        total_miou += cur_miou

        if is_main_process and (i < 20 or (i + 1) % 50 == 0):
            mem = psutil.virtual_memory().used / (1024**3)
            print(f"  step {i+1:04d}/{n_batches} | loss={cur_loss:.4f}  miou={cur_miou:.4f} | RAM={mem:.1f}GB")
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if hasattr(torch, "clear_autocast_cache"):
                torch.clear_autocast_cache()
            gc.collect()

        # Aggressively break references
        del image, target, out, loss_val

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
            img_cpu, tgt_cpu = next(loader_iter)
        except StopIteration:
            break

        image  = img_cpu.to(device, non_blocking=True)
        target = tgt_cpu.to(device, non_blocking=True)
        del img_cpu, tgt_cpu

        with autocast("cuda", enabled=use_amp, cache_enabled=False):
            out = model(image)
            loss_val = criterion(out, target)

        cur_loss = loss_val.item()
        cur_miou = compute_miou(out[0].detach(), target)
        
        total_loss += cur_loss
        total_miou += cur_miou

        # Aggressively break references
        del image, target, out, loss_val

    if n_batches == 0:
        return 0.0, 0.0
    return total_loss / n_batches, total_miou / n_batches


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    distributed, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main_process = (not distributed) or local_rank == 0

    amp_supported = False
    if device.type == "cuda":
        major, _ = torch.cuda.get_device_capability(device)
        amp_supported = major >= 7
    use_amp = (not args.no_amp) and device.type == "cuda" and amp_supported

    if is_main_process:
        print(f"Device: {device}  |  AMP: {use_amp}  |  DDP: {distributed}  |  WORLD_SIZE: {world_size}")
        if distributed:
            print("Using DistributedDataParallel for multi-GPU to avoid DataParallel RAM growth.")

    local_batch_size = args.batch_size
    if distributed:
        local_batch_size = max(1, args.batch_size // world_size)
        if is_main_process and args.batch_size % world_size != 0:
            print(f"[DDP] batch_size={args.batch_size} not divisible by world size {world_size}; using per-GPU batch_size={local_batch_size}.")

    # ── Data ────────────────────────────────────────────────────────────────────
    train_loader, val_loader = make_loaders(
        args.data_root,
        batch_size   = local_batch_size,
        num_workers  = args.num_workers,
        image_size   = args.image_size,
        distributed  = distributed,
        rank         = local_rank,
        world_size   = world_size,
        hard_negative_prob = args.hard_neg_prob,
    )

    # ── Model ───────────────────────────────────────────────────────────────────
    model = NailVTONModel(image_size=args.image_size, pretrained=True).to(device)
    if is_main_process:
        model.count_parameters()
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # ── Loss ───────────────────────────────────────────────────────────────────
    criterion = BinarySegLoss().to(device)

    # ── Optimizer (encoder 10× lower LR) ───────────────────────────────────────
    base_model = model.module if hasattr(model, "module") else model
    encoder_params = list(base_model.encoder_low.parameters()) + list(base_model.encoder_high.parameters())
    encoder_ids    = {id(p) for p in encoder_params}
    decoder_params = [p for p in base_model.parameters() if id(p) not in encoder_ids]

    optimizer = optim.AdamW([
        {"params": encoder_params, "lr": args.lr * 0.1},
        {"params": decoder_params, "lr": args.lr},
    ], weight_decay=1e-4)

    base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def set_lr(epoch):
        scale = get_lr_scale(epoch, args.warmup_epochs, args.epochs)
        for pg, base in zip(optimizer.param_groups, base_lrs):
            pg["lr"] = base * scale

    scaler        = GradScaler("cuda", enabled=use_amp)
    start_epoch   = 0
    best_val_miou = 0.0
    epochs_no_improve = 0
    history       = []

    # ── Resume ──────────────────────────────────────────────────────────────────
    if args.resume and Path(args.resume).exists():
        ckpt       = torch.load(args.resume, map_location=device)
        state_dict = ckpt["model"]
        # Strip 'module.' prefix if saved from DataParallel
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        base_model.load_state_dict(state_dict)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch       = ckpt["epoch"] + 1
        best_val_miou     = ckpt.get("best_val_miou", 0.0)
        epochs_no_improve = ckpt.get("epochs_no_improve", 0)
        history           = ckpt.get("history", [])
        if is_main_process:
            print(f"Resumed from epoch {start_epoch}  (best mIoU={best_val_miou:.4f}, no-improve streak={epochs_no_improve})")

    if args.history_json and Path(args.history_json).exists():
        with open(args.history_json, "r") as f:
            history = json.load(f)
        if is_main_process:
            print(f"Loaded training history from {args.history_json} ({len(history)} epochs)")

    ckpt_dir = Path(args.ckpt_dir)
    if is_main_process:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ───────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        if distributed and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        set_lr(epoch)
        lrs = [f"{pg['lr']:.2e}" for pg in optimizer.param_groups]
        t0  = time.time()

        if is_main_process:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{args.epochs}  LR=[enc={lrs[0]}, dec={lrs[1]}]")

        train_loss, train_miou = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler,
            device, use_amp, limit=args.limit_train_batches,
            is_main_process=is_main_process
        )
        val_loss, val_miou = validate(
            model, val_loader, criterion,
            device, use_amp, limit=args.limit_val_batches
        )

        train_loss = reduce_scalar(train_loss, device, distributed)
        train_miou = reduce_scalar(train_miou, device, distributed)
        val_loss   = reduce_scalar(val_loss, device, distributed)
        val_miou   = reduce_scalar(val_miou, device, distributed)

        elapsed = time.time() - t0
        if is_main_process:
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

        ckpt_payload = None
        if is_main_process:
            ckpt_payload = {
                "epoch"            : epoch,
                "model"            : base_model.state_dict(),
                "optimizer"        : optimizer.state_dict(),
                "best_val_miou"    : best_val_miou,
                "epochs_no_improve": epochs_no_improve,
                "history"          : history,
                "args"             : vars(args),
            }

            torch.save(ckpt_payload, ckpt_dir / "latest.pt")

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            epochs_no_improve = 0
            if is_main_process:
                ckpt_payload["best_val_miou"] = best_val_miou
                torch.save(ckpt_payload, ckpt_dir / "best.pt")
                print(f"  ✓ New best val mIoU: {best_val_miou:.4f}  — saved best.pt")
        else:
            epochs_no_improve += 1
            if is_main_process:
                print(f"  No improvement for {epochs_no_improve}/{args.patience} epochs")
            if epochs_no_improve >= args.patience:
                if is_main_process:
                    print(f"\nEarly stopping — no improvement for {args.patience} consecutive epochs.")
                break

        if is_main_process:
            with open(ckpt_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)

    if is_main_process:
        print(f"\nTraining complete. Best val mIoU: {best_val_miou:.4f}")
        print(f"Best checkpoint: {ckpt_dir / 'best.pt'}")

    if distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()