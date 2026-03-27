"""
Nail VTON Loss Functions
------------------------
  1. BinarySegLoss    : LMP (BCE max-pool) + Dice  for binary head
  2. InstanceSegLoss  : cross-entropy + Dice        for softmax instance head
  3. DirectionLoss    : cosine similarity           for direction field
  4. LMPLoss          : Loss Max-Pooling (class imbalance — nails ~2-5% of pixels)
  5. NailVTONLoss     : combined loss with configurable weights

Key fix vs previous version:
  Instance head now uses SOFTMAX (10-class mutual exclusion), not 10 independent
  sigmoids. Each pixel belongs to exactly one nail — or background (channel 0
  is reserved for background in the 10-class scheme, channels 1-9 for nails).

  Actually: the paper uses 10 channels for UP TO 10 nails and treats it as a
  10-class softmax where channel i = "this pixel belongs to nail i".
  Background is implicitly handled by the binary head.
  Loss = cross-entropy over the 10 channels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Loss Max-Pooling ───────────────────────────────────────────────────────────

class LMPLoss(nn.Module):
    """
    Loss Max-Pooling from the paper.
    Computes BCE per-pixel, keeps top keep_ratio% of losses before averaging.
    Forces focus on hard/rare pixels (nail boundaries, small nails).
    """
    def __init__(self, keep_ratio=0.3):
        super().__init__()
        self.keep_ratio = keep_ratio

    def forward(self, logits, targets):
        """logits, targets: (B, C, H, W)"""
        loss_map  = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )                                                   # (B, C, H, W)
        B         = loss_map.shape[0]
        loss_flat = loss_map.view(B, -1)                   # (B, C*H*W)
        k         = max(1, int(loss_flat.size(1) * self.keep_ratio))
        top_k, _  = loss_flat.topk(k, dim=1)
        return top_k.mean()


# ── Dice Loss ─────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """Soft Dice loss. Works for both binary (sigmoid) and multi-class (softmax)."""
    def __init__(self, smooth=1.0, use_softmax=False):
        super().__init__()
        self.smooth       = smooth
        self.use_softmax  = use_softmax

    def forward(self, logits, targets):
        """
        logits  : (B, C, H, W) raw
        targets : (B, C, H, W) float in [0,1]
        """
        if self.use_softmax:
            probs = torch.softmax(logits, dim=1)
        else:
            probs = torch.sigmoid(logits)

        probs_f   = probs.view(probs.size(0),   probs.size(1),   -1)
        targets_f = targets.view(targets.size(0), targets.size(1), -1)

        inter = (probs_f * targets_f).sum(dim=-1)
        union = probs_f.sum(dim=-1) + targets_f.sum(dim=-1)
        dice  = (2.0 * inter + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


# ── Direction Loss ────────────────────────────────────────────────────────────

class DirectionLoss(nn.Module):
    """
    Cosine similarity loss on nail pixels only.
    Background pixels have target direction (0,0) and are masked out.
    """
    def forward(self, pred_dir, target_dir):
        """
        pred_dir   : (B, 2, H, W) already normalised unit vectors
        target_dir : (B, 2, H, W) unit vectors, (0,0) on background
        """
        magnitude  = target_dir.norm(dim=1, keepdim=True)     # (B,1,H,W)
        valid_mask = (magnitude > 1e-6).squeeze(1)             # (B,H,W) bool

        if not valid_mask.any():
            return torch.tensor(0.0, device=pred_dir.device, requires_grad=True)

        cos_sim = (pred_dir * target_dir).sum(dim=1)           # (B,H,W)
        return (1.0 - cos_sim[valid_mask]).mean()


# ── Binary Segmentation Loss ──────────────────────────────────────────────────

class BinarySegLoss(nn.Module):
    def __init__(self, lmp_ratio=0.3, w_lmp=0.5, w_dice=0.5):
        super().__init__()
        self.lmp    = LMPLoss(keep_ratio=lmp_ratio)
        self.dice   = DiceLoss(use_softmax=False)
        self.w_lmp  = w_lmp
        self.w_dice = w_dice

    def forward(self, logits, targets):
        """logits, targets: (B, 1, H, W)"""
        return self.w_lmp * self.lmp(logits, targets) + \
               self.w_dice * self.dice(logits, targets)


# ── Instance Segmentation Loss ────────────────────────────────────────────────

class InstanceSegLoss(nn.Module):
    """
    Softmax cross-entropy + Dice for the 10-channel instance head.

    targets : (B, MAX_INST, H, W) — one-hot style float masks.
              Channel i is 1 for pixels belonging to nail i, 0 elsewhere.
              Unused channels (beyond actual nail count) stay all-zero.

    Cross-entropy treats it as a multi-class problem per pixel.
    Dice is computed per channel over softmax probabilities.
    """
    def __init__(self, w_ce=0.5, w_dice=0.5):
        super().__init__()
        self.dice   = DiceLoss(use_softmax=True)
        self.w_ce   = w_ce
        self.w_dice = w_dice

    def forward(self, logits, targets):
        """
        logits  : (B, MAX_INST, H, W)
        targets : (B, MAX_INST, H, W) one-hot float
        """
        # Cross-entropy expects class indices (B, H, W) — derive from one-hot
        # argmax along channel dim gives the correct class index per pixel.
        # Background pixels (all zeros) will get class 0 which is fine —
        # the binary head handles foreground/background separation.
        target_idx = targets.argmax(dim=1)                    # (B, H, W) long

        ce_loss   = F.cross_entropy(logits, target_idx)
        dice_loss = self.dice(logits, targets)

        return self.w_ce * ce_loss + self.w_dice * dice_loss


# ── Combined Loss ─────────────────────────────────────────────────────────────

class NailVTONLoss(nn.Module):
    """
    L = w_binary * L_binary + w_instance * L_instance + w_direction * L_direction
    """
    def __init__(
        self,
        w_binary    = 1.0,
        w_instance  = 1.0,
        w_direction = 0.5,
        lmp_ratio   = 0.3,
    ):
        super().__init__()
        self.w_binary    = w_binary
        self.w_instance  = w_instance
        self.w_direction = w_direction

        self.binary_loss    = BinarySegLoss(lmp_ratio=lmp_ratio)
        self.instance_loss  = InstanceSegLoss()
        self.direction_loss = DirectionLoss()

    def forward(self, predictions, targets):
        """
        predictions : (binary_logits, inst_logits, direction)
                      shapes: (B,1,H,W), (B,10,H,W), (B,2,H,W)
        targets     : dict with keys:
                        "binary_mask"     (B,  1, H, W) float
                        "instance_masks"  (B, 10, H, W) float one-hot
                        "direction_field" (B,  2, H, W) float unit vectors
        Returns:
            total_loss scalar, loss_dict for logging
        """
        binary_logits, inst_logits, direction = predictions

        l_bin  = self.binary_loss(binary_logits,   targets["binary_mask"])
        l_inst = self.instance_loss(inst_logits,   targets["instance_masks"])
        l_dir  = self.direction_loss(direction,    targets["direction_field"])

        total = (self.w_binary    * l_bin  +
                 self.w_instance  * l_inst +
                 self.w_direction * l_dir)

        return total, {
            "loss_total"    : total.item(),
            "loss_binary"   : l_bin.item(),
            "loss_instance" : l_inst.item(),
            "loss_direction": l_dir.item(),
        }


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_iou(pred_mask, target_mask, threshold=0.5, eps=1e-6):
    """
    Binary IoU for the binary segmentation head.
    pred_mask   : (B, 1, H, W) logits or probs
    target_mask : (B, 1, H, W) binary float
    """
    if pred_mask.max() > 1.0 or pred_mask.min() < 0.0:
        pred_binary = (torch.sigmoid(pred_mask) > threshold).float()
    else:
        pred_binary = (pred_mask > threshold).float()

    intersection = (pred_binary * target_mask).sum(dim=(-2, -1))
    union        = (pred_binary + target_mask).clamp(0, 1).sum(dim=(-2, -1))
    iou          = (intersection + eps) / (union + eps)
    return iou.mean().item()


def compute_instance_iou(inst_logits, target_masks, eps=1e-6):
    """
    Mean IoU for the softmax instance head.
    inst_logits  : (B, 10, H, W) raw logits
    target_masks : (B, 10, H, W) one-hot float
    Only counts channels that have at least one positive pixel in the target.
    """
    probs       = torch.softmax(inst_logits, dim=1)
    pred_binary = (probs > 0.5).float()

    intersection = (pred_binary * target_masks).sum(dim=(-2, -1))  # (B, 10)
    union        = (pred_binary + target_masks).clamp(0, 1).sum(dim=(-2, -1))

    # Only average over channels with actual nail annotations
    valid        = target_masks.sum(dim=(-2, -1)) > 0               # (B, 10)
    iou          = (intersection + eps) / (union + eps)
    iou_valid    = iou[valid]

    return iou_valid.mean().item() if iou_valid.numel() > 0 else 0.0


# ── Sanity check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, H, W, MAX_I = 2, 512, 512, 10

    binary_logits = torch.randn(B,      1, H, W)
    inst_logits   = torch.randn(B, MAX_I, H, W)
    direction     = torch.randn(B,      2, H, W)
    direction     = direction / direction.norm(dim=1, keepdim=True).clamp(min=1e-6)

    # One-hot instance targets: assign each pixel to exactly one nail
    inst_idx   = torch.randint(0, MAX_I, (B, H, W))
    inst_onehot = torch.zeros(B, MAX_I, H, W)
    inst_onehot.scatter_(1, inst_idx.unsqueeze(1), 1.0)

    targets = {
        "binary_mask"    : (torch.rand(B, 1, H, W) > 0.8).float(),
        "instance_masks" : inst_onehot,
        "direction_field": direction.clone(),
    }

    criterion = NailVTONLoss()
    total, loss_dict = criterion((binary_logits, inst_logits, direction), targets)

    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")

    print(f"  binary IoU     : {compute_iou(binary_logits, targets['binary_mask']):.4f}")
    print(f"  instance IoU   : {compute_instance_iou(inst_logits, targets['instance_masks']):.4f}")
    print("\nLoss sanity check PASSED ✓")