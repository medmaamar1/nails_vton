"""
Nail VTON Loss Functions
------------------------
Strict adherence to VTNFP (Duke et al., 2019):
  1. LMPLoss          : Loss Max-Pooling (p=1, keep_ratio=0.1)
  2. BinarySegLoss    : LMP over BCE (NLL for 2 classes)
  3. InstanceSegLoss  : NLL (Cross-entropy) calculated ONLY inside fingernail regions
  4. DirectionLoss    : L2 distance for direction field
  5. NailVTONLoss     : L_fgbg + L_class + L_field (unweighted sum)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Loss Max-Pooling ───────────────────────────────────────────────────────────

class LMPLoss(nn.Module):
    """
    Loss Max-Pooling from the paper (Section 3.3).
    "taking the mean over the top 10% of pixels as the minibatch loss."
    """
    def __init__(self, keep_ratio=0.1):
        super().__init__()
        self.keep_ratio = keep_ratio

    def forward(self, logits, targets):
        """logits, targets: (B, 1, H, W)"""
        loss_map  = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )                                                   # (B, 1, H, W)
        B         = loss_map.shape[0]
        loss_flat = loss_map.view(B, -1)                   # (B, 1*H*W)
        k         = max(1, int(loss_flat.size(1) * self.keep_ratio))
        top_k, _  = loss_flat.topk(k, dim=1)
        return top_k.mean()


# ── Direction Loss ────────────────────────────────────────────────────────────

class DirectionLoss(nn.Module):
    """
    Strict L2 loss on normalised base to tip direction.
    Equation 3 in paper: ||u_pred - u_target||_2^2 over valid pixels.
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

        # L2 Distance squared: ||pred - target||^2
        diff = pred_dir - target_dir
        l2_sq = (diff ** 2).sum(dim=1)                         # (B,H,W)
        return l2_sq[valid_mask].mean()


# ── Binary Segmentation Loss ──────────────────────────────────────────────────

class BinarySegLoss(nn.Module):
    """Binary NLL (BCE) with LMP."""
    def __init__(self, keep_ratio=0.1):
        super().__init__()
        self.lmp = LMPLoss(keep_ratio=keep_ratio)

    def forward(self, logits, targets):
        """logits, targets: (B, 1, H, W)"""
        return self.lmp(logits, targets)


# ── Instance Segmentation Loss ────────────────────────────────────────────────

class InstanceSegLoss(nn.Module):
    """
    Strict 10-class NLL. The paper says: 
    "since fingernail class predictions are only valid in the fingernail region,
    those classes are balanced and do not require LMP."
    """
    def forward(self, logits, targets, valid_mask):
        """
        logits     : (B, 10, H, W)
        targets    : (B, 10, H, W) one-hot float
        valid_mask : (B,  1, H, W) float from binary target
        """
        target_idx = targets.argmax(dim=1)                    # (B, H, W) long
        valid_bool = valid_mask.squeeze(1) > 0.5              # (B, H, W) bool

        if not valid_bool.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        logits_valid = logits.permute(0, 2, 3, 1)[valid_bool] # (N_valid, 10)
        target_valid = target_idx[valid_bool]                 # (N_valid,)

        return F.cross_entropy(logits_valid, target_valid)


# ── Combined Loss ─────────────────────────────────────────────────────────────

class NailVTONLoss(nn.Module):
    """
    L = w_binary*L_fgbg + w_class*L_class + w_field*L_field
    Paper defaults to unweighted (1.0).
    """
    def __init__(self, lmp_ratio=0.1, w_binary=1.0, w_instance=1.0, w_direction=1.0):
        super().__init__()
        self.w_binary    = w_binary
        self.w_instance  = w_instance
        self.w_direction = w_direction
        
        self.binary_loss    = BinarySegLoss(keep_ratio=lmp_ratio)
        self.instance_loss  = InstanceSegLoss()
        self.direction_loss = DirectionLoss()

    def forward(self, predictions, targets):
        binary_logits, inst_logits, direction = predictions

        l_bin  = self.binary_loss(binary_logits, targets["binary_mask"])
        l_inst = self.instance_loss(inst_logits, targets["instance_masks"], targets["binary_mask"])
        l_dir  = self.direction_loss(direction,  targets["direction_field"])

        # Strictly following the paper's math: L = L_fgbg + L_class + L_field
        total = l_bin + l_inst + l_dir

        return total, {
            "loss_total"    : total.item(),
            "loss_binary"   : l_bin.item(),
            "loss_instance" : l_inst.item(),
            "loss_direction": l_dir.item(),
        }


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_iou(pred_mask, target_mask, threshold=0.5, eps=1e-6):
    if pred_mask.max() > 1.0 or pred_mask.min() < 0.0:
        pred_binary = (torch.sigmoid(pred_mask) > threshold).float()
    else:
        pred_binary = (pred_mask > threshold).float()

    intersection = (pred_binary * target_mask).sum(dim=(-2, -1))
    union        = (pred_binary + target_mask).clamp(0, 1).sum(dim=(-2, -1))
    iou          = (intersection + eps) / (union + eps)
    return iou.mean().item()


def compute_instance_iou(inst_logits, target_masks, binary_mask, eps=1e-6):
    """
    Mean IoU over valid nail regions (channels 0-9).
    """
    valid_bool = binary_mask.squeeze(1) > 0.5                 # (B, H, W)
    if not valid_bool.any():
        return 0.0

    preds = inst_logits.argmax(dim=1)                         # (B, H, W)

    total_iou = 0.0
    valid_classes = 0

    for c in range(inst_logits.shape[1]):
        pred_c   = (preds == c) & valid_bool
        target_c = (target_masks[:, c, :, :] > 0.5) & valid_bool

        if target_c.sum() > 0:
            inter = (pred_c & target_c).sum().float()
            union = (pred_c | target_c).sum().float()
            total_iou += (inter + eps) / (union + eps)
            valid_classes += 1

    return (total_iou / valid_classes).item() if valid_classes > 0 else 0.0


# ── Sanity check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, H, W, MAX_I = 2, 512, 512, 10

    binary_logits = torch.randn(B,      1, H, W)
    inst_logits   = torch.randn(B,  MAX_I, H, W)
    direction     = torch.randn(B,      2, H, W)
    direction     = direction / direction.norm(dim=1, keepdim=True).clamp(min=1e-6)

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
    print(f"  instance IoU   : {compute_instance_iou(inst_logits, targets['instance_masks'], targets['binary_mask']):.4f}")
    print("\nLoss sanity check PASSED ✓")
