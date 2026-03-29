"""
Nail VTON Loss Functions
------------------------
Strict adherence to VTNFP (Duke et al., 2019):
  Laplacian Pyramid Loss: L = sum(L_level)
  Each level loss: L_level = L_fgbg + L_class + L_field
  LMP used for fgbg at 10% ratio.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Loss Max-Pooling ───────────────────────────────────────────────────────────

class LMPLoss(nn.Module):
    def __init__(self, keep_ratio=0.1):
        super().__init__()
        self.keep_ratio = keep_ratio

    def forward(self, logits, targets):
        """logits, targets: (B, 1, H, W)"""
        loss_map  = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        B         = loss_map.shape[0]
        loss_flat = loss_map.view(B, -1)
        k         = max(1, int(loss_flat.size(1) * self.keep_ratio))
        top_k, _  = loss_flat.topk(k, dim=1)
        return top_k.mean()


# ── Direction Loss ────────────────────────────────────────────────────────────

class DirectionLoss(nn.Module):
    def forward(self, pred_dir, target_dir):
        """
        pred_dir   : (B, 2, H, W)
        target_dir : (B, 2, H, W) normalized + (0,0) mask
        """
        magnitude  = target_dir.norm(dim=1, keepdim=True)
        valid_mask = (magnitude > 1e-6).squeeze(1)

        if not valid_mask.any():
            return torch.tensor(0.0, device=pred_dir.device, requires_grad=True)

        diff = pred_dir - target_dir
        l2_sq = (diff ** 2).sum(dim=1)
        return l2_sq[valid_mask].mean()


# ── Binary Segmentation Loss ──────────────────────────────────────────────────

class BinarySegLoss(nn.Module):
    def __init__(self, keep_ratio=0.1):
        super().__init__()
        self.lmp = LMPLoss(keep_ratio=keep_ratio)

    def forward(self, logits, targets):
        return self.lmp(logits, targets)


# ── Instance Segmentation Loss ────────────────────────────────────────────────

class InstanceSegLoss(nn.Module):
    def forward(self, logits, targets, valid_mask):
        """
        logits     : (B, 10, H, W)
        targets    : (B, 10, H, W) one-hot float
        valid_mask : (B,  1, H, W) float
        """
        target_idx = targets.argmax(dim=1)
        valid_bool = valid_mask.squeeze(1) > 0.5

        if not valid_bool.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        logits_valid = logits.permute(0, 2, 3, 1)[valid_bool]
        target_valid = target_idx[valid_bool]

        return F.cross_entropy(logits_valid, target_valid)


# ── Pyramid Loss ─────────────────────────────────────────────────────────────

class NailVTONLoss(nn.Module):
    """
    Combined loss over the Laplacian pyramid.
    Sum of unweighted losses across all Levels returned by the model.
    """
    def __init__(self, lmp_ratio=0.1):
        super().__init__()
        self.binary_loss    = BinarySegLoss(keep_ratio=lmp_ratio)
        self.instance_loss  = InstanceSegLoss()
        self.direction_loss = DirectionLoss()

    def _get_target_level(self, targets, size):
        """Linearly interpolate targets to match the scale of the pyramid level."""
        H, W = size
        bin_t  = F.interpolate(targets["binary_mask"], size=(H, W), mode="nearest")
        inst_t = F.interpolate(targets["instance_masks"], size=(H, W), mode="nearest")
        # Direction field is float32 vectors, should use bilinear interpolation
        dir_t  = F.interpolate(targets["direction_field"], size=(H, W), mode="bilinear", align_corners=False)
        
        # Norm dir_t after interpolation using broadcasting
        norm  = dir_t.norm(dim=1, keepdim=True)
        dir_t = dir_t / norm.clamp(min=1e-6)

        return {"binary_mask": bin_t, "instance_masks": inst_t, "direction_field": dir_t}

    def forward(self, multi_predictions, targets):
        """
        multi_predictions: list of tuples (binary, instance, direction)
        """
        total_loss = 0.0
        details = {}

        for i, preds in enumerate(multi_predictions):
            p_bin, p_inst, p_dir = preds
            h, w = p_bin.shape[-2:]
            
            # Prepare targets for this resolution
            target_lvl = self._get_target_level(targets, (h, w))
            
            l_bin  = self.binary_loss(p_bin,   target_lvl["binary_mask"])
            l_inst = self.instance_loss(p_inst, target_lvl["instance_masks"], target_lvl["binary_mask"])
            l_dir  = self.direction_loss(p_dir,  target_lvl["direction_field"])
            
            l_lvl = l_bin + l_inst + l_dir
            total_loss += l_lvl
            
            details[f"l{i}_total"] = l_lvl.item()
            details[f"l{i}_bin"]   = l_bin.item()
            details[f"l{i}_inst"]  = l_inst.item()
            details[f"l{i}_dir"]   = l_dir.item()

        details["loss_total"] = total_loss.item()
        return total_loss, details

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
    valid_bool = binary_mask.squeeze(1) > 0.5
    if not valid_bool.any():
        return 0.0

    preds = inst_logits.argmax(dim=1)
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

if __name__ == "__main__":
    from model import NailVTONModel
    model = NailVTONModel(image_size=512, pretrained=False)
    dummy = torch.randn(2, 3, 512, 512)
    outs = model(dummy)
    
    # Fake targets at 512x512
    targets = {
        "binary_mask": (torch.rand(2, 1, 512, 512) > 0.8).float(),
        "instance_masks": torch.zeros(2, 10, 512, 512),
        "direction_field": torch.randn(2, 2, 512, 512)
    }
    
    criterion = NailVTONLoss()
    total, logs = criterion(outs, targets)
    
    print("Loss logs:")
    for k, v in logs.items():
        print(f"  {k}: {v:.4f}")
    print("\nLoss sanity check PASSED ✓")
