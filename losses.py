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
        """logits, targets: (B, 1, H, W) or (B, C, H, W)"""
        loss_map  = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        # Global LMP: Flatten ALL pixels in the minibatch
        loss_flat = loss_map.view(-1)
        k         = max(1, int(loss_flat.size(0) * self.keep_ratio))
        top_k, _  = loss_flat.topk(k)
        return top_k.mean()


# ── Direction Loss ────────────────────────────────────────────────────────────

class DirectionLoss(nn.Module):
    def forward(self, pred_dir, target_dir, valid_mask):
        """
        pred_dir   : (B, 2, H, W)
        target_dir : (B, 2, H, W) normalized
        valid_mask : (B, 1, H, W) float
        
        Strict Equation 3 (arXiv:1906.02222): 1/(H*W) * sum(||u_pred - u_target||^2)
        Penalized ONLY in the annotated fingernail area.
        """
        B, C, H, W = pred_dir.shape
        diff  = pred_dir - target_dir
        l2_sq = (diff ** 2).sum(dim=1)  # (B, H, W)
        
        valid_sq = (valid_mask.squeeze(1) > 0.5).float()
        l2_sq_masked = l2_sq * valid_sq
        
        # We sum over all FOREGROUND pixels and divide by the total area B*H*W
        return l2_sq_masked.sum() / (B * H * W)


# ── Binary Segmentation Loss ──────────────────────────────────────────────────

class BinarySegLoss(nn.Module):
    def __init__(self, keep_ratio=0.1):
        super().__init__()
        self.lmp = LMPLoss(keep_ratio=keep_ratio)

    def forward(self, logits, targets):
        return self.lmp(logits, targets)




class NailVTONLoss(nn.Module):
    """
    Combined loss over the Laplacian pyramid.
    Sum of unweighted losses across all Levels returned by the model.
    """
    def __init__(self, lmp_ratio=0.1, w_binary=1.0, w_direction=1.0, **kwargs):
        super().__init__()
        self.w_bin = w_binary
        self.w_dir = w_direction
        self.binary_loss    = BinarySegLoss(keep_ratio=lmp_ratio)
        self.direction_loss = DirectionLoss()

    def _get_target_level(self, targets, size):
        """Linearly interpolate targets to match the scale of the pyramid level."""
        H, W = size
        bin_t  = F.interpolate(targets["binary_mask"], size=(H, W), mode="nearest")
        # Direction field is float32 vectors, should use bilinear interpolation
        dir_t  = F.interpolate(targets["direction_field"], size=(H, W), mode="bilinear", align_corners=False)
        
        # Norm dir_t after interpolation using broadcasting
        norm  = dir_t.norm(dim=1, keepdim=True)
        dir_t = dir_t / norm.clamp(min=1e-6)

        return {"binary_mask": bin_t, "direction_field": dir_t}

    def forward(self, multi_predictions, targets):
        """
        multi_predictions: list of tuples (binary, direction)
        """
        total_loss = 0.0
        details = {}

        for i, preds in enumerate(multi_predictions):
            p_bin, p_dir = preds
            h, w = p_bin.shape[-2:]
            
            # Prepare targets for this resolution
            target_lvl = self._get_target_level(targets, (h, w))
            valid_mask = target_lvl["binary_mask"]  # Foreground mask
            
            l_bin  = self.binary_loss(p_bin, target_lvl["binary_mask"])
            l_dir  = self.direction_loss(p_dir, target_lvl["direction_field"], valid_mask)
            
            l_lvl = (self.w_bin * l_bin) + (self.w_dir * l_dir)
            total_loss += l_lvl
            
            details[f"l{i}_total"] = l_lvl.item()
            details[f"l{i}_bin"]   = l_bin.item()
            details[f"l{i}_dir"]   = l_dir.item()

        details["loss_total"] = total_loss.item()
        return total_loss, details

# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_iou(pred_mask, target_mask, threshold=0.5, eps=1e-6):
    if pred_mask.max() > 1.0 or pred_mask.min() < 0.0:
        pred_binary = (torch.sigmoid(pred_mask) > threshold).float()
    else:
        pred_binary = (pred_mask > threshold).float()

    # Foreground IoU
    intersection_fg = (pred_binary * target_mask).sum(dim=(-2, -1))
    union_fg        = (pred_binary + target_mask).clamp(0, 1).sum(dim=(-2, -1))
    iou_fg          = (intersection_fg + eps) / (union_fg + eps)
    
    # Background IoU
    pred_bg   = 1.0 - pred_binary
    target_bg = 1.0 - target_mask
    intersection_bg = (pred_bg * target_bg).sum(dim=(-2, -1))
    union_bg        = (pred_bg + target_bg).clamp(0, 1).sum(dim=(-2, -1))
    iou_bg          = (intersection_bg + eps) / (union_bg + eps)
    
    # binary mIoU
    miou = (iou_fg + iou_bg) / 2.0
    return miou.mean().item()


if __name__ == "__main__":
    from model import NailVTONModel
    model = NailVTONModel(image_size=512, pretrained=False)
    dummy = torch.randn(2, 3, 512, 512)
    outs = model(dummy)
    
    targets = {
        "binary_mask": (torch.rand(2, 1, 512, 512) > 0.8).float(),
        "direction_field": torch.randn(2, 2, 512, 512)
    }
    
    criterion = NailVTONLoss()
    total, logs = criterion(outs, targets)
    
    print("Loss logs:")
    for k, v in logs.items():
        print(f"  {k}: {v:.4f}")
    print("\nLoss sanity check PASSED ✓")
