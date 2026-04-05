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
    def __init__(self, keep_ratio=0.15):  # Increased 0.10 -> 0.15 for harder background
        super().__init__()
        self.keep_ratio = keep_ratio

    def forward(self, logits, targets):
        """
        logits  : (B, 2, H, W)
        targets : (B, 1, H, W) float mask (0 or 1)
        """
        # Equation 1: Multinomial NLL (Cross Entropy)
        targets_long = targets.squeeze(1).long()
        loss_map = F.cross_entropy(logits, targets_long, reduction="none")
        B         = loss_map.shape[0]
        loss_flat = loss_map.view(B, -1)
        k         = max(1, int(loss_flat.size(1) * self.keep_ratio))
        top_k, _  = loss_flat.topk(k, dim=1)
        return top_k.mean()


# ── Soft Dice Loss (fixes Patchiness & Ghost Nails) ───────────────────────────

class SoftDiceLoss(nn.Module):
    """
    Treats the entire nail as ONE connected object.
    Mathematically punishes 'holes' and 'patches' inside the predicted mask.
    Complements NLL which works pixel-by-pixel.
    """
    def forward(self, logits, targets, eps=1e-6):
        # Get the foreground probability (channel 1)
        prob_fg = torch.softmax(logits, dim=1)[:, 1]   # (B, H, W)
        target  = targets.squeeze(1).float()             # (B, H, W)
        
        intersection = (prob_fg * target).sum(dim=(-2, -1))
        cardinality  = prob_fg.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
        dice         = (2. * intersection + eps) / (cardinality + eps)
        return 1.0 - dice.mean()  # Loss = 1 - Dice Score


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
        
        # Precise Equation 3: 1/(H*W) * sum(||u_pred - u_target||^2)
        # Penalized only in foreground, but divided by total area (B*H*W).
        return l2_sq_masked.sum() / (B * H * W)


# ── Binary Segmentation Loss ──────────────────────────────────────────────────

# ── Sobel Edge Loss (forces sharp boundaries) ─────────────────────────────────

class SobelEdgeLoss(nn.Module):
    """
    Calculates the difference in image gradients (edges) between prediction and truth.
    Forces the model to align the 'cuticle line' curve perfectly.
    """
    def __init__(self):
        super().__init__()
        # Sobel kernels: 1x1x3x3
        kx = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32)
        ky = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32)
        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)

    def _get_edges(self, x):
        # x is (B, 1, H, W)
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        return torch.sqrt(gx**2 + gy**2 + 1e-6)

    def forward(self, logits, targets):
        # prob_fg is (B, 1, H, W)
        prob_fg = torch.softmax(logits, dim=1)[:, 1:2]
        pred_edges = self._get_edges(prob_fg)
        true_edges = self._get_edges(targets.float())
        return F.mse_loss(pred_edges, true_edges)


# ── Binary Segmentation Loss ──────────────────────────────────────────────────

class BinarySegLoss(nn.Module):
    """
    Combined NLL-LMP + Soft Dice Loss + Sobel Edge Loss.
    - NLL-LMP (40%): Fixes hard pixels (background mistakes).
    - Dice (40%): Fixes connectivity and holes.
    - Edge (20%): Fixes high-precision boundary alignment.
    """
    def __init__(self, keep_ratio=0.15, alpha=0.4, beta=0.2):
        super().__init__()
        self.lmp   = LMPLoss(keep_ratio=keep_ratio)
        self.dice  = SoftDiceLoss()
        self.edge  = SobelEdgeLoss()
        self.alpha = alpha # Weight for Dice
        self.beta  = beta  # Weight for Edge

    def forward(self, logits, targets):
        l_nll  = self.lmp(logits, targets)
        l_dice = self.dice(logits, targets)
        l_edge = self.edge(logits, targets)
        
        # Weighted sum: (1 - alpha - beta) * NLL + alpha * Dice + beta * Edge
        # Default: 0.4*NLL + 0.4*Dice + 0.2*Edge
        gamma = 1.0 - self.alpha - self.beta
        return gamma * l_nll + self.alpha * l_dice + self.beta * l_edge




class NailVTONLoss(nn.Module):
    """
    Combined loss over the Laplacian pyramid.
    Sum of unweighted losses across all Levels returned by the model.
    """
    def __init__(self, lmp_ratio=0.15, dice_alpha=0.5):
        super().__init__()
        self.binary_loss    = BinarySegLoss(keep_ratio=lmp_ratio, alpha=dice_alpha)
        self.direction_loss = DirectionLoss()

    def _get_target_level(self, t_bin, t_dir, size):
        """Linearly interpolate targets to match the scale of the pyramid level."""
        H, W = size
        bin_t  = F.interpolate(t_bin, size=(H, W), mode="nearest")
        # Direction field is float32 vectors, should use bilinear interpolation
        dir_t  = F.interpolate(t_dir, size=(H, W), mode="bilinear", align_corners=False)
        
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
            
            # Phase 6: targets is a 3-tuple (img_t, bin_t, dir_t)
            t_bin, t_dir = targets[1], targets[2]
            
            # Prepare targets for this resolution
            target_lvl = self._get_target_level(t_bin, t_dir, (h, w))
            valid_mask = target_lvl["binary_mask"]  # Foreground mask
            
            l_bin  = self.binary_loss(p_bin, target_lvl["binary_mask"])
            l_dir  = self.direction_loss(p_dir, target_lvl["direction_field"], valid_mask)
            
            l_lvl = l_bin + l_dir
            total_loss += l_lvl
            
            # Phase 4 Lockdown: Explicitly detach and copy to CPU item
            details[f"l{i}_total"] = l_lvl.detach().cpu().item()
            details[f"l_bin_{i}"]   = l_bin.detach().cpu().item()
            details[f"l_dir_{i}"]   = l_dir.detach().cpu().item()

        details["l2_dir"]     = float(sum(details[f"l_dir_{i}"] for i in range(len(multi_predictions))) / len(multi_predictions))
        details["loss_total"] = total_loss.detach().cpu().item()
        return total_loss, details

# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_miou(pred_mask, target_mask, threshold=0.5, eps=1e-6):
    """
    Computes Mean Intersection over Union (mIoU) for binary segmentation.
    mIoU = (IoU_foreground + IoU_background) / 2
    """
    if pred_mask.shape[1] == 2:
        # Multinomial NLL setup: Take index 1 (foreground)
        prob_fg = torch.softmax(pred_mask, dim=1)[:, 1:2]
        pred_binary = (prob_fg > threshold).float()
    else:
        # Legacy single-channel setup
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

    # mIoU is mean of foreground and background
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
