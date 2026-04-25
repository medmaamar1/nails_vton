"""
Nail VTON Loss Functions
------------------------
Binary segmentation only. Works with DeepLabV3+ output.

Input convention (all losses):
  logits  : (B, 1, H, W)  raw logits from DeepLabV3+
  targets : (B, 1, H, W)  float32 binary {0.0, 1.0}

Active losses:
  LMPLoss        — Loss Max-Pooling: mean over top-10% hardest pixels
  TverskyLoss    — Heavily penalizes False Positives (alpha=0.15, beta=0.85)
  SoftDiceLoss   — Penalises holes and patchy masks
  SobelEdgeLoss  — Forces sharp, precise cuticle boundaries
  BinarySegLoss  — 30% LMP + 15% Tversky + 25% Dice + 30% Edge
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Loss Max-Pooling ───────────────────────────────────────────────────────────

class LMPLoss(nn.Module):
    """
    Loss Max-Pooling: mean over the top-p fraction of hardest pixels.
    Uses binary cross-entropy per pixel (model outputs 1 logit per pixel).
    Paper reports +2% mIoU vs class-weighted baseline for nail segmentation.
    """
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Per-pixel BCE, no reduction — shape: (B, 1, H, W)
        loss_map  = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        loss_flat = loss_map.view(-1)
        n_keep    = max(1, int(self.p * loss_flat.numel()))
        top_loss, _ = torch.topk(loss_flat, n_keep)
        return top_loss.mean()


# ── Tversky Loss ───────────────────────────────────────────────────────────────

class TverskyLoss(nn.Module):
    """
    A generalisation of Dice loss.
    alpha=0.15, beta=0.85 → very heavily penalizes False Positives (Ghost Nails).
    """
    def __init__(self, alpha: float = 0.15, beta: float = 0.85, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.eps   = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs  = torch.sigmoid(logits).squeeze(1)   # (B, H, W)
        target = targets.squeeze(1).float()          # (B, H, W)

        tp = (probs * target).sum(dim=(-2, -1))
        fn = ((1 - probs) * target).sum(dim=(-2, -1))
        fp = (probs * (1 - target)).sum(dim=(-2, -1))

        tversky = (tp + self.eps) / (tp + self.alpha * fn + self.beta * fp + self.eps)
        return 1.0 - tversky.mean()


# ── Soft Dice Loss ─────────────────────────────────────────────────────────────

class SoftDiceLoss(nn.Module):
    """Treats the nail as ONE connected object. Punishes holes and patches."""
    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                eps: float = 1e-6) -> torch.Tensor:
        prob_fg = torch.sigmoid(logits).squeeze(1)   # (B, H, W)
        target  = targets.squeeze(1).float()          # (B, H, W)

        intersection = (prob_fg * target).sum(dim=(-2, -1))
        cardinality  = prob_fg.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
        dice         = (2.0 * intersection + eps) / (cardinality + eps)
        return 1.0 - dice.mean()


# ── Sobel Edge Loss ────────────────────────────────────────────────────────────

class SobelEdgeLoss(nn.Module):
    """
    Penalises blurry or misaligned nail boundaries.
    Computes MSE between the Sobel edge maps of prediction and ground truth.
    """
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[[[-1,  0,  1],
                              [-2,  0,  2],
                              [-1,  0,  1]]]], dtype=torch.float32)
        ky = torch.tensor([[[[-1, -2, -1],
                              [ 0,  0,  0],
                              [ 1,  2,  1]]]], dtype=torch.float32)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def _edges(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, 1, H, W)
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        prob_fg    = torch.sigmoid(logits)        # (B, 1, H, W)
        pred_edges = self._edges(prob_fg)
        true_edges = self._edges(targets.float())
        return F.mse_loss(pred_edges, true_edges)


# ── Combined Binary Segmentation Loss ─────────────────────────────────────────

class BinarySegLoss(nn.Module):
    """
    Main training loss for DeepLabV3+ binary nail segmentation.

    Weights:
      30% LMP     — hardest-pixel BCE (top-10% loss pixels)
      15% Tversky — FP suppression (ghost nail prevention)
      25% Dice    — mask connectivity and completeness
      30% Edge    — sharp cuticle boundaries

    Args:
        logits  : (B, 1, H, W) raw logits from model
        targets : (B, 1, H, W) float32 binary {0, 1}
    Returns:
        Scalar loss tensor.
    """
    def __init__(self, alpha: float = 0.15, beta: float = 0.85):
        super().__init__()
        self.lmp     = LMPLoss(p=0.1)
        self.tversky = TverskyLoss(alpha=alpha, beta=beta)
        self.dice    = SoftDiceLoss()
        self.edge    = SobelEdgeLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        l_lmp     = self.lmp(logits, targets)
        l_tversky = self.tversky(logits, targets)
        l_dice    = self.dice(logits, targets)
        l_edge    = self.edge(logits, targets)
        return 0.30 * l_lmp + 0.15 * l_tversky + 0.25 * l_dice + 0.30 * l_edge


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_miou(logits: torch.Tensor, targets: torch.Tensor,
                 threshold: float = 0.5, eps: float = 1e-6) -> float:
    """
    Mean IoU for binary segmentation.

    Args:
        logits  : (B, 1, H, W) raw logits
        targets : (B, 1, H, W) float32 {0, 1}
    Returns:
        Scalar mean IoU (foreground + background averaged).
    """
    prob_fg     = torch.sigmoid(logits)
    pred_binary = (prob_fg > threshold).float()

    # Foreground IoU
    i_fg   = (pred_binary * targets).sum(dim=(-2, -1))
    u_fg   = (pred_binary + targets).clamp(0, 1).sum(dim=(-2, -1))
    iou_fg = (i_fg + eps) / (u_fg + eps)

    # Background IoU
    pred_bg = 1.0 - pred_binary
    tgt_bg  = 1.0 - targets
    i_bg    = (pred_bg * tgt_bg).sum(dim=(-2, -1))
    u_bg    = (pred_bg + tgt_bg).clamp(0, 1).sum(dim=(-2, -1))
    iou_bg  = (i_bg + eps) / (u_bg + eps)

    return ((iou_fg + iou_bg) / 2.0).mean().item()
