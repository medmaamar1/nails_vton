"""
Nail VTON Loss Functions
------------------------
Single-task binary segmentation only. Direction loss removed.

Active losses:
  TverskyLoss    — Heavily penalizes False Positives (Ghost-Buster)
  SoftDiceLoss   — Penalises holes and patchy masks
  SobelEdgeLoss  — Forces sharp, precise cuticle boundaries
  BinarySegLoss  — Combined: 20% Tversky + 40% Dice + 40% Edge
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Tversky Loss (The Ghost-Buster) ─────────────────────────────────────────────

class TverskyLoss(nn.Module):
    """
    A generalization of Dice loss.
    alpha=0.3, beta=0.7 -> Heavily penalizes False Positives (Ghost Nails).
    """
    def __init__(self, alpha=0.3, beta=0.7, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.eps   = eps

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)[:, 1]  # (B, H, W)
        target = targets.squeeze(1).float()          # (B, H, W)

        tp = (probs * target).sum(dim=(-2, -1))
        fn = ((1 - probs) * target).sum(dim=(-2, -1))
        fp = (probs * (1 - target)).sum(dim=(-2, -1))

        tversky = (tp + self.eps) / (tp + self.alpha * fn + self.beta * fp + self.eps)
        return 1.0 - tversky.mean()


# ── Soft Dice Loss ─────────────────────────────────────────────────────────────

class SoftDiceLoss(nn.Module):
    """Treats the nail as ONE connected object. Punishes holes and patches."""
    def forward(self, logits, targets, eps=1e-6):
        prob_fg = torch.softmax(logits, dim=1)[:, 1]   # (B, H, W)
        target  = targets.squeeze(1).float()             # (B, H, W)
        intersection = (prob_fg * target).sum(dim=(-2, -1))
        cardinality  = prob_fg.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
        dice         = (2. * intersection + eps) / (cardinality + eps)
        return 1.0 - dice.mean()


# ── Sobel Edge Loss ────────────────────────────────────────────────────────────

class SobelEdgeLoss(nn.Module):
    """
    Penalises blurry or mis-aligned nail boundaries.
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

    def _edges(self, x):
        # x: (B, 1, H, W)
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        return torch.sqrt(gx**2 + gy**2 + 1e-6)

    def forward(self, logits, targets):
        prob_fg    = torch.softmax(logits, dim=1)[:, 1:2]  # (B, 1, H, W)
        pred_edges = self._edges(prob_fg)
        true_edges = self._edges(targets.float())
        return F.mse_loss(pred_edges, true_edges)


# ── Combined Binary Segmentation Loss ─────────────────────────────────────────

class BinarySegLoss(nn.Module):
    """
    20% Tversky Loss — focus ONLY on False Positives (Ghost-Busting)
    40% Soft Dice — overall mask connectivity
    40% Sobel Edge — sharp cuticle boundary lock-on
    """
    def __init__(self, alpha=0.3, beta=0.7, edge_weight=0.4, dice_weight=0.4):
        super().__init__()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta)
        self.dice    = SoftDiceLoss()
        self.edge    = SobelEdgeLoss()
        self.alpha_w = dice_weight
        self.beta_w  = edge_weight

    def forward(self, logits, targets):
        # logits : (B, 2, H, W)
        # targets: (B, 1, H, W) float32
        l_tversky = self.tversky(logits, targets)
        l_dice    = self.dice(logits, targets)
        l_edge    = self.edge(logits, targets)
        
        gamma_w   = 1.0 - self.alpha_w - self.beta_w  # Tversky weight = 0.2
        loss      = gamma_w * l_tversky + self.alpha_w * l_dice + self.beta_w * l_edge
        
        # Background-Only Penalty: If the image has zero nail pixels, punish FPs double
        with torch.no_grad():
            is_empty_batch = targets.sum() == 0
        if is_empty_batch:
            loss = loss * 2.0
            
        return loss


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_miou(logits, targets, threshold=0.5, eps=1e-6):
    """
    mIoU for binary segmentation.
    logits : (B, 2, H, W)
    targets: (B, 1, H, W) float32 {0,1}
    """
    prob_fg     = torch.softmax(logits, dim=1)[:, 1:2]
    pred_binary = (prob_fg > threshold).float()

    # Foreground IoU
    i_fg  = (pred_binary * targets).sum(dim=(-2, -1))
    u_fg  = (pred_binary + targets).clamp(0, 1).sum(dim=(-2, -1))
    iou_fg = (i_fg + eps) / (u_fg + eps)

    # Background IoU
    pred_bg   = 1.0 - pred_binary
    tgt_bg    = 1.0 - targets
    i_bg  = (pred_bg * tgt_bg).sum(dim=(-2, -1))
    u_bg  = (pred_bg + tgt_bg).clamp(0, 1).sum(dim=(-2, -1))
    iou_bg = (i_bg + eps) / (u_bg + eps)

    return ((iou_fg + iou_bg) / 2.0).mean().item()
