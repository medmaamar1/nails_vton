"""
Nail VTON Loss Functions
------------------------
Single-task binary segmentation only. Direction loss removed.

Active losses:
  LMPLoss        — NLL with top-K hard pixel mining
  SoftDiceLoss   — Penalises holes and patchy masks
  SobelEdgeLoss  — Forces sharp, precise cuticle boundaries
  BinarySegLoss  — Combined: 0.4*NLL + 0.4*Dice + 0.2*Edge
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Loss Max-Pooling ───────────────────────────────────────────────────────────

class LMPLoss(nn.Module):
    """Top-K hard pixel NLL. Keeps only the 15% hardest pixels per batch."""
    def __init__(self, keep_ratio=0.15):
        super().__init__()
        self.keep_ratio = keep_ratio

    def forward(self, logits, targets):
        # logits : (B, 2, H, W)
        # targets: (B, 1, H, W) float32  {0.0, 1.0}
        targets_long = targets.squeeze(1).long()                      # (B, H, W)
        loss_map     = F.cross_entropy(logits, targets_long, reduction="none")
        B            = loss_map.shape[0]
        loss_flat    = loss_map.view(B, -1)
        k            = max(1, int(loss_flat.size(1) * self.keep_ratio))
        top_k, _     = loss_flat.topk(k, dim=1)
        return top_k.mean()


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
    40% NLL-LMP  — hard pixel mining (knuckle / palm FPs)
    40% Soft Dice — connectivity and hole prevention
    20% Sobel Edge — cuticle boundary sharpness
    """
    def __init__(self, keep_ratio=0.15, alpha=0.4, beta=0.2):
        super().__init__()
        self.lmp   = LMPLoss(keep_ratio=keep_ratio)
        self.dice  = SoftDiceLoss()
        self.edge  = SobelEdgeLoss()
        self.alpha = alpha  # Dice weight
        self.beta  = beta   # Edge weight

    def forward(self, logits, targets):
        # logits : (B, 2, H, W)
        # targets: (B, 1, H, W) float32
        l_nll  = self.lmp(logits, targets)
        l_dice = self.dice(logits, targets)
        l_edge = self.edge(logits, targets)
        gamma  = 1.0 - self.alpha - self.beta  # NLL weight = 0.4
        return gamma * l_nll + self.alpha * l_dice + self.beta * l_edge


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


# ── Sanity check ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    criterion = BinarySegLoss()
    logits  = torch.randn(2, 2, 448, 448)
    targets = (torch.rand(2, 1, 448, 448) > 0.8).float()
    loss    = criterion(logits, targets)
    miou    = compute_miou(logits, targets)
    print(f"Loss : {loss.item():.4f}")
    print(f"mIoU : {miou:.4f}")
    assert not torch.isnan(loss), "NaN detected in loss!"
    print("Loss sanity check PASSED ✓")
