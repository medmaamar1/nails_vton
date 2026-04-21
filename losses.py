"""
Nail VTON Loss Functions
------------------------
Single-task binary segmentation only. Direction loss removed.

Active losses:
  LMPLoss        — Loss Max-Pooling: mean over top-10% hardest pixels (paper §3.3, +2% mIoU)
  TverskyLoss    — Very heavily penalizes False Positives (alpha=0.15, beta=0.85)
  SoftDiceLoss   — Penalises holes and patchy masks
  SobelEdgeLoss  — Forces sharp, precise cuticle boundaries
  PresenceGate   — BCE on global nail-presence gate
  BinarySegLoss  — Main: 30% LMP + 15% Tversky + 25% Dice + 30% Edge
                   + 40% weight * LMP on intermediate (Laplacian pyramid, paper §3.2)
                   + 30% Gate BCE
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Loss Max-Pooling (paper §3.3) ─────────────────────────────────────────────

class LMPLoss(nn.Module):
    """Loss Max-Pooling: sort all pixels in the minibatch by cross-entropy loss,
    take the mean of the top-p fraction (default p=0.1 = hardest 10%).
    Paper reports +2% mIoU vs class-weighted baseline for nail segmentation."""
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, logits, targets):
        # Per-pixel cross-entropy, no reduction
        target_long = targets.squeeze(1).long()                       # (B, H, W)
        loss_map    = F.cross_entropy(logits, target_long, reduction="none")  # (B, H, W)

        loss_flat = loss_map.view(-1)
        n_keep    = max(1, int(self.p * loss_flat.numel()))
        top_loss, _ = torch.topk(loss_flat, n_keep)
        return top_loss.mean()


# ── Tversky Loss (The Ghost-Buster) ─────────────────────────────────────────────

class TverskyLoss(nn.Module):
    """
    A generalization of Dice loss.
    alpha=0.15, beta=0.85 -> Very heavily penalizes False Positives (Ghost Nails).
    """
    def __init__(self, alpha=0.15, beta=0.85, eps=1e-6):
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
    Main output (full-res):
      30% LMP     — hardest-pixel cross-entropy (paper §3.3)
      15% Tversky — FP suppression
      25% Dice    — mask connectivity
      30% Edge    — sharp cuticle boundaries

    Intermediate output (Laplacian pyramid, paper §3.2):
      40% weight × LMP at f0 resolution — forces low-res features to be meaningful

    Gate: 30% weight × BCE on presence prediction
    """
    def __init__(self, alpha=0.15, beta=0.85, inter_weight=0.4, gate_weight=0.3):
        super().__init__()
        self.lmp         = LMPLoss(p=0.1)
        self.tversky     = TverskyLoss(alpha=alpha, beta=beta)
        self.dice        = SoftDiceLoss()
        self.edge        = SobelEdgeLoss()
        self.inter_weight = inter_weight
        self.gate_weight  = gate_weight

    def forward(self, output, targets):
        # output  : (logits, logits_inter, gate)
        #   logits       : (B, 2, H, W)       — full-resolution output
        #   logits_inter : (B, 2, H/16, W/16) — intermediate Laplacian level
        #   gate         : (B, 1)             — presence probability
        # targets : (B, 1, H, W) float32
        logits, logits_inter, gate = output

        # ── Main segmentation loss ────────────────────────────────────────────
        l_lmp     = self.lmp(logits, targets)
        l_tversky = self.tversky(logits, targets)
        l_dice    = self.dice(logits, targets)
        l_edge    = self.edge(logits, targets)
        loss      = 0.30 * l_lmp + 0.15 * l_tversky + 0.25 * l_dice + 0.30 * l_edge

        # ── Intermediate (Laplacian pyramid) supervision ──────────────────────
        # Downsample GT to match f0 spatial size, apply LMP at that scale
        inter_h, inter_w = logits_inter.shape[2:]
        targets_ds = F.interpolate(targets, size=(inter_h, inter_w), mode="nearest")
        l_inter    = self.lmp(logits_inter, targets_ds)
        loss       = loss + self.inter_weight * l_inter

        # ── Gate BCE ──────────────────────────────────────────────────────────
        has_nail = (targets.sum(dim=(-1, -2, -3)) > 0).float()  # (B,)
        l_gate   = F.binary_cross_entropy(gate.squeeze(1), has_nail)
        loss     = loss + self.gate_weight * l_gate

        # ── Background-Only Penalty ───────────────────────────────────────────
        with torch.no_grad():
            is_empty_batch = targets.sum() == 0
        if is_empty_batch:
            loss = loss * 4.0

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
