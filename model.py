"""
Nail VTON Strong Model
-----------------------
Architecture : DeepLabV3+ (segmentation_models_pytorch)
Encoder      : ResNet-101, ImageNet pre-trained weights
Task         : Binary segmentation — Nail vs. Background
Output       : (B, 1, H, W) raw logits.  Apply sigmoid for probability.

The model is fully convolutional and accepts any spatial resolution at
inference — training at 640×640 does not restrict production input size.

Usage:
    from model import build_model, count_parameters
    model  = build_model()
    logits = model(x)          # (B, 1, H, W)
    probs  = torch.sigmoid(logits)
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


# ── Factory ────────────────────────────────────────────────────────────────────

def build_model(image_size: int = 640) -> nn.Module:
    """
    Build DeepLabV3+ with ResNet-101 encoder.

    Args:
        image_size: Training resolution (stored as metadata only).
                    The model is fully convolutional and will accept any
                    spatial resolution at inference.
    Returns:
        nn.Module — raw logits (B, 1, H, W)
    """
    model = smp.DeepLabV3Plus(
        encoder_name    = "resnet101",
        encoder_weights = "imagenet",   # ImageNet pre-trained encoder
        in_channels     = 3,
        classes         = 1,            # Binary: one logit per pixel
        activation      = None,         # Raw logits — losses handle activation
    )
    model.image_size = image_size
    return model


# ── Parameter count ────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> tuple:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters — total: {total:,}  trainable: {trainable:,}")
    return total, trainable


# ── Sanity check ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    m = build_model(image_size=640)
    count_parameters(m)

    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        out = m(x)

    print(f"Output shape : {out.shape}")   # Expected: (1, 1, 640, 640)
    assert out.shape == (1, 1, 640, 640), f"Shape mismatch: {out.shape}"
    print("Model Architecture Check PASSED ✓")
