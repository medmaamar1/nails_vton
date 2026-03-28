"""
Nail VTON Model
---------------
Dual-path cascade (Duke et al.) with MobileNetV3-Small backbone:

  Backbone   : MobileNetV3-Small (pretrained ImageNet)
               Better than V2: SE blocks recalibrate channels per-layer,
               h-swish gives smoother gradients — both help nail/skin separation.

  Encoder paths (asymmetric depth, shared weights):
    High-res path : features[:4]  on full-res input  -> 1/8,   24ch
    Low-res path  : features[:9]  on half-res input  -> 1/16,  48ch
    (deep low-res path compensates for reduced spatial resolution)

  Fusion : 2x CFF (Cascade Feature Fusion) blocks
    Fusion0 : CFF(low=48ch,  high=24ch) -> 128ch  at 1/8
    Fusion1 : CFF(fused=128ch, high=24ch) -> 128ch  at 1/8

  Heads (all upsample to IMAGE_SIZE):
    head_binary    -> (B,  1, H, W)  raw logits       - sigmoid at inference
    head_instances -> (B, 10, H, W)  raw logits       - softmax at inference
    head_direction -> (B,  2, H, W)  normalised unit vectors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

MAX_INSTANCES = 10


# -- Building blocks -----------------------------------------------------------

class DepthwiseSeparable(nn.Module):
    """Lightweight depthwise-separable conv for decoder heads."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.dw  = nn.Conv2d(in_ch, in_ch,  3, padding=1, groups=in_ch, bias=False)
        self.pw  = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn  = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class CFF(nn.Module):
    """
    Cascade Feature Fusion (ICNet-style).
      low_feat  -> bilinear upsample to high_feat size -> dilated conv (d=2)
      high_feat -> 1x1 projection
      output    -> element-wise sum -> ReLU6
    """
    def __init__(self, low_ch, high_ch, out_ch):
        super().__init__()
        self.low_conv = nn.Sequential(
            nn.Conv2d(low_ch,  out_ch, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.high_conv = nn.Sequential(
            nn.Conv2d(high_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.act = nn.ReLU6(inplace=True)

    def forward(self, low_feat, high_feat):
        low_up = F.interpolate(low_feat, size=high_feat.shape[2:],
                               mode="bilinear", align_corners=False)
        return self.act(self.low_conv(low_up) + self.high_conv(high_feat))


class SegHead(nn.Module):
    """
    Decoder head:
      depthwise-separable conv -> bilinear upsample to output_size -> 1x1 conv
    """
    def __init__(self, in_ch, out_ch, output_size):
        super().__init__()
        self.output_size = output_size
        self.ds   = DepthwiseSeparable(in_ch, in_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.ds(x)
        x = F.interpolate(x, size=(self.output_size, self.output_size),
                          mode="bilinear", align_corners=False)
        return self.conv(x)


# -- MobileNetV3-Small encoder -------------------------------------------------

class MobileNetV3Encoder(nn.Module):
    """
    Two asymmetric prefix sub-networks sharing weights:

      high_path = features[:4]  shallow, run on full-res  -> 1/8,  24ch
      low_path  = features[:9]  deep,    run on half-res  -> 1/16, 48ch
    """
    def __init__(self, pretrained=True):
        super().__init__()
        backbone = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            if pretrained else None
        )
        f = backbone.features

        self.high_path = nn.Sequential(*f[:4])   # -> 1/8,  24ch
        self.low_path  = nn.Sequential(*f[:9])   # -> 1/16, 48ch

    def forward_high(self, x):
        return self.high_path(x)

    def forward_low(self, x_half):
        return self.low_path(x_half)


# -- Main model ----------------------------------------------------------------

class NailVTONModel(nn.Module):
    def __init__(self, image_size=512, max_instances=MAX_INSTANCES, pretrained=True):
        super().__init__()
        self.image_size    = image_size
        self.max_instances = max_instances

        self.encoder  = MobileNetV3Encoder(pretrained=pretrained)

        HIGH_CH = 24
        LOW_CH  = 48
        FUSE    = 128

        self.fusion0 = CFF(LOW_CH,  HIGH_CH, FUSE)
        self.fusion1 = CFF(FUSE,    HIGH_CH, FUSE)

        self.head_binary    = SegHead(FUSE, 1,             image_size)
        self.head_instances = SegHead(FUSE, max_instances, image_size)
        self.head_direction = SegHead(FUSE, 2,             image_size)

    def forward(self, x):
        x_half    = F.interpolate(x, scale_factor=0.5, mode="bilinear",
                                  align_corners=False)
        feat_high = self.encoder.forward_high(x)
        feat_low  = self.encoder.forward_low(x_half)

        f0 = self.fusion0(feat_low,  feat_high)
        f1 = self.fusion1(f0,        feat_high)

        binary    = self.head_binary(f1)
        instances = self.head_instances(f1)
        direction = self.head_direction(f1)
        direction = direction / direction.norm(dim=1, keepdim=True).clamp(min=1e-6)

        return binary, instances, direction

    @torch.no_grad()
    def predict(self, x, binary_thresh=0.5):
        self.eval()
        binary_logits, inst_logits, direction = self(x)
        return (
            torch.sigmoid(binary_logits) > binary_thresh,
            torch.softmax(inst_logits, dim=1),
            direction,
        )

    def count_parameters(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Parameters -- total: {total:,}  trainable: {trainable:,}")
        return total, trainable


# -- Sanity check --------------------------------------------------------------

if __name__ == "__main__":
    model = NailVTONModel(image_size=512, pretrained=False)
    model.count_parameters()

    dummy = torch.randn(1, 3, 64, 64)
    binary, instances, direction = model(dummy)

    print(f"binary    : {binary.shape}")
    print(f"instances : {instances.shape}")
    probs = torch.softmax(instances, dim=1)
    print(f"softmax sum at [0,:,16,16] : {probs[0,:,16,16].sum().item():.6f}  (must be 1.0)")
    print(f"direction : {direction.shape}")
    print("\nModel sanity check PASSED ✓")