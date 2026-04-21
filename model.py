"""
Nail VTON Model
---------------
Backbone : MobileNetV3-Large (two independent prefix encoders).
Task     : Binary segmentation ONLY. Direction head removed.

Output: (logits, logits_inter, gate)
  logits       : (B, 2, H, W)       — full-res nail/background logits
  logits_inter : (B, 2, H/16, W/16) — intermediate Laplacian level
  gate         : (B, 1)             — nail presence probability

Channel map (verified on (1,3,224,224) input):
  encoder_high : features[0:4]  -> 24ch  @ H/8  of original input
  encoder_low  : features[0:7]  -> 40ch  @ H/16 of half-res input (feat_s4)
                 features[7:13] -> 112ch @ H/16 of half-res (feat_s8, surgery)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ── Building blocks ────────────────────────────────────────────────────────────

class DepthwiseSeparable(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.dw  = nn.Conv2d(in_ch, in_ch,  3, padding=1, groups=in_ch, bias=False)
        self.pw  = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn  = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class CFF(nn.Module):
    """Cross-scale Feature Fusion."""
    def __init__(self, low_ch, high_ch, out_ch=320):
        super().__init__()
        self.low_conv  = nn.Sequential(nn.Conv2d(low_ch,  out_ch, 3, padding=2, dilation=2, bias=False), nn.BatchNorm2d(out_ch))
        self.high_conv = nn.Sequential(nn.Conv2d(high_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
        self.act       = nn.ReLU6(inplace=True)

    def forward(self, low_feat, high_feat):
        low_up = F.interpolate(low_feat, size=high_feat.shape[2:], mode="bilinear", align_corners=False)
        return self.act(self.low_conv(low_up) + self.high_conv(high_feat))


class SegHead(nn.Module):
    """Single-task binary segmentation head. Direction head removed."""
    def __init__(self, in_ch):
        super().__init__()
        self.shared = DepthwiseSeparable(in_ch, 10)
        self.binary = nn.Conv2d(10, 2, 1)  # 2 classes: background, nail

    def forward(self, x):
        return self.binary(self.shared(x))  # (B, 2, H, W)


class PresenceGate(nn.Module):
    """Global binary classifier: 'are there any nails in this image?'
    Learned gate that zeroes out the segmentation when no nails are present."""
    def __init__(self, in_ch):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(in_ch, 1)

    def forward(self, x):
        return self.fc(self.pool(x).flatten(1))  # (B, 1) — raw logits, sigmoid applied at loss/inference


# ── MobileNetV3-Large prefix encoder ──────────────────────────────────────────

class MobileNetV3LargePrefix(nn.Module):
    def __init__(self, end_idx, split_idx=None, pretrained=True, surgery=False):
        super().__init__()
        self.split_idx = split_idx
        weights        = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        backbone       = models.mobilenet_v3_large(weights=weights)
        self.features  = nn.Sequential(*backbone.features[:end_idx])

        if surgery:
            # Patch only the deep segment (features[split_idx:]) so the shallow
            # branch keeps its correct strides for feat_s4. Without this restriction
            # the old code accidentally patched features[2] (first stride-2 dw conv
            # found) instead of features[7] (the one we actually want to dilate).
            target = self.features[split_idx:] if split_idx is not None else self.features
            for m in target.modules():
                if (isinstance(m, nn.Conv2d)
                        and m.groups == m.in_channels
                        and m.stride == (2, 2)):
                    m.stride = (1, 1)
                    m.dilation = (2, 2)
                    m.padding = (2, 2)

    def forward(self, x):
        if self.split_idx is None:
            return self.features(x)
        feat_s4    = self.features[:self.split_idx](x)
        feat_final = self.features[self.split_idx:](feat_s4)
        return feat_s4, feat_final


# ── Main model ─────────────────────────────────────────────────────────────────

class NailVTONModel(nn.Module):
    def __init__(self, image_size=448, pretrained=True):
        super().__init__()
        self.image_size = image_size

        HIGH_CH   = 24
        LOW_S4_CH = 40
        LOW_S8_CH = 112
        FUSE      = 320

        self.encoder_high = MobileNetV3LargePrefix(end_idx=4,  split_idx=None, pretrained=pretrained, surgery=False)
        self.encoder_low  = MobileNetV3LargePrefix(end_idx=13, split_idx=7,    pretrained=pretrained, surgery=True)

        self.fusion_low   = CFF(LOW_S8_CH, LOW_S4_CH, FUSE)
        self.fusion_high  = CFF(FUSE,      HIGH_CH,   FUSE)

        self.head       = SegHead(FUSE)   # final output head (full-res)
        self.head_inter = SegHead(FUSE)   # Laplacian pyramid: intermediate head at f0 scale
        self.gate       = PresenceGate(FUSE)

    def forward(self, x):
        with torch.amp.autocast("cuda", enabled=torch.is_autocast_enabled()):
            x_half                   = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
            feat_high                = self.encoder_high(x)
            feat_low_s4, feat_low_s8 = self.encoder_low(x_half)

            f0 = self.fusion_low(feat_low_s8, feat_low_s4)
            f1 = self.fusion_high(f0, feat_high)

            # Laplacian pyramid: intermediate logits at f0 resolution (H/16 of input)
            logits_inter = self.head_inter(f0)   # (B, 2, H/16, W/16)
            logits       = self.head(f1)          # (B, 2, H/8,  W/8)
            gate         = self.gate(f1)          # (B, 1)

        out = F.interpolate(logits, size=(self.image_size, self.image_size),
                            mode="bilinear", align_corners=False)
        return out, logits_inter, gate  # (B,2,H,W), (B,2,H/16,W/16), (B,1)

    @torch.no_grad()
    def predict(self, x, threshold=0.5, gate_threshold=0.3):
        """Returns binary mask (B, 1, H, W). Gate suppresses output when no nails detected."""
        self.eval()
        logits, _logits_inter, gate = self(x)
        prob_nail = torch.softmax(logits, dim=1)[:, 1:2]
        gate_mask = (torch.sigmoid(gate) > gate_threshold).float().unsqueeze(-1).unsqueeze(-1)
        return (prob_nail * gate_mask > threshold).float()

    def count_parameters(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Parameters — total: {total:,}  trainable: {trainable:,}")
        return total, trainable


# ── Sanity check ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = NailVTONModel(image_size=448, pretrained=False)
    model.count_parameters()
    out, inter, gate = model(torch.randn(1, 3, 448, 448))
    print(f"Output shape : {out.shape}")    # Expected: (1, 2, 448, 448)
    print(f"Inter shape  : {inter.shape}")  # Expected: (1, 2, 28,  28)
    print(f"Gate shape   : {gate.shape}")   # Expected: (1, 1)
    assert out.shape   == (1, 2, 448, 448), "Shape mismatch!"
    assert inter.shape[1] == 2,             "Inter channels mismatch!"
    assert gate.shape  == (1, 1),           "Gate shape mismatch!"
    print("Model Architecture Check PASSED ✓")
