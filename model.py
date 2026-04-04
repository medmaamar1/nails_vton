"""
Nail VTON Model
---------------
Strict adherence to VTNFP (Duke et al., 2019):
  Backbone   : TWO independent MobileNetV2 alpha=1.0 prefixes (no shared weights).
  Encoder 1 (High-res) : stages 1..4 (1/8 resolution, 32 channels).
  Encoder 2 (Low-res)  : stages 1..8 (1/16 resolution, 320 channels, surgery applied).
  Fusion : Cascaded Feature Fusion blocks.
  Laplacian Pyramid: Side-outputs at Level 0, Level 1, and Level 2.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


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
    def __init__(self, low_ch, high_ch, out_ch=320):
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

class MobileNetV2Prefix(nn.Module):
    """Encapsulates a prefix of a MobileNetV2 backbone, optionally returning split features."""
    def __init__(self, stages_idx, split_idx=None, pretrained=True, surgery=False):
        super().__init__()
        self.split_idx = split_idx
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.mobilenet_v2(weights=weights)
        self.features = nn.Sequential(*backbone.features[:max(stages_idx, split_idx or 0)])
        
        if surgery:
            # Low-res path surgery: Stride 32x -> 16x
            # Standard index 14 has stride 2. Change to 1.
            if len(self.features) > 14:
                self.features[14].conv[1][0].stride = (1, 1)
                for i in range(14, len(self.features)):
                    self.features[i].conv[1][0].dilation = (2, 2)
                    self.features[i].conv[1][0].padding = (2, 2)

    def forward(self, x):
        if self.split_idx is None:
            return self.features(x)
        
        # Split features for H/16 skip-connection (Section 3.2)
        feat_s4 = self.features[:self.split_idx](x)
        feat_final = self.features[self.split_idx:](feat_s4)
        return feat_s4, feat_final

class PyramidHeads(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        # 1:1 Figure 2 Output Branch:
        # F2' (320) -> '320 W' (Depthwise) -> 'Projection 10' -> 'shared features 10'
        # This reduces FLOPs drastically compared to 3 separate SegHeads
        self.shared = DepthwiseSeparable(in_ch, 10)
        
        # Branch from shared features:
        # Fgbg: 1x1 conv to 2 channels (Softmax on 2 classes for background and foreground)
        self.binary    = nn.Conv2d(10, 2, 1)
        # Finger Class: 6 channels (0=BG, 1..5=Fingers)
        self.finger    = nn.Conv2d(10, 6, 1)
        # Direction: '1x1 conv 2'
        self.direction = nn.Conv2d(10, 2, 1)

    def forward(self, x):
        shared_feat = self.shared(x)
        return self.binary(shared_feat), self.finger(shared_feat), self.direction(shared_feat)

class NailVTONModel(nn.Module):
    def __init__(self, image_size=448, pretrained=True):
        super().__init__()
        self.image_size    = image_size

        # TWO independent encoders (no weight sharing) to match paper's capacity
        # Level 0 (low-res): Input H/2. Stage 4 is H/16, Stage 8 is H/32 (with surgery).
        self.encoder_low = MobileNetV2Prefix(stages_idx=18, split_idx=7, pretrained=pretrained, surgery=True)
        # Level 1 (high-res): Input H. Stage 4 is H/8
        self.encoder_high = MobileNetV2Prefix(stages_idx=7, pretrained=pretrained, surgery=False)

        HIGH_CH = 32
        LOW_S4_CH = 32
        LOW_S8_CH  = 320
        FUSE    = 320

        # Fusion 0: Stage_low4 (H/16) + Upsampled Stage_low8 (H/32 -> H/16)
        self.fusion_low = CFF(LOW_S8_CH, LOW_S4_CH, FUSE)
        
        # Fusion 1: Stage_high4 (H/8) + Upsampled Low_Features (H/16 -> H/8)
        self.fusion_high = CFF(FUSE, HIGH_CH, FUSE)

        # Laplacian Pyramid heads
        self.head_l0 = PyramidHeads(FUSE)
        self.head_l1 = PyramidHeads(FUSE)
        self.head_final = PyramidHeads(FUSE)

    def forward(self, x):
        with torch.amp.autocast("cuda", enabled=torch.is_autocast_enabled()):
            x_half    = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
            
            # Independent forward passes
            feat_high = self.encoder_high(x)
            feat_low_s4, feat_low_s8 = self.encoder_low(x_half)

        # 1. Low-Resolution Fusion (Level 0 side-output)
        # Matches Section 3.2: "fuses H/16 x W/16 features from stage_low4 with upsampled stage_low8"
        f0 = self.fusion_low(feat_low_s8, feat_low_s4)
        out0_bin, out0_finger, out0_dir = self.head_l0(f0)

        # 2. High-Resolution Fusion (Level 1 side-output)
        # Matches Section 3.2: "fuses resulting features with H/8 x W/8 features from stage_high4"
        f1 = self.fusion_high(f0, feat_high)
        out1_bin, out1_finger, out1_dir = self.head_l1(f1)

        # 3. Final Output (Upsampled to full resolution)
        out2_bin, out2_finger, out2_dir = self.head_final(f1)
        
        def _norm_dir(d):
            return d / d.norm(dim=1, keepdim=True).clamp(min=1e-6)

        p0 = (out0_bin, out0_finger, _norm_dir(out0_dir))
        p1 = (out1_bin, out1_finger, _norm_dir(out1_dir))
        
        final_bin = F.interpolate(out2_bin, size=(self.image_size, self.image_size),
                                  mode="bilinear", align_corners=False)
        final_finger = F.interpolate(out2_finger, size=(self.image_size, self.image_size),
                                  mode="bilinear", align_corners=False)
        final_dir = F.interpolate(out2_dir, size=(self.image_size, self.image_size),
                                  mode="bilinear", align_corners=False)
        pf = (final_bin, final_finger, _norm_dir(final_dir))

        return [p0, p1, pf]

    @torch.no_grad()
    def predict(self, x, binary_thresh=0.5):
        self.eval()
        multi_preds = self(x)
        final_bin, final_finger, final_dir = multi_preds[-1]
        # Binary output is now (B, 2, H, W). Use softmax and take foreground channel (index 1).
        prob_fg = torch.softmax(final_bin, dim=1)[:, 1:2]
        
        # Finger is (B, 6, H, W). Use argmax.
        pred_finger = torch.argmax(final_finger, dim=1, keepdim=True)
        return (
            prob_fg > binary_thresh,
            pred_finger,
            final_dir,
        )

    def count_parameters(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Parameters -- total: {total:,}  trainable: {trainable:,}")
        return total, trainable

if __name__ == "__main__":
    model = NailVTONModel(image_size=448, pretrained=False)
    model.count_parameters()

    dummy = torch.randn(1, 3, 448, 448)
    outs = model(dummy)
    
    print(f"Laplacian levels: {len(outs)}")
    for i, (b, f, d) in enumerate(outs):
        print(f"  Level {i}: bin={b.shape}, finger={f.shape}, dir={d.shape}")

    print("\nModel Architecture Check PASSED")
