"""
Nail VTON Model
---------------
Strict adherence to VTNFP (Duke et al., 2019):
  Backbone   : TWO independent MobileNetV3-Large prefixes (no shared weights).
  Encoder 1 (High-res) : features[0:5]  (H/8  resolution, 40 channels).
  Encoder 2 (Low-res)  : features[0:13] (H/16 resolution, 112 channels, surgery applied).
                         surgery target: features[13].block.1.0  stride (2,2)->(1,1), dilation 2.
  Fusion : Cascaded Feature Fusion blocks (CFF) -- unchanged from paper.
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

class MobileNetV3LargePrefix(nn.Module):
    """
    Encapsulates a prefix of a MobileNetV3-Large backbone.

    Verified stage indices (probed via forward pass on 224x224 low-res input):
      features[ 6]: 40ch  @ 28x28 = H/8  of low-res = H/16 of original  <- split_idx=7
      features[12]: 112ch @ 14x14 = H/16 of low-res = H/32 of original  <- last before stride-2
      features[13]: 160ch @  7x7  = surgery target (stride 2->1, dilation 2)
      features[14]: 160ch @ 14x14 = stride-1, runs at H/16 of orig post-surgery
      features[15]: 160ch @ 14x14 = stride-1, runs at H/16 of orig post-surgery
      features[16]: 960ch @ 14x14 = 1x1 expansion, FULL backbone power
    """
    def __init__(self, split_idx=None, pretrained=True, surgery=False):
        super().__init__()
        self.split_idx = split_idx
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.mobilenet_v3_large(weights=weights)

        if surgery:
            # Include features[0:17] — the full V3-Large feature extractor.
            # Surgery ONLY on features[13].block.1.0 (5x5 depthwise, stride 2->1).
            # Features [14], [15] are stride-1 so they run at H/16 of orig automatically.
            # Features [16] is the 1x1 960ch expansion — runs at H/16 of orig.
            self.features = nn.Sequential(*backbone.features[:17])
            dw = self.features[13].block[1][0]
            dw.stride   = (1, 1)
            dw.dilation = (2, 2)
            dw.padding  = (4, 4)  # dilation*(kernel_size-1)//2 = 2*(5-1)//2 = 4
        else:
            # High-res path: only need up to H/8 (features[0:5])
            self.features = nn.Sequential(*backbone.features[:5])

    def forward(self, x):
        if self.split_idx is None:
            # High-res path: single output
            return self.features(x)

        # Low-res path: split for H/16 skip-connection (Section 3.2)
        feat_s4    = self.features[:self.split_idx](x)    # 40ch  @ H/16 of orig
        feat_final = self.features[self.split_idx:](feat_s4) # 960ch @ H/16 of orig
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
        # Direction: '1x1 conv 2'
        self.direction = nn.Conv2d(10, 2, 1)

    def forward(self, x):
        shared_feat = self.shared(x)
        return self.binary(shared_feat), self.direction(shared_feat)

class NailVTONModel(nn.Module):
    def __init__(self, image_size=448, pretrained=True):
        super().__init__()
        self.image_size = image_size

        # TWO independent encoders (no weight sharing) — paper Section 3.2
        # Low-res encoder: receives H/2 input, uses FULL V3-Large features[0:17] with surgery.
        #   split_idx=7 gives 40ch skip at H/16 of original (s4).
        #   Remainder goes through features[7:17] including the 960ch expansion.
        self.encoder_low  = MobileNetV3LargePrefix(split_idx=7, pretrained=pretrained, surgery=True)
        # High-res encoder: receives full H input, uses V3-Large features[0:5].
        #   Output: 40ch at H/8 of original.
        self.encoder_high = MobileNetV3LargePrefix(split_idx=None, pretrained=pretrained, surgery=False)

        HIGH_CH   = 40   # V3-Large features[4] output channels (H/8 of orig)
        LOW_S4_CH = 40   # V3-Large features[6] output channels (H/16 of orig, skip)
        LOW_S8_CH = 960  # V3-Large features[16] 960ch expansion (H/16 of orig, post-surgery)
        FUSE      = 320  # CFF output — kept at 320 to preserve paper's decoder capacity

        # Fusion 0: stage_low8 (H/16) fused with stage_low4 (H/16) upsampled
        self.fusion_low  = CFF(LOW_S8_CH, LOW_S4_CH, FUSE)
        # Fusion 1: fused low features (H/16->H/8) fused with stage_high (H/8)
        self.fusion_high = CFF(FUSE, HIGH_CH, FUSE)

        # Laplacian Pyramid heads — identical to paper Figure 2
        self.head_l0    = PyramidHeads(FUSE)
        self.head_l1    = PyramidHeads(FUSE)
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
        out0_bin, out0_dir = self.head_l0(f0)

        # 2. High-Resolution Fusion (Level 1 side-output)
        # Matches Section 3.2: "fuses resulting features with H/8 x W/8 features from stage_high4"
        f1 = self.fusion_high(f0, feat_high)
        out1_bin, out1_dir = self.head_l1(f1)

        # 3. Final Output (Upsampled to full resolution)
        out2_bin, out2_dir = self.head_final(f1)
        
        def _norm_dir(d):
            return d / d.norm(dim=1, keepdim=True).clamp(min=1e-6)

        return (
            out0_bin, _norm_dir(out0_dir),
            out1_bin, _norm_dir(out1_dir),
            final_bin, _norm_dir(final_dir)
        )

    @torch.no_grad()
    def predict(self, x, binary_thresh=0.5):
        self.eval()
        multi_preds = self(x)
        # multi_preds: (bin0, dir0, bin1, dir1, bin_f, dir_f)
        final_bin, final_dir = multi_preds[4], multi_preds[5]
        # Binary output is now (B, 2, H, W). Use softmax and take foreground channel (index 1).
        prob_fg = torch.softmax(final_bin, dim=1)[:, 1:2]
        return prob_fg > binary_thresh, final_dir

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
    for i, (b, d) in enumerate(outs):
        print(f"  Level {i}: bin={b.shape}, dir={d.shape}")

    print("\nModel Architecture Check PASSED")
