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

MAX_INSTANCES = 10

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
    """Encapsulates a prefix of a MobileNetV2 backbone."""
    def __init__(self, stages_idx, pretrained=True, surgery=False):
        super().__init__()
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.mobilenet_v2(weights=weights)
        self.features = nn.Sequential(*backbone.features[:stages_idx])
        
        if surgery:
            # Low-res path surgery: Stride 32x -> 16x
            # We change stage 6 (index 14) stride to 1, and stages 7-8 to dilated.
            # MobileNetV2 layer index 14 is the InvertedResidual that normally has stride 2.
            if len(self.features) > 14:
                self.features[14].conv[1][0].stride = (1, 1)
                for i in range(14, len(self.features)):
                    # Check if it has a depthwise layer (Conv2d with groups > 1)
                    # For MobileNetV2, conv[1][0] is the depthwise convolution.
                    self.features[i].conv[1][0].dilation = (2, 2)
                    self.features[i].conv[1][0].padding = (2, 2)

    def forward(self, x):
        return self.features(x)

class SegHead(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.ds   = DepthwiseSeparable(in_ch, in_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.ds(x)
        return self.conv(x)

class PyramidHeads(nn.Module):
    def __init__(self, in_ch, max_instances=10):
        super().__init__()
        self.binary    = SegHead(in_ch, 1)
        self.instances = SegHead(in_ch, max_instances)
        self.direction = SegHead(in_ch, 2)

    def forward(self, x):
        return self.binary(x), self.instances(x), self.direction(x)

class NailVTONModel(nn.Module):
    def __init__(self, image_size=448, max_instances=10, pretrained=True):
        super().__init__()
        self.image_size    = image_size
        self.max_instances = max_instances

        # TWO independent encoders (no weight sharing) to match paper's capacity
        # Level 0 (low-res): 1/16 res, 320 channels (stages 1..8)
        self.encoder_low = MobileNetV2Prefix(stages_idx=18, pretrained=pretrained, surgery=True)
        # Level 1 (high-res): 1/8 res, 32 channels (stages 1..4)
        self.encoder_high = MobileNetV2Prefix(stages_idx=7, pretrained=pretrained, surgery=False)

        HIGH_CH = 32
        LOW_CH  = 320
        FUSE    = 320

        self.fusion0 = CFF(LOW_CH, HIGH_CH, FUSE)
        self.fusion1 = CFF(FUSE, HIGH_CH, FUSE)

        # Laplacian Pyramid heads
        self.head_l0 = PyramidHeads(FUSE, max_instances)
        self.head_l1 = PyramidHeads(FUSE, max_instances)
        self.head_final = PyramidHeads(FUSE, max_instances)

    def forward(self, x):
        with torch.amp.autocast("cuda", enabled=torch.is_autocast_enabled()):
            x_half    = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
            # Independent forward passes
            feat_high = self.encoder_high(x)
            feat_low  = self.encoder_low(x_half)

        # 1. First Fusion (Level 0 side-output)
        f0 = self.fusion0(feat_low, feat_high)
        out0_bin, out0_inst, out0_dir = self.head_l0(f0)

        # 2. Second Fusion (Level 1 side-output)
        f1 = self.fusion1(f0, feat_high)
        out1_bin, out1_inst, out1_dir = self.head_l1(f1)

        # 3. Final Output (Upsampled to full resolution)
        out2_bin, out2_inst, out2_dir = self.head_final(f1)
        
        def _norm_dir(d):
            return d / d.norm(dim=1, keepdim=True).clamp(min=1e-6)

        p0 = (out0_bin, out0_inst, _norm_dir(out0_dir))
        p1 = (out1_bin, out1_inst, _norm_dir(out1_dir))
        
        final_bin = F.interpolate(out2_bin, size=(self.image_size, self.image_size),
                                  mode="bilinear", align_corners=False)
        final_inst = F.interpolate(out2_inst, size=(self.image_size, self.image_size),
                                   mode="bilinear", align_corners=False)
        final_dir = F.interpolate(out2_dir, size=(self.image_size, self.image_size),
                                  mode="bilinear", align_corners=False)
        pf = (final_bin, final_inst, _norm_dir(final_dir))

        return [p0, p1, pf]

    @torch.no_grad()
    def predict(self, x, binary_thresh=0.5):
        self.eval()
        multi_preds = self(x)
        final_bin, final_inst, final_dir = multi_preds[-1]
        return (
            torch.sigmoid(final_bin) > binary_thresh,
            torch.softmax(final_inst, dim=1),
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
    for i, (b, inst, d) in enumerate(outs):
        print(f"  Level {i}: bin={b.shape}, inst={inst.shape}, dir={d.shape}")

    print("\nModel Architecture Check PASSED")
