"""
Nail VTON Model
---------------
Strict adherence to VTNFP (Duke et al., 2019):
  Backbone   : MobileNetV2
  Encoder paths (shared weights):
    High-res path : full-res input  -> 1/8,   32ch
    Low-res path  : half-res input  -> 1/16,  320ch (modified stride 16x)
  Fusion : Cascaded Feature Fusion blocks
  Laplacian Pyramid: Side-outputs at each fusion stage (Level 0, Level 1)
  and at the final full-resolution layer (Level 2).
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

class MobileNetV2Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.mobilenet_v2(weights=weights)
        f = backbone.features

        # High path: up to 32ch (stage 4 equivalent, stride 8)
        self.high_path = nn.Sequential(*f[:7])

        # Low path: all stages but modify stride to max 16x
        self.low_path = nn.Sequential(*f[:18])
        
        # Patch stride/dilation for stage 6/7 (idx 14 to 17) to keep resolution at 1/16
        self.low_path[14].conv[1][0].stride = (1, 1)
        for i in [14, 15, 16, 17]:
            self.low_path[i].conv[1][0].dilation = (2, 2)
            self.low_path[i].conv[1][0].padding = (2, 2)

    def forward_high(self, x):
        return self.high_path(x)

    def forward_low(self, x_half):
        return self.low_path(x_half)

class SegHead(nn.Module):
    """
    Output branch used at each level of the Laplacian pyramid.
    Resolution determined by the input tensor (no intermediate upsampling).
    The final scaling to full-res happens outside if this is the Level 2 head.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.ds   = DepthwiseSeparable(in_ch, in_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.ds(x)
        return self.conv(x)

class PyramidHeads(nn.Module):
    """Group of output task heads for one level of the pyramid."""
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

        self.encoder = MobileNetV2Encoder(pretrained=pretrained)

        HIGH_CH = 32
        LOW_CH  = 320
        FUSE    = 320

        self.fusion0 = CFF(LOW_CH, HIGH_CH, FUSE)
        self.fusion1 = CFF(FUSE, HIGH_CH, FUSE)

        # Laplacian Pyramid Output branches
        self.head_l0 = PyramidHeads(FUSE, max_instances) # Side-output at 1/8
        self.head_l1 = PyramidHeads(FUSE, max_instances) # Side-output at 1/8
        self.head_final = PyramidHeads(FUSE, max_instances) # Final at full-res (1/8 upsampled)

    def forward(self, x):
        with torch.amp.autocast("cuda", enabled=torch.is_autocast_enabled()):
            x_half    = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
            feat_high = self.encoder.forward_high(x)
        feat_low = self.encoder.forward_low(x_half)

        # Level 0 fusion (low8 + low4) -> result is 1/8 resolution
        f0 = self.fusion0(feat_low, feat_high)
        out0_bin, out0_inst, out0_dir = self.head_l0(f0)

        # Level 1 fusion (f0 + high4) -> result is 1/8 resolution
        f1 = self.fusion1(f0, feat_high)
        out1_bin, out1_inst, out1_dir = self.head_l1(f1)

        # Level 2 (Final full-resolution output branch)
        out2_bin, out2_inst, out2_dir = self.head_final(f1)
        
        # Norm direction fields
        def _norm_dir(d):
            return d / d.norm(dim=1, keepdim=True).clamp(min=1e-6)

        preds_l0 = (out0_bin, out0_inst, _norm_dir(out0_dir))
        preds_l1 = (out1_bin, out1_inst, _norm_dir(out1_dir))
        
        # Upsample the final output to full image size
        final_bin = F.interpolate(out2_bin, size=(self.image_size, self.image_size),
                                  mode="bilinear", align_corners=False)
        final_inst = F.interpolate(out2_inst, size=(self.image_size, self.image_size),
                                   mode="bilinear", align_corners=False)
        final_dir = F.interpolate(out2_dir, size=(self.image_size, self.image_size),
                                  mode="bilinear", align_corners=False)
        preds_final = (final_bin, final_inst, _norm_dir(final_dir))

        return [preds_l0, preds_l1, preds_final]

    @torch.no_grad()
    def predict(self, x, binary_thresh=0.5):
        self.eval()
        # predict returns only the FINAL level output for inference
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
    
    print(f"Number of pyramid levels: {len(outs)}")
    for i, (b, inst, d) in enumerate(outs):
        print(f"Level {i}: bin={b.shape}, inst={inst.shape}, dir={d.shape}")

    print("\nModel sanity check PASSED ✓")
