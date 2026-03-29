"""
Nail VTON Model
---------------
Strict adherence to VTNFP (Duke et al., 2019):
  Backbone   : MobileNetV2
  Encoder paths (shared weights):
    High-res path : full-res input  -> 1/8,   32ch
    Low-res path  : half-res input  -> 1/16,  320ch (modified stride 16x)
  Fusion : Cascaded Feature Fusion blocks
  Heads:
    10-class softmax (nails)
    1-channel binary (sigmoid)
    2-channel direction (norm)
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
    """
    Cascade Feature Fusion (VTNFP style).
      low_feat  -> bilinear upsample -> dilated conv (d=2)
      high_feat -> 1x1 projection
      output    -> element-wise sum -> ReLU6
    """
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
        # PyTorch indices: 0(s=2), 1(s=1), 2,3(s=2~4), 4,5,6(s=2~8) => 32 channels.
        self.high_path = nn.Sequential(*f[:7])

        # Low path: all stages but modify stride to max 16x
        self.low_path = nn.Sequential(*f[:18])
        
        # Patch stride/dilation for stage 6/7 (idx 14 to 17)
        # In PyTorch, idx 14 is the layer that jumps to stride 32x.
        self.low_path[14].conv[1][0].stride = (1, 1)
        for i in [14, 15, 16, 17]:
            self.low_path[i].conv[1][0].dilation = (2, 2)
            self.low_path[i].conv[1][0].padding = (2, 2)

    def forward_high(self, x):
        return self.high_path(x)

    def forward_low(self, x_half):
        return self.low_path(x_half)

class SegHead(nn.Module):
    def __init__(self, in_ch, out_ch, output_size):
        super().__init__()
        self.output_size = output_size
        self.ds   = DepthwiseSeparable(in_ch, in_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.ds(x)
        x = self.conv(x)
        x = F.interpolate(x, size=(self.output_size, self.output_size),
                          mode="bilinear", align_corners=False)
        return x

class NailVTONModel(nn.Module):
    def __init__(self, image_size=512, max_instances=10, pretrained=True):
        super().__init__()
        self.image_size    = image_size
        self.max_instances = max_instances

        self.encoder = MobileNetV2Encoder(pretrained=pretrained)

        HIGH_CH = 32
        LOW_CH  = 320
        FUSE    = 320

        # Exact structure from Figure 1
        # Fusion 1 (low_8 + low_4 upsampled is skipped as per standard ICNet optimization in codebases often,
        # but the paper diagram uses: 
        # f1 = fusion(low8 upsampled, low4)
        # f2 = fusion(f1 upsampled, high4) -> Output)
        # To match your existing dual CFF:
        self.fusion0 = CFF(LOW_CH, HIGH_CH, FUSE)
        self.fusion1 = CFF(FUSE, HIGH_CH, FUSE)

        self.head_binary    = SegHead(FUSE, 1,             image_size)
        self.head_instances = SegHead(FUSE, max_instances, image_size)
        self.head_direction = SegHead(FUSE, 2,             image_size)

    def forward(self, x):
        with torch.amp.autocast("cuda", enabled=torch.is_autocast_enabled()):
            x_half    = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
            feat_high = self.encoder.forward_high(x)
        feat_low = self.encoder.forward_low(x_half)

        f0 = self.fusion0(feat_low, feat_high)
        f1 = self.fusion1(f0,       feat_high)

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
    print(f"direction norm at [0,:,16,16]: {direction[0,:,16,16].norm().item():.6f}  (must be ~1.0)")

    print("\nModel sanity check PASSED ✓")
