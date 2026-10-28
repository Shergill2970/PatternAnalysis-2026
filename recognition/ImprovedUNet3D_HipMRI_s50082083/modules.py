"""
Model components (PyTorch only, no NumPy, as required by the spec).

Contains:
    UNet         standard U-Net (Ronneberger et al., 2015), the 2D baseline
    DiceCELoss   soft Dice + cross-entropy loss for extreme class imbalance
    build_model  name -> model factory used by train.py and predict.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ops(dims):
    """Pick 2D or 3D layer types so the same code builds either network."""
    if dims == 2:
        return nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d, nn.BatchNorm2d, nn.InstanceNorm2d
    if dims == 3:
        return nn.Conv3d, nn.ConvTranspose3d, nn.MaxPool3d, nn.BatchNorm3d, nn.InstanceNorm3d
    raise ValueError(f"dims must be 2 or 3, got {dims}")


def _norm(dims, kind, channels):
    _, _, _, batch_norm, instance_norm = _ops(dims)
    if kind == "batch":
        return batch_norm(channels)
    if kind == "instance":
        return instance_norm(channels, affine=True)
    raise ValueError(f"norm must be 'batch' or 'instance', got {kind}")


class DoubleConv(nn.Module):
    """(3x3 conv -> norm -> ReLU) twice. Padding 1 keeps the spatial size."""

    def __init__(self, in_ch, out_ch, dims=2, norm="batch"):
        super().__init__()
        conv = _ops(dims)[0]
        self.block = nn.Sequential(
            conv(in_ch, out_ch, 3, padding=1, bias=False), _norm(dims, norm, out_ch), nn.ReLU(inplace=True),
            conv(out_ch, out_ch, 3, padding=1, bias=False), _norm(dims, norm, out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """Encoder step: halve the spatial size with 2x2 max-pooling, then DoubleConv."""

    def __init__(self, in_ch, out_ch, dims=2, norm="batch"):
        super().__init__()
        pool = _ops(dims)[2]
        self.block = nn.Sequential(pool(2), DoubleConv(in_ch, out_ch, dims, norm))

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """Decoder step: double the size with a transposed conv, concatenate the
    matching encoder feature map (skip connection), then DoubleConv."""

    def __init__(self, in_ch, out_ch, dims=2, norm="batch"):
        super().__init__()
        conv_t = _ops(dims)[1]
        self.up = conv_t(in_ch, out_ch, 2, stride=2)
        self.conv = DoubleConv(out_ch * 2, out_ch, dims, norm)

    def forward(self, x, skip):
        x = self.up(x)
        return self.conv(torch.cat([skip, x], dim=1))


class UNet(nn.Module):
    """U-Net: `depth` down steps, `depth` up steps with skip connections, then a
    1x1 conv giving one logit per class per pixel/voxel.

    Channels double at each level: base, 2*base, ..., base * 2**depth.
    Input height/width (and depth in 3D) must be divisible by 2**depth.
    """

    def __init__(self, in_channels=1, num_classes=6, base=32, depth=4, dims=2, norm="batch"):
        super().__init__()
        self.depth = depth
        chs = [base * 2 ** i for i in range(depth + 1)]
        self.inc = DoubleConv(in_channels, chs[0], dims, norm)
        self.downs = nn.ModuleList(Down(chs[i], chs[i + 1], dims, norm) for i in range(depth))
        self.ups = nn.ModuleList(Up(chs[i + 1], chs[i], dims, norm) for i in reversed(range(depth)))
        self.head = _ops(dims)[0](chs[0], num_classes, 1)

    def forward(self, x):
        factor = 2 ** self.depth
        assert all(s % factor == 0 for s in x.shape[2:]), (
            f"Spatial size {tuple(x.shape[2:])} must be divisible by {factor}")
        skips = []
        x = self.inc(x)
        for down in self.downs:
            skips.append(x)
            x = down(x)
        for up in self.ups:
            x = up(x, skips.pop())
        return self.head(x)


class DiceCELoss(nn.Module):
    """Cross-entropy + (1 - mean soft Dice over classes).

    Cross-entropy gives stable per-voxel gradients; the Dice term weights every
    class equally regardless of size, so the model cannot score well by
    ignoring the prostate (0.1% of voxels). Dice uses softmax probabilities
    (soft Dice) so it is differentiable.
    """

    def __init__(self, num_classes=6, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, target):
        logits = logits.float()                              # stable under mixed precision
        ce = F.cross_entropy(logits, target)
        probs = logits.softmax(dim=1)                        # softmax BEFORE Dice
        onehot = F.one_hot(target, self.num_classes).movedim(-1, 1).float()
        dims = (0,) + tuple(range(2, logits.ndim))           # sum over batch + space
        intersection = (probs * onehot).sum(dims)
        total = probs.sum(dims) + onehot.sum(dims)
        dice = (2 * intersection + self.smooth) / (total + self.smooth)
        return ce + (1 - dice.mean())


# name -> (constructor, "2d" or "3d" input mode). More models are added here later.
MODELS = {
    "unet2d": (lambda: UNet(dims=2, base=32, depth=4, norm="batch"), "2d"),
}


def build_model(name):
    """Return (model, mode) for a registered model name."""
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from {sorted(MODELS)}")
    make, mode = MODELS[name]
    return make(), mode
