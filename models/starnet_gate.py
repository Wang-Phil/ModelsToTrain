"""
StarNet with Multiple Gating Options
    gate: 'none'   – original StarNet
          'intra'  – GLU/MambaOut-style gating inside star operator
          'pre'    – pre-star dual-branch gating
          'post'   – post-star gating
          'swiglu' – SwiGLU-style second branch gating

This version keeps the original StarNet structure and adds configurable gating.
"""

import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model


# ---------------------------------------------------------
# Basic Conv-BN block
# ---------------------------------------------------------
class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0,
                 dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(
            in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)


# ---------------------------------------------------------
# StarNet Block with multiple gating mechanisms
# ---------------------------------------------------------
class Block(nn.Module):
    """
    StarNet Block with configurable gating:
    - none   : original starnet (x = act(x1) * x2)
    - intra  : GLU/MambaOut gate on second branch x2 = a * sigmoid(b)
    - pre    : gate each branch before star
    - post   : gate star output
    - swiglu : SwiGLU style: x2 = act(a) * b
    """
    def __init__(self, dim, mlp_ratio=3, drop_path=0., gate='none'):
        super().__init__()

        self.gate = gate
        hidden = mlp_ratio * dim

        self.dwconv = ConvBN(dim, dim, 7, 1, 3, groups=dim, with_bn=True)

        # base star branches
        self.f1 = ConvBN(dim, hidden, 1, with_bn=False)

        # for f2, depends on gate type
        if gate in ['none', 'post']:
            # original f2
            self.f2 = ConvBN(dim, hidden, 1, with_bn=False)

        elif gate == 'intra':
            # GLU/MambaOut style x2 = a * sigmoid(b)
            self.f2_a = ConvBN(dim, hidden, 1, with_bn=False)
            self.f2_b = ConvBN(dim, hidden, 1, with_bn=False)
            nn.init.constant_(self.f2_b.conv.bias, 1.0)

        elif gate == 'pre':
            # pre-gate: f2 and gating conv
            self.f2 = ConvBN(dim, hidden, 1, with_bn=False)
            self.gf1 = ConvBN(dim, hidden, 1, with_bn=False)
            self.gf2 = ConvBN(dim, hidden, 1, with_bn=False)
            nn.init.constant_(self.gf1.conv.bias, 1.0)
            nn.init.constant_(self.gf2.conv.bias, 1.0)

        elif gate == 'swiglu':
            # SwiGLU: x2 = swish(a) * b
            self.f2_a = ConvBN(dim, hidden, 1, with_bn=False)
            self.f2_b = ConvBN(dim, hidden, 1, with_bn=False)

        # gate on star output
        if gate == 'post':
            self.gpost = ConvBN(hidden, hidden, 1, with_bn=False)
            nn.init.constant_(self.gpost.conv.bias, 1.0)

        self.g = ConvBN(hidden, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, 3, groups=dim, with_bn=False)

        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x

        x = self.dwconv(x)
        x1 = self.f1(x)

        # -----------------------------------------------------
        # Branch gating logic
        # -----------------------------------------------------
        if self.gate == 'none':
            # original starnet
            x2 = self.f2(x)

        elif self.gate == 'intra':
            # GLU / MambaOut style
            xa = self.f2_a(x)
            xb = self.f2_b(x)
            x2 = xa * torch.sigmoid(xb)

        elif self.gate == 'pre':
            # pre-star gate for each branch
            gate1 = torch.sigmoid(self.gf1(x))
            gate2 = torch.sigmoid(self.gf2(x))
            x1 = self.act(x1) * gate1
            x2 = self.f2(x) * gate2

        elif self.gate == 'post':
            x2 = self.f2(x)

        elif self.gate == 'swiglu':
            # SwiGLU variant
            xa = self.f2_a(x)
            xb = self.f2_b(x)
            x2 = nn.functional.silu(xa) * xb

        # -----------------------------------------------------
        # Star multiplication
        # -----------------------------------------------------
        out = self.act(x1) * x2

        # -----------------------------------------------------
        # Post-star gate
        # -----------------------------------------------------
        if self.gate == 'post':
            g = torch.sigmoid(self.gpost(out))
            out = out * g

        out = self.dwconv2(self.g(out))
        return shortcut + self.drop_path(out)


# ---------------------------------------------------------
# Full StarNet Backbone
# ---------------------------------------------------------
class StarNet(nn.Module):
    def __init__(self, base_dim=32, depths=[3,3,12,5], mlp_ratio=4,
                 drop_path_rate=0.0, num_classes=1000, gate='none'):
        super().__init__()

        self.num_classes = num_classes
        self.in_channel = 32

        # stem
        self.stem = nn.Sequential(
            ConvBN(3, self.in_channel, kernel_size=3, stride=2, padding=1),
            nn.ReLU6()
        )

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # stages
        self.stages = nn.ModuleList()
        cur = 0
        for i in range(len(depths)):
            embed_dim = base_dim * 2 ** i
            down = ConvBN(self.in_channel, embed_dim, 3, 2, 1)
            self.in_channel = embed_dim

            blocks = [
                Block(self.in_channel, mlp_ratio, dpr[cur + j], gate=gate)
                for j in range(depths[i])
            ]
            cur += depths[i]

            self.stages.append(nn.Sequential(down, *blocks))

        self.norm = nn.BatchNorm2d(self.in_channel)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.in_channel, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = torch.flatten(self.avgpool(self.norm(x)), 1)
        return self.head(x)


# ---------------------------------------------------------
# Factory Functions
# ---------------------------------------------------------
@register_model
def starnet_gated_s1(pretrained=False, gate='pre', **kwargs):
    return StarNet(base_dim=24, depths=[2,2,8,3], gate=gate, **kwargs)

@register_model
def starnet_gated_s2(pretrained=False, gate='none', **kwargs):
    return StarNet(base_dim=32, depths=[1,2,6,2], gate=gate, **kwargs)

@register_model
def starnet_gated_s3(pretrained=False, gate='none', **kwargs):
    return StarNet(base_dim=32, depths=[2,2,8,4], gate=gate, **kwargs)

@register_model
def starnet_gated_s4(pretrained=False, gate='none', **kwargs):
    return StarNet(base_dim=32, depths=[3,3,12,5], gate=gate, **kwargs)
