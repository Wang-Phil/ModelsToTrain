from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

try:
    from torchvision.ops import FeaturePyramidNetwork
except Exception:  # pragma: no cover
    FeaturePyramidNetwork = None  # type: ignore


def _make_conv(in_c: int, out_c: int, k: int = 1, s: int = 1, p: int | None = None) -> nn.Sequential:
    if p is None:
        p = (k - 1) // 2
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, s, p, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class SimpleFPN(nn.Module):
    """Wrapper around torchvision.ops.FPN; falls back to naive upsample-sum if torchvision not available."""

    def __init__(self, in_channels_list: List[int], out_channels: int = 256) -> None:
        super().__init__()
        self.out_channels = out_channels
        if FeaturePyramidNetwork is not None:
            self.fpn = FeaturePyramidNetwork(in_channels_list=in_channels_list, out_channels=out_channels)
        else:
            self.lateral = nn.ModuleList([_make_conv(c, out_channels, 1) for c in in_channels_list])
            self.output = nn.ModuleList([_make_conv(out_channels, out_channels, 3) for _ in in_channels_list])

    def forward(self, feats: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        # feats: list from low->high resolution (C3, C4, C5 ...)
        if hasattr(self, "fpn"):
            inputs = {str(i): f for i, f in enumerate(feats)}
            return self.fpn(inputs)  # keys are "0","1",...
        # naive FPN
        laterals = [l(f) for l, f in zip(self.lateral, feats)]
        results = [None] * len(laterals)
        results[-1] = self.output[-1](laterals[-1])
        for i in range(len(laterals) - 2, -1, -1):
            up = F.interpolate(results[i + 1], size=laterals[i].shape[-2:], mode="nearest")
            results[i] = self.output[i](laterals[i] + up)
        return {str(i): t for i, t in enumerate(results)}


class BiFPN(nn.Module):
    """A light BiFPN: top-down then bottom-up bidirectional fusion with learnable weights."""

    def __init__(self, in_channels_list: List[int], out_channels: int = 160) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.proj = nn.ModuleList([_make_conv(c, out_channels, 1) for c in in_channels_list])
        self.td = nn.ModuleList([_make_conv(out_channels, out_channels, 3) for _ in in_channels_list])
        self.bu = nn.ModuleList([_make_conv(out_channels, out_channels, 3) for _ in in_channels_list])
        self.w1 = nn.Parameter(torch.ones(len(in_channels_list), 2))
        self.w2 = nn.Parameter(torch.ones(len(in_channels_list), 2))

    def _norm_weights(self, w: torch.Tensor) -> torch.Tensor:
        w = F.relu(w)
        return w / (w.sum(dim=1, keepdim=True) + 1e-6)

    def forward(self, feats: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        xs = [p(f) for p, f in zip(self.proj, feats)]  # low->high resolution
        # top-down
        w1 = self._norm_weights(self.w1)
        td = [None] * len(xs)
        td[-1] = self.td[-1](xs[-1])
        for i in range(len(xs) - 2, -1, -1):
            up = F.interpolate(td[i + 1], size=xs[i].shape[-2:], mode="nearest")
            td[i] = self.td[i](w1[i, 0] * xs[i] + w1[i, 1] * up)
        # bottom-up
        w2 = self._norm_weights(self.w2)
        bu = [None] * len(xs)
        bu[0] = self.bu[0](td[0])
        for i in range(1, len(xs)):
            down = F.max_pool2d(bu[i - 1], kernel_size=2)
            bu[i] = self.bu[i](w2[i, 0] * td[i] + w2[i, 1] * down)
        return {str(i): bu[i] for i in range(len(bu))}


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling block."""

    def __init__(self, in_channels: int, out_channels: int = 256, rates: List[int] | None = None) -> None:
        super().__init__()
        if rates is None:
            rates = [1, 6, 12, 18]
        self.convs = nn.ModuleList(
            [
                _make_conv(in_channels, out_channels, 1),
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=rates[1], dilation=rates[1], bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=rates[2], dilation=rates[2], bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=rates[3], dilation=rates[3], bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ),
            ]
        )
        self.project = _make_conv(out_channels * 4, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [m(x) for m in self.convs]
        y = torch.cat(feats, dim=1)
        return self.project(y)


class NeckClassifier(nn.Module):
    """Fuse multi-level features and classify.
    strategy='concat' or 'sum'."""

    def __init__(self, in_channels_list: List[int], neck: str = "fpn", out_channels: int = 256, num_classes: int = 2):
        super().__init__()
        self.neck_type = neck.lower()
        if self.neck_type == "fpn":
            self.neck = SimpleFPN(in_channels_list, out_channels)
            fuse_c = out_channels
        elif self.neck_type == "bifpn":
            self.neck = BiFPN(in_channels_list, out_channels)
            fuse_c = out_channels
        elif self.neck_type == "aspp":
            # Only use last feature for ASPP
            self.aspp = ASPP(in_channels_list[-1], out_channels)
            fuse_c = out_channels
        else:
            # no neck: project last feature
            self.proj = _make_conv(in_channels_list[-1], out_channels, 1)
            fuse_c = out_channels
        self.classifier = nn.Linear(fuse_c, num_classes)

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        if self.neck_type in ("fpn", "bifpn"):
            maps = self.neck(feats)  # dict of pyramid maps
            # global average pool each level then average
            pooled = []
            for k in sorted(maps.keys(), key=lambda x: int(x)):
                m = maps[k]
                pooled.append(F.adaptive_avg_pool2d(m, 1).flatten(1))
            x = torch.stack(pooled, dim=0).mean(dim=0)
        elif self.neck_type == "aspp":
            x = self.aspp(feats[-1])
            x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        else:
            x = self.proj(feats[-1])
            x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.classifier(x)


