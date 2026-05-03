import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model


STARNET_PRETRAINED_URLS = {
    "starnet_s1": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s1.pth.tar",
    "starnet_s2": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s2.pth.tar",
    "starnet_s3": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s3.pth.tar",
    "starnet_s4": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s4.pth.tar",
}


def _load_pretrained_starnet(model: nn.Module, model_name: str, num_classes: int) -> nn.Module:
    if model_name not in STARNET_PRETRAINED_URLS:
        return model
    url = STARNET_PRETRAINED_URLS[model_name]
    try:
        checkpoint = torch.hub.load_state_dict_from_url(
            url, map_location="cpu", file_name=f"{model_name}.pth.tar"
        )
        state_dict = checkpoint.get("state_dict", checkpoint)
        if not isinstance(state_dict, dict):
            return model
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model_state = model.state_dict()
        loaded = {k: v for k, v in state_dict.items() if k in model_state and model_state[k].shape == v.shape}
        if loaded:
            model_state.update(loaded)
            model.load_state_dict(model_state, strict=False)
            print(
                f"[CASGNet] loaded {model_name} pretrained params: {len(loaded)} "
                f"(head re-init for num_classes={num_classes})"
            )
    except Exception as e:
        print(f"[CASGNet] failed loading pretrained from {url}: {e}")
    return model


class ConvBN(nn.Sequential):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        with_bn=True,
    ):
        super().__init__()
        self.add_module(
            "conv",
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                dilation,
                groups,
                bias=not with_bn,
            ),
        )
        if with_bn:
            self.add_module("bn", nn.BatchNorm2d(out_planes))
            nn.init.constant_(self.bn.weight, 1)
            nn.init.constant_(self.bn.bias, 0)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * scale


class GRN(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = x.pow(2).mean(dim=[2, 3], keepdim=True)
        rx = torch.sqrt(gx + self.eps)
        norm_x = x / rx
        return self.gamma * norm_x + self.beta + x


class SelectiveKernel(nn.Module):
    def __init__(self, dim: int, kernel_sizes=(3, 7), reduction: int = 4):
        super().__init__()
        self.dim = dim
        self.num_branches = len(kernel_sizes)
        self.branches = nn.ModuleList()
        for ks in kernel_sizes:
            if ks == 1:
                branch = nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=1, groups=dim, bias=False),
                    nn.BatchNorm2d(dim),
                    nn.ReLU6(inplace=True),
                )
            else:
                dilation = 1 if ks == 3 else (ks - 1) // 2
                padding = dilation
                branch = nn.Sequential(
                    nn.Conv2d(
                        dim,
                        dim,
                        kernel_size=3,
                        padding=padding,
                        dilation=dilation,
                        groups=dim,
                        bias=False,
                    ),
                    nn.BatchNorm2d(dim),
                    nn.ReLU6(inplace=True),
                )
            self.branches.append(branch)

        hidden_dim = max(dim // reduction, 32)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc_reduce = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
        )
        self.fc_expand = nn.Conv2d(hidden_dim, dim * self.num_branches, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        b = x.shape[0]
        feats = [branch(x) for branch in self.branches]
        feats_stack = torch.stack(feats, dim=1)
        fused = feats_stack.sum(dim=1)
        z = self.fc_reduce(self.gap(fused))
        attn = self.fc_expand(z).view(b, self.num_branches, self.dim, 1, 1)
        attn = self.softmax(attn)
        return (feats_stack * attn).sum(dim=1)


class ASGBlock(nn.Module):
    def __init__(self, dim: int, mlp_ratio: int = 4, drop_path: float = 0.0):
        super().__init__()
        mid_dim = int(dim * mlp_ratio)
        self.sa = SpatialAttention(kernel_size=7)
        self.dwconv = ConvBN(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mid_dim, kernel_size=1, with_bn=False)
        self.f2 = ConvBN(dim, mid_dim, kernel_size=1, with_bn=False)
        self.act = nn.ReLU6(inplace=True)
        self.grn = GRN(mid_dim)
        self.g = ConvBN(mid_dim, dim, kernel_size=1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim, with_bn=False)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        identity = x
        x = self.sa(x)
        x = self.dwconv(x)
        x = self.act(self.f1(x)) * self.f2(x)
        x = self.grn(x)
        x = self.dwconv2(self.g(x))
        return identity + self.drop_path(x)


class SKSGBlock(nn.Module):
    def __init__(self, dim: int, mlp_ratio: int = 4, drop_path: float = 0.0, sk_kernel_sizes=(3, 7)):
        super().__init__()
        mid_dim = int(dim * mlp_ratio)
        self.content_dw = ConvBN(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim, with_bn=True)
        self.content_proj = ConvBN(dim, mid_dim, kernel_size=1, with_bn=False)
        self.sk = SelectiveKernel(dim, kernel_sizes=sk_kernel_sizes)
        self.gate_proj = ConvBN(dim, mid_dim, kernel_size=1, with_bn=False)
        self.act = nn.ReLU6(inplace=True)
        self.grn = GRN(mid_dim)
        self.out_proj = ConvBN(mid_dim, dim, kernel_size=1, with_bn=True)
        self.out_dw = ConvBN(dim, dim, kernel_size=7, stride=1, padding=3, groups=dim, with_bn=False)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        identity = x
        content = self.content_proj(self.content_dw(x))
        gate = self.gate_proj(self.sk(x))
        x = self.act(content) * gate
        x = self.grn(x)
        x = self.out_dw(self.out_proj(x))
        return identity + self.drop_path(x)


class CASGNet(nn.Module):
    def __init__(
        self,
        base_dim: int = 32,
        depths=(2, 2, 8, 3),
        mlp_ratio: int = 4,
        drop_path_rate: float = 0.0,
        num_classes: int = 1000,
        sk_kernel_sizes=(3, 7),
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channel = 32

        self.stem = nn.Sequential(
            ConvBN(3, self.in_channel, kernel_size=3, stride=2, padding=1),
            nn.ReLU6(inplace=True),
        )

        self.stages = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i_layer, depth in enumerate(depths):
            embed_dim = base_dim * (2 ** i_layer)
            down_sampler = ConvBN(self.in_channel, embed_dim, kernel_size=3, stride=2, padding=1)
            self.in_channel = embed_dim
            blocks = []
            for i in range(depth):
                if i_layer < len(depths) - 1:
                    blk = ASGBlock(self.in_channel, mlp_ratio=mlp_ratio, drop_path=dpr[cur + i])
                else:
                    blk = SKSGBlock(
                        self.in_channel,
                        mlp_ratio=mlp_ratio,
                        drop_path=dpr[cur + i],
                        sk_kernel_sizes=sk_kernel_sizes,
                    )
                blocks.append(blk)
            cur += depth
            self.stages.append(nn.Sequential(down_sampler, *blocks))

        self.norm = nn.BatchNorm2d(self.in_channel)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.in_channel, num_classes)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.norm(x)
        x = torch.flatten(self.avgpool(x), 1)
        return x

    def forward(self, x):
        return self.head(self.forward_features(x))


@register_model
def casgnet_s1(pretrained=False, **kwargs):
    model = CASGNet(base_dim=24, depths=(2, 2, 8, 3), mlp_ratio=4, sk_kernel_sizes=(3, 9), **kwargs)
    if pretrained:
        num_classes = kwargs.get("num_classes", 1000)
        _load_pretrained_starnet(model, "starnet_s1", num_classes)
    return model


@register_model
def casgnet_s2(pretrained=False, **kwargs):
    model = CASGNet(base_dim=32, depths=(2, 2, 8, 3), mlp_ratio=4, sk_kernel_sizes=(3, 7), **kwargs)
    if pretrained:
        num_classes = kwargs.get("num_classes", 1000)
        _load_pretrained_starnet(model, "starnet_s2", num_classes)
    return model
