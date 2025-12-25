import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model

# ==========================================
# 公共 Conv+BN 工具
# ==========================================
class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)

# ==========================================
# 空间注意力模块（可选）
# ==========================================
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.conv1(x_cat))
        return x * scale

# ==========================================
# Selective Kernel Unit（多尺度分支 + 自适应权重）
# - 兼顾轻量：使用 depthwise conv branches，然后基于 GAP 做分支权重
# ==========================================
class SelectiveKernelUnit(nn.Module):
    def __init__(self, dim, kernel_sizes=(1, 3), use_dilation=True, reduction=4):
        super().__init__()
        self.dim = dim
        self.kernel_sizes = kernel_sizes
        self.num_branches = len(kernel_sizes)

        # branches: depthwise convs with different kernel/dilation
        self.branches = nn.ModuleList()
        for ks in kernel_sizes:
            if use_dilation and ks > 3:
                dilation = (ks - 1) // 2
                padding = dilation
                branch = nn.Sequential(
                    nn.Conv2d(dim, dim, 3, padding=padding, dilation=dilation, groups=dim, bias=False),
                    nn.BatchNorm2d(dim),
                    nn.ReLU6(inplace=True)
                )
            else:
                padding = ks // 2
                branch = nn.Sequential(
                    nn.Conv2d(dim, dim, ks, padding=padding, groups=dim, bias=False),
                    nn.BatchNorm2d(dim),
                    nn.ReLU6(inplace=True)
                )
            self.branches.append(branch)

        # attention generator
        hidden_dim = max(dim // reduction, 32)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, dim * self.num_branches, 1, bias=False)
        )
        self.softmax = nn.Softmax(dim=1)

        # projection after fusion (pointwise)
        self.proj = ConvBN(dim, dim, 1, with_bn=True)

    def forward(self, x):
        # parallel branch features
        branch_feats = [b(x) for b in self.branches]  # list of [B, C, H, W]
        feat_sum = sum(branch_feats)                  # fusion for global descriptor
        attn = self.gap(feat_sum)                     # [B, C, 1, 1]
        attn_weights = self.fc(attn)                  # [B, num_branches*C, 1, 1]
        B = x.shape[0]
        attn_weights = attn_weights.view(B, self.num_branches, self.dim, 1, 1)
        attn_weights = self.softmax(attn_weights)     # softmax over branches
        # weighted sum
        fused = sum(attn_weights[:, i] * branch_feats[i] for i in range(self.num_branches))
        out = self.proj(fused)
        return out

# ==========================================
# 综合 Block：保持 StarNet 的结构，同时可插入 SK 单元与空间注意力
# design:
#  - dwconv (7x7, groups=dim) -> 分支（f1,f2）形式激活 -> g -> dwconv2 -> 可选 SK 融合 -> 残差
#  - or: 将 SK 放到 dwconv 后对特征做多尺度融合
# ==========================================
class MultiScaleBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0., use_sk=False, sk_kernel_sizes=(1, 3), use_attn=False):
        super().__init__()
        self.dim = dim
        self.dwconv = ConvBN(dim, dim, 7, 1, padding=(7-1)//2, groups=dim, with_bn=True)   # large receptive field
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, padding=(7-1)//2, groups=dim, with_bn=False)
        self.act = nn.ReLU6(inplace=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # 可选 SK 单元（放在 g 输出之后做多尺度融合）
        self.use_sk = use_sk
        if self.use_sk:
            self.sk = SelectiveKernelUnit(dim, kernel_sizes=sk_kernel_sizes, use_dilation=True)

        # 可选空间注意力
        self.use_attn = use_attn
        if self.use_attn:
            self.sa = SpatialAttention(kernel_size=7)

    def forward(self, x):
        identity = x
        # optionally spatial attention
        if self.use_attn:
            x = self.sa(x)
        out = self.dwconv(x)   
        
        # optionally multiscale fusion
        if self.use_sk:
            out = self.sk(out)

        # [B,C,H,W]
        x1, x2 = self.f1(out), self.f2(out)
        out = self.act(x1) * x2      # gating
        out = self.g(out)

        # second depthwise conv
        out = self.dwconv2(out)

        out = identity + self.drop_path(out)
        return out

# ==========================================
# StarNet 主干（可配置：use_sk、use_attn、sk_kernel_sizes）
# ==========================================
class StarNetSK(nn.Module):
    def __init__(self, base_dim=32, depths=[3,3,12,5], mlp_ratio=3, drop_path_rate=0.0,
                 num_classes=1000, dropout_rate=0.0, use_sk=False, use_attn=False,
                 sk_kernel_sizes=(1,3), **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.in_channel = 32

        # stem
        self.stem = nn.Sequential(ConvBN(3, self.in_channel, kernel_size=3, stride=2, padding=1), nn.ReLU6(inplace=True))

        # stochastic depth schedule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # stages
        self.stages = nn.ModuleList()
        cur = 0
        for i_layer in range(len(depths)):
            embed_dim = base_dim * (2 ** i_layer)
            down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1)
            self.in_channel = embed_dim

            blocks = [
                MultiScaleBlock(self.in_channel, mlp_ratio, dpr[cur + i], use_sk=use_sk, sk_kernel_sizes=sk_kernel_sizes, use_attn=use_attn)
                for i in range(depths[i_layer])
            ]
            cur += depths[i_layer]
            self.stages.append(nn.Sequential(down_sampler, *blocks))

        # head
        self.norm = nn.BatchNorm2d(self.in_channel)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        feat_dim = self.in_channel
        self.head = nn.Linear(feat_dim, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        # 修正 isinstance 用法
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            try:
                trunc_normal_(m.weight, std=.02)
            except Exception:
                pass
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        feat = torch.flatten(self.avgpool(self.norm(x)), 1)
        feat = self.dropout(feat)
        return self.head(feat)

# simple register helpers
@register_model
def starnet_sk_s1(pretrained=False, **kwargs):
    # base_dim=24 depths similar to your example
    model = StarNetSK(base_dim=24, depths=[2,2,8,3],use_sk=True, sk_kernel_sizes=(1, 3), mlp_ratio=3, drop_path_rate=0.0, **kwargs)
    return model
