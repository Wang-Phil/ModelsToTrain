"""
Implementation of Prof-of-Concept Network: StarNet.

We make StarNet as simple as possible [to show the key contribution of element-wise multiplication]:
    - like NO layer-scale in network design,
    - and NO EMA during training,
    - which would improve the performance further.

Created by: Xu Ma (Email: ma.xu1@northeastern.edu)
Modified Date: Mar/29/2024
"""
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model


# ── 注意力模块 ────────────────────────────────────────────────────────────────

class SpatialAttention(nn.Module):
    """空间注意力: 沿通道维度做 AvgPool+MaxPool 后卷积生成空间掩码"""
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = self.sigmoid(self.conv1(torch.cat([avg_out, max_out], dim=1)))
        return x * scale


class ChannelAttention(nn.Module):
    """通道注意力: 双路 GAP+GMP 经 MLP 生成通道权重"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x))
        return x * self.sigmoid(attn)


class CBAM(nn.Module):
    """CBAM: 先通道注意力，再空间注意力"""
    def __init__(self, channels, reduction=16, sa_kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=sa_kernel_size)

    def forward(self, x):
        return self.sa(self.ca(x))


# ─────────────────────────────────────────────────────────────────────────────

class GRN(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        
        # gamma (γ) 和 beta (β) 是通道维度的可学习参数
        # 初始化为 1 和 0，形状为 (1, C, 1, 1)，便于广播
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.gamma.data.fill_(1.0) # γ 初始化为 1
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 计算响应范数（Response Normalization）
        # across spatial dimensions (H, W)
        
        # 计算 x^2 在 H, W 维度上的均值
        # 结果 shape: (B, C, 1, 1)
        Gx = x.pow(2).mean(dim=[2, 3], keepdim=True)
        
        # 开根号得到 L2 范数（Response Norm）
        # R_norm = sqrt(Gx) + epsilon
        Rx = torch.sqrt(Gx + self.eps)

        # 2. 归一化和竞争增强
        # Response = X / R_norm
        NormX = x / Rx
        
        # 3. 最终输出（根据用户提供的公式）
        # Output = gamma * NormX + beta + X
        out = self.gamma * NormX + self.beta + x
        
        return out


class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)


# ── Selective Kernel Unit (CompleteSKUnit) ────────────────────────────────────

class CompleteSKUnit(nn.Module):
    """严格遵循 SKNet 的 Split -> Fuse -> Select 流程。
    使用 3×3 + dilation 模拟大核，减少参数量。
    """
    def __init__(self, dim, kernel_sizes=[3, 9], reduction=4):
        super().__init__()
        self.dim = dim
        self.num_branches = len(kernel_sizes)
        # Split: 多尺度分支
        self.branches = nn.ModuleList()
        for ks in kernel_sizes:
            if ks == 1:
                branch = nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=1, padding=0, groups=dim, bias=False),
                    nn.BatchNorm2d(dim), nn.ReLU6()
                )
            else:
                dilation = 1 if ks == 3 else (ks - 1) // 2
                branch = nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=3, padding=dilation,
                              dilation=dilation, groups=dim, bias=False),
                    nn.BatchNorm2d(dim), nn.ReLU6()
                )
            self.branches.append(branch)
        # Fuse & Select
        hidden_dim = max(dim // reduction, 32)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc_reduce = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim), nn.ReLU6()
        )
        self.fc_expand = nn.Conv2d(hidden_dim, dim * self.num_branches, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        B = x.shape[0]
        feats = [b(x) for b in self.branches]
        feats_stack = torch.stack(feats, dim=1)           # [B, K, C, H, W]
        U = feats_stack.sum(dim=1)                         # [B, C, H, W]
        z = self.fc_reduce(self.gap(U))
        weights = self.fc_expand(z)                        # [B, K*C, 1, 1]
        weights = self.softmax(weights.view(B, self.num_branches, self.dim, 1, 1))
        return (feats_stack * weights).sum(dim=1)          # [B, C, H, W]


class SKBlock(nn.Module):
    """StarNet Block，用 CompleteSKUnit 替换首段 dwconv；可选在 SK 前加 SA、在 Star 后加 GRN（与 Block 中位置一致）。"""
    def __init__(self, dim, mlp_ratio=3, drop_path=0., sk_kernel_sizes=[3, 9],
                 use_attn=False, use_grn=False):
        super().__init__()
        self.attn = SpatialAttention(kernel_size=7) if use_attn else nn.Identity()
        self.sk = CompleteSKUnit(dim, kernel_sizes=sk_kernel_sizes)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mid_dim = int(dim * mlp_ratio)
        self.grn = GRN(mid_dim) if use_grn else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.attn(x)
        x = self.sk(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.grn(x)
        x = self.dwconv2(self.g(x))
        return shortcut + self.drop_path(x)


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0., attn_type="spatial", use_grn=False,
                 sk_kernel_sizes=None):
        """
        Args:
            attn_type: "spatial" | "channel" | "cbam" | "none"
                       注意力施加在 dwconv 之前（与 starnet.py 位置一致）
            use_grn  : 是否在 star 乘法之后插入 GRN（作用于 mid_dim 中间特征）
            sk_kernel_sizes: 若为非 None 列表，则用 CompleteSKUnit 替代首段 7×7 dwconv（SA→SKUnit→f1/f2…）
        """
        super().__init__()
        self.attn = self._build_attn(attn_type, dim)
        if sk_kernel_sizes is not None:
            self.dwconv = CompleteSKUnit(dim, kernel_sizes=sk_kernel_sizes)
        else:
            self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mid_dim = int(dim * mlp_ratio)
        self.grn = GRN(mid_dim) if use_grn else nn.Identity()

    @staticmethod
    def _build_attn(attn_type, channels):
        t = (attn_type or "none").lower()
        if t == "spatial": return SpatialAttention(kernel_size=7)
        if t == "channel": return ChannelAttention(channels)
        if t == "cbam":    return CBAM(channels)
        return nn.Identity()

    def forward(self, x):
        input = x
        x = self.attn(x)
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.grn(x)                  # GRN: star 乘法后，g 投影前
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x


class AbStarNet(nn.Module):
    def __init__(self, base_dim=32, depths=[3, 3, 12, 5], mlp_ratio=4, drop_path_rate=0.0,
                 num_classes=1000, attn_type="spatial", use_sa_stages=None, use_grn=False,
                 use_sk_stages=None, sk_kernel_sizes=[3, 9], sk_tail_plain_block=False,
                 sk_blocks_attn=False, sk_blocks_grn=False,
                 sk_last_block_only=False,
                 **kwargs):
        """
        Args:
            attn_type      : 全局注意力类型，"spatial" | "channel" | "cbam" | "none"。
            use_sa_stages  : list[bool]，逐 stage 控制是否启用注意力，None 表示全部使用 attn_type。
            use_grn        : 是否在所有 Block 的 star 乘法后插入 GRN。
            use_sk_stages  : list[bool]，逐 stage 控制是否将普通 Block 替换为 SKBlock。
                             None 表示全部使用普通 Block。
                             SKBlock 优先级高于注意力设置（SKBlock 内无 attn/grn）。
            sk_kernel_sizes: SKBlock 中 CompleteSKUnit 使用的多尺度卷积核列表，默认 [3, 9]。
            sk_tail_plain_block: 若某 stage 使用 SKBlock，则该 stage **最后一个** block 改为 Block：
                             SA +（可选）用 sk_kernel_sizes 的 CompleteSKUnit 替代首层 dwconv + GRN + Star 后半；
                             不传 sk_kernel_sizes 给该 Block 时即为 SA+7x7dw+GRN。full 模型在末块传入与 SK 段相同核组 → SA+SKUnit+GRN。
            sk_blocks_attn: 若为 True，**该 stage 内每个 SKBlock** 均在 SK 前加 SpatialAttention（与 sk_tail_plain 末块 Block 分支无关）。
            sk_blocks_grn: 若为 True，**每个 SKBlock** 均在 Star 后加 GRN。
            sk_last_block_only: 若为 True，则仅在**最后一个 stage 的最后一个 block** 用 SKBlock（只替换该 block 的首段 dw 为 SKUnit）。
                               该 SKBlock 的 SA/GRN 由 sk_blocks_attn/sk_blocks_grn 控制；该开关与 use_sk_stages 无关。
        """
        super().__init__()
        self.num_classes = num_classes
        self.in_channel = 32
        # stem layer
        self.stem = nn.Sequential(ConvBN(3, self.in_channel, kernel_size=3, stride=2, padding=1), nn.ReLU6())
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] # stochastic depth
        # build stages
        self.stages = nn.ModuleList()
        cur = 0
        for i_layer in range(len(depths)):
            embed_dim = base_dim * 2 ** i_layer
            down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1)
            self.in_channel = embed_dim
            # 判断该 stage 是否使用 SKBlock
            stage_use_sk = (use_sk_stages is not None
                            and i_layer < len(use_sk_stages)
                            and bool(use_sk_stages[i_layer]))
            if stage_use_sk:
                n_blk = depths[i_layer]
                last_plain = (
                    sk_tail_plain_block
                    and i_layer == len(depths) - 1
                    and n_blk >= 1
                )
                blocks = []
                for i in range(n_blk):
                    if last_plain and i == n_blk - 1:
                        if use_sa_stages is not None:
                            st_en = bool(use_sa_stages[i_layer]) if i_layer < len(use_sa_stages) else False
                            stage_attn = attn_type if st_en else "none"
                        else:
                            stage_attn = attn_type
                        blocks.append(
                            Block(self.in_channel, mlp_ratio, dpr[cur + i],
                                  attn_type=stage_attn, use_grn=use_grn,
                                  sk_kernel_sizes=sk_kernel_sizes)
                        )
                    else:
                        blocks.append(
                            SKBlock(self.in_channel, mlp_ratio, dpr[cur + i],
                                    sk_kernel_sizes=sk_kernel_sizes,
                                    use_attn=bool(sk_blocks_attn),
                                    use_grn=bool(sk_blocks_grn))
                        )
            else:
                # 按 stage 决定实际注意力类型
                if use_sa_stages is not None:
                    stage_enabled = bool(use_sa_stages[i_layer]) if i_layer < len(use_sa_stages) else False
                    stage_attn = attn_type if stage_enabled else "none"
                else:
                    stage_attn = attn_type
                if sk_last_block_only and i_layer == len(depths) - 1 and depths[i_layer] >= 1:
                    blocks = []
                    for i in range(depths[i_layer]):
                        if i == depths[i_layer] - 1:
                            blocks.append(
                                SKBlock(self.in_channel, mlp_ratio, dpr[cur + i],
                                        sk_kernel_sizes=sk_kernel_sizes,
                                        use_attn=bool(sk_blocks_attn),
                                        use_grn=bool(sk_blocks_grn))
                            )
                        else:
                            blocks.append(
                                Block(self.in_channel, mlp_ratio, dpr[cur + i],
                                      attn_type=stage_attn, use_grn=use_grn)
                            )
                else:
                    blocks = [Block(self.in_channel, mlp_ratio, dpr[cur + i],
                                    attn_type=stage_attn, use_grn=use_grn)
                              for i in range(depths[i_layer])]
            cur += depths[i_layer]
            self.stages.append(nn.Sequential(down_sampler, *blocks))
        # head
        self.norm = nn.BatchNorm2d(self.in_channel)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.in_channel, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear or nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm or nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = torch.flatten(self.avgpool(self.norm(x)), 1)
        return self.head(x)


@register_model
def ab_starnet_s1(pretrained=False, **kwargs):
    return AbStarNet(24, [2, 2, 8, 3], **kwargs)


@register_model
def ab_starnet_s2(pretrained=False, **kwargs):
    return AbStarNet(32, [1, 2, 6, 2], **kwargs)


@register_model
def ab_starnet_s3(pretrained=False, **kwargs):
    return AbStarNet(32, [2, 2, 8, 4], **kwargs)


@register_model
def ab_starnet_s4(pretrained=False, **kwargs):
    return AbStarNet(32, [3, 3, 12, 5], **kwargs)


# very small networks #
@register_model
def ab_starnet_s050(pretrained=False, **kwargs):
    return AbStarNet(16, [1, 1, 3, 1], 3, **kwargs)


@register_model
def ab_starnet_s100(pretrained=False, **kwargs):
    return AbStarNet(20, [1, 2, 4, 1], 4, **kwargs)


@register_model
def ab_starnet_s150(pretrained=False, **kwargs):
    return AbStarNet(24, [1, 2, 4, 2], 3, **kwargs)


# ── 注意力位置消融实验 (base: AbStarNet-S1, depths=[2,2,8,3]) ──────────────
# Stage 索引:  0       1       2       3
# 通道数:     24      48      96     192

@register_model
def ab_starnet_sa_all(pretrained=False, **kwargs):
    """消融-1: 全部4个Stage均加入空间注意力"""
    return AbStarNet(24, [2, 2, 8, 3],
                     use_sa_stages=[True, True, True, True], **kwargs)


@register_model
def ab_starnet_sa_last3(pretrained=False, **kwargs):
    """消融-2: 后3个Stage加入空间注意力 (Stage 1-3)"""
    return AbStarNet(24, [2, 2, 8, 3],
                     use_sa_stages=[False, True, True, True], **kwargs)


@register_model
def ab_starnet_sa_last2(pretrained=False, **kwargs):
    """消融-3: 后2个Stage加入空间注意力 (Stage 2-3)"""
    return AbStarNet(24, [2, 2, 8, 3],
                     use_sa_stages=[False, False, True, True], **kwargs)


@register_model
def ab_starnet_sa_last1(pretrained=False, **kwargs):
    """消融-4: 仅最后1个Stage加入空间注意力 (Stage 3)"""
    return AbStarNet(24, [2, 2, 8, 3],
                     use_sa_stages=[False, False, False, True], **kwargs)


# ── 注意力类别消融实验 (base: AbStarNet-S1, 全部Stage均加注意力) ─────────────

@register_model
def ab_starnet_attn_spatial(pretrained=False, **kwargs):
    """注意力类别消融-1: 全部Block使用空间注意力 (SpatialAttention)"""
    return AbStarNet(24, [2, 2, 8, 3],
                     attn_type="spatial", **kwargs)


@register_model
def ab_starnet_attn_channel(pretrained=False, **kwargs):
    """注意力类别消融-2: 全部Block使用通道注意力 (ChannelAttention)"""
    return AbStarNet(24, [2, 2, 8, 3],
                     attn_type="channel", **kwargs)


@register_model
def ab_starnet_attn_cbam(pretrained=False, **kwargs):
    """注意力类别消融-3: 全部Block使用CBAM (Channel + Spatial Attention)"""
    return AbStarNet(24, [2, 2, 8, 3],
                     attn_type="cbam", **kwargs)


# ── 纯基线模型 (无任何注意力 / 无 GRN) ───────────────────────────────────────

@register_model
def ab_starnet_baseline(pretrained=False, **kwargs):
    """纯 StarNet-S1 基线：无 SA、无 GRN，与原始 StarNet 论文结构一致"""
    return AbStarNet(24, [2, 2, 8, 3],
                     attn_type="none", use_grn=False, **kwargs)


# ── SKUnit 位置消融实验 (kernel_sizes=[3,9], 无 attn, 无 GRN) ──────────────────
# Stage 索引:  0       1       2       3
# 通道数:     24      48      96     192
# SKBlock 替换对应 stage 的普通 Block，其余 stage 保持纯基线（attn=none, grn=False）

@register_model
def ab_starnet_sk39_all(pretrained=False, **kwargs):
    """SK消融-1: 全部4个Stage均使用 SKBlock (kernel=[3,9])"""
    return AbStarNet(24, [2, 2, 8, 3], attn_type="none", use_grn=False,
                     use_sk_stages=[True, True, True, True],
                     sk_kernel_sizes=[3, 9], **kwargs)


@register_model
def ab_starnet_sk39_last3(pretrained=False, **kwargs):
    """SK消融-2: 后3个Stage使用 SKBlock (Stage 1-3)，Stage 0 为普通 Block"""
    return AbStarNet(24, [2, 2, 8, 3], attn_type="none", use_grn=False,
                     use_sk_stages=[False, True, True, True],
                     sk_kernel_sizes=[3, 9], **kwargs)


@register_model
def ab_starnet_sk39_last2(pretrained=False, **kwargs):
    """SK消融-3: 后2个Stage使用 SKBlock (Stage 2-3)，Stage 0-1 为普通 Block"""
    return AbStarNet(24, [2, 2, 8, 3], attn_type="none", use_grn=False,
                     use_sk_stages=[False, False, True, True],
                     sk_kernel_sizes=[3, 9], **kwargs)


@register_model
def ab_starnet_sk39_last1(pretrained=False, **kwargs):
    """SK消融-4: 仅最后1个Stage使用 SKBlock (Stage 3)，其余为普通 Block"""
    return AbStarNet(24, [2, 2, 8, 3], attn_type="none", use_grn=False,
                     use_sk_stages=[False, False, False, True],
                     sk_kernel_sizes=[3, 9], **kwargs)


# ── SKUnit 卷积核大小消融实验 (仅 Stage3, 无 attn, 无 GRN) ─────────────────────
# 枚举 1~9（步长2）的全部两两组合，共 10 组：
#   [1,3] [1,5] [1,7] [1,9]  [3,5] [3,7] [3,9]*  [5,7] [5,9]  [7,9]
# * [3,9] 即 ab_starnet_sk39_last1，已在上方定义，此处略过
# 命名规则: ab_starnet_sk{k1}{k2}_last1

_SK_LAST1 = dict(attn_type="none", use_grn=False,
                 use_sk_stages=[False, False, False, True])


@register_model
def ab_starnet_sk13_last1(pretrained=False, **kwargs):
    """SKUnit 核大小消融: kernel=[1,3], 仅 Stage3"""
    return AbStarNet(24, [2, 2, 8, 3], sk_kernel_sizes=[1, 3], **_SK_LAST1, **kwargs)


@register_model
def ab_starnet_sk15_last1(pretrained=False, **kwargs):
    """SKUnit 核大小消融: kernel=[1,5], 仅 Stage3"""
    return AbStarNet(24, [2, 2, 8, 3], sk_kernel_sizes=[1, 5], **_SK_LAST1, **kwargs)


@register_model
def ab_starnet_sk17_last1(pretrained=False, **kwargs):
    """SKUnit 核大小消融: kernel=[1,7], 仅 Stage3"""
    return AbStarNet(24, [2, 2, 8, 3], sk_kernel_sizes=[1, 7], **_SK_LAST1, **kwargs)


@register_model
def ab_starnet_sk19_last1(pretrained=False, **kwargs):
    """SKUnit 核大小消融: kernel=[1,9], 仅 Stage3"""
    return AbStarNet(24, [2, 2, 8, 3], sk_kernel_sizes=[1, 9], **_SK_LAST1, **kwargs)


@register_model
def ab_starnet_sk35_last1(pretrained=False, **kwargs):
    """SKUnit 核大小消融: kernel=[3,5], 仅 Stage3"""
    return AbStarNet(24, [2, 2, 8, 3], sk_kernel_sizes=[3, 5], **_SK_LAST1, **kwargs)


@register_model
def ab_starnet_sk37_last1(pretrained=False, **kwargs):
    """SKUnit 核大小消融: kernel=[3,7], 仅 Stage3"""
    return AbStarNet(24, [2, 2, 8, 3], sk_kernel_sizes=[3, 7], **_SK_LAST1, **kwargs)


@register_model
def ab_starnet_sk57_last1(pretrained=False, **kwargs):
    """SKUnit 核大小消融: kernel=[5,7], 仅 Stage3"""
    return AbStarNet(24, [2, 2, 8, 3], sk_kernel_sizes=[5, 7], **_SK_LAST1, **kwargs)


@register_model
def ab_starnet_sk59_last1(pretrained=False, **kwargs):
    """SKUnit 核大小消融: kernel=[5,9], 仅 Stage3"""
    return AbStarNet(24, [2, 2, 8, 3], sk_kernel_sizes=[5, 9], **_SK_LAST1, **kwargs)


@register_model
def ab_starnet_sk79_last1(pretrained=False, **kwargs):
    """SKUnit 核大小消融: kernel=[7,9], 仅 Stage3"""
    return AbStarNet(24, [2, 2, 8, 3], sk_kernel_sizes=[7, 9], **_SK_LAST1, **kwargs)


# ── SA + GRN 联合实验 ──────────────────────────────────────────────────────────

@register_model
def ab_starnet_sa_grn(pretrained=False, **kwargs):
    """全部Block加入空间注意力 + GRN（在 ab_starnet_sa_all 基础上叠加 GRN）
    
    结构: SA → dwconv → f1/f2 → act(x1)*x2 → GRN → g → dwconv2 → residual
    """
    return AbStarNet(24, [2, 2, 8, 3],
                     attn_type="spatial", use_grn=True, **kwargs)


# ── CASGNet 整体消融 (StarNet-S1 主干 depths=[2,2,8,3]) ───────────────────────
# 与 train_multiclass / run_multi_model_cv 中 casgnet_ab_all 目录对应。
# 含 SK 的实验：Stage3 内为 SKBlock，核 [1,7]。
# ab_starnet_casg_ab_full：Stage3 三格均为 SKBlock，SA→SKUnit[1,7]→Star→GRN（sk_blocks_attn+sk_blocks_grn）。

@register_model
def ab_starnet_casg_ab_sa(pretrained=False, **kwargs):
    """1) 仅空间注意力：各 Block 前加 SA，无 GRN、无 SK。"""
    return AbStarNet(24, [2, 2, 8, 3],
                     attn_type="spatial", use_grn=False, **kwargs)


@register_model
def ab_starnet_casg_ab_grn(pretrained=False, **kwargs):
    """2) 仅 GRN：无注意力，全部 Block 在 star 后接 GRN，无 SK。"""
    return AbStarNet(24, [2, 2, 8, 3],
                     attn_type="none", use_grn=True, **kwargs)


@register_model
def ab_starnet_casg_ab_sk17(pretrained=False, **kwargs):
    """3) 仅 SKUnit：最后 Stage 的 Block 换为 SKBlock，核 [1,7]；其余为纯基线 Block。"""
    return AbStarNet(24, [2, 2, 8, 3], attn_type="none", use_grn=False,
                     use_sk_stages=[False, False, False, True],
                     sk_kernel_sizes=[1, 7], **kwargs)


@register_model
def ab_starnet_casg_ab_sk17_all_sa(pretrained=False, **kwargs):
    """与 sk17 相同，Stage3 内每个 SKBlock：SK 前加 SpatialAttention（SA→SK→…）。"""
    return AbStarNet(24, [2, 2, 8, 3], attn_type="spatial", use_grn=False,
                     use_sk_stages=[False, False, False, True],
                     sk_kernel_sizes=[1, 7],
                     sk_blocks_attn=True, **kwargs)


@register_model
def ab_starnet_casg_ab_sk17_all_grn(pretrained=False, **kwargs):
    """与 sk17 相同，Stage3 内每个 SKBlock：Star 乘积后加 GRN。"""
    return AbStarNet(24, [2, 2, 8, 3], attn_type="none", use_grn=True,
                     use_sk_stages=[False, False, False, True],
                     sk_kernel_sizes=[1, 7],
                     sk_blocks_grn=True, **kwargs)


@register_model
def ab_starnet_casg_ab_sk17_last_sa(pretrained=False, **kwargs):
    """Deprecated 名称：与 ab_starnet_casg_ab_sk17_all_sa 相同（全 SKBlock 加 SA）。"""
    return ab_starnet_casg_ab_sk17_all_sa(pretrained=pretrained, **kwargs)


@register_model
def ab_starnet_casg_ab_sk17_last_grn(pretrained=False, **kwargs):
    """Deprecated 名称：与 ab_starnet_casg_ab_sk17_all_grn 相同（全 SKBlock Star 后 GRN）。"""
    return ab_starnet_casg_ab_sk17_all_grn(pretrained=pretrained, **kwargs)


@register_model
def ab_starnet_casg_ab_sa_grn(pretrained=False, **kwargs):
    """4) 空间注意力 + GRN（全 Stage Block，无 SK）。等价于 ab_starnet_sa_grn。"""
    return AbStarNet(24, [2, 2, 8, 3],
                     attn_type="spatial", use_grn=True, **kwargs)


@register_model
def ab_starnet_casg_ab_sa_sk17(pretrained=False, **kwargs):
    """5) 空间注意力 + 仅最后 Stage 的 SK [1,7]（前 3 个 Stage 为带 SA 的普通 Block）。"""
    return AbStarNet(24, [2, 2, 8, 3],
                     attn_type="spatial", use_grn=False,
                     use_sk_stages=[False, False, False, True],
                     sk_kernel_sizes=[1, 7], **kwargs)


@register_model
def ab_starnet_casg_ab_grn_sk17(pretrained=False, **kwargs):
    """6) GRN + 仅最后 Stage 的 SK [1,7]（前 3 个 Stage 为仅 GRN 的普通 Block）。"""
    return AbStarNet(24, [2, 2, 8, 3],
                     attn_type="none", use_grn=True,
                     use_sk_stages=[False, False, False, True],
                     sk_kernel_sizes=[1, 7], **kwargs)


@register_model
def ab_starnet_casg_ab_full(pretrained=False, **kwargs):
    """7) Stage0–2：每 block 为 SA+GRN。Stage3：3 个 block 均为 SA→SKUnit[1,7]→Star→GRN（先注意力，再 SK 融合，Star 后 GRN）。"""
    return AbStarNet(24, [2, 2, 8, 3],
                     attn_type="spatial", use_grn=True,
                     use_sk_stages=[False, False, False, True],
                     sk_kernel_sizes=[1, 7],
                     sk_blocks_attn=True, sk_blocks_grn=True, **kwargs)


@register_model
def ab_starnet_casg_ab_sa_grn_last2_last_sk17(pretrained=False, **kwargs):
    """8) Stage0–2：每 block 为 SA+GRN。Stage3：前 2 个为 SA+GRN 的普通 Block；最后 1 个为仅 SKUnit[1,7] 的 SKBlock（无 SA/GRN）。"""
    return AbStarNet(24, [2, 2, 8, 3],
                     attn_type="spatial", use_grn=True,
                     use_sk_stages=[False, False, False, False],
                     sk_kernel_sizes=[1, 7],
                     sk_last_block_only=True,
                     sk_blocks_attn=False, sk_blocks_grn=False,
                     **kwargs)
