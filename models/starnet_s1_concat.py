"""
Implementation of Prof-of-Concept Network: StarNet.

We make StarNet as simple as possible [to show the key contribution of element-wise multiplication]:
    - like NO layer-scale in network design,
    - and NO EMA during training,
    - which would improve the performance further.

Created by: Xu Ma (Email: ma.xu1@northeastern.edu)
Modified Date: Mar/29/2024
"""
# import torch
# import torch.nn as nn
# from timm.models.layers import DropPath, trunc_normal_
# from timm.models.registry import register_model

# model_urls = {
#     "starnet_s1": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s1.pth.tar",
#     "starnet_s2": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s2.pth.tar",
#     "starnet_s3": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s3.pth.tar",
#     "starnet_s4": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s4.pth.tar",
# }

# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         # 1. 沿着通道维度做 AvgPool 和 MaxPool
#         # x: [B, C, H, W] -> avg_out: [B, 1, H, W], max_out: [B, 1, H, W]
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
        
#         # 2. 拼接
#         x_cat = torch.cat([avg_out, max_out], dim=1)
        
#         # 3. 卷积 + Sigmoid 生成空间掩码
#         scale = self.sigmoid(self.conv1(x_cat))
        
#         # 4. 施加注意力
#         return x * scale


# class ConvBN(torch.nn.Sequential):
#     def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
#         super().__init__()
#         self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
#         if with_bn:
#             self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
#             torch.nn.init.constant_(self.bn.weight, 1)
#             torch.nn.init.constant_(self.bn.bias, 0)


# class Block(nn.Module):
#     def __init__(self, dim, mlp_ratio=3, drop_path=0.):
#         super().__init__()
#         self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
#         self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
#         self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
#         self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
#         self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
#         self.act = nn.ReLU6()
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.sa = SpatialAttention(kernel_size=7)

#     def forward(self, x):
#         input = x
#         x = self.sa(x)
#         x = self.dwconv(x)
#         x1, x2 = self.f1(x), self.f2(x)
#         x = self.act(x1) * x2
#         x = self.dwconv2(self.g(x))
#         x = input + self.drop_path(x)
#         return x


# class StarNet(nn.Module):
#     def __init__(self, base_dim=32, depths=[3, 3, 12, 5], mlp_ratio=4, drop_path_rate=0.0, num_classes=1000, **kwargs):
#         super().__init__()
#         self.num_classes = num_classes
#         self.in_channel = 32
#         # stem layer
#         self.stem = nn.Sequential(ConvBN(3, self.in_channel, kernel_size=3, stride=2, padding=1), nn.ReLU6())
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] # stochastic depth
#         # build stages
#         self.stages = nn.ModuleList()
#         cur = 0
#         for i_layer in range(len(depths)):
#             embed_dim = base_dim * 2 ** i_layer
#             down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1)
#             self.in_channel = embed_dim
#             blocks = [Block(self.in_channel, mlp_ratio, dpr[cur + i]) for i in range(depths[i_layer])]
#             cur += depths[i_layer]
#             self.stages.append(nn.Sequential(down_sampler, *blocks))
#         # head
#         self.norm = nn.BatchNorm2d(self.in_channel)
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.head = nn.Linear(self.in_channel, num_classes)
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear or nn.Conv2d):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm or nn.BatchNorm2d):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)

#     def forward(self, x):
#         x = self.stem(x)
#         for stage in self.stages:
#             x = stage(x)
#         x = torch.flatten(self.avgpool(self.norm(x)), 1)
#         return self.head(x)


import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from .multi_head_classifiers import ArcFace, CosFace

# ==========================================
# 1. 新增： 空间注意力机制模块
# ==========================================
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. 沿着通道维度做 AvgPool 和 MaxPool
        # x: [B, C, H, W] -> avg_out: [B, 1, H, W], max_out: [B, 1, H, W]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # 2. 拼接
        x_cat = torch.cat([avg_out, max_out], dim=1)
        
        # 3. 卷积 + Sigmoid 生成空间掩码
        scale = self.sigmoid(self.conv1(x_cat))
        
        # 4. 施加注意力
        return x * scale
# ==========================================


# ==========================================
# 2. 新增： 通道注意力机制模块
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        """
        通道注意力模块 (Channel Attention Module)
        
        Args:
            in_channels: 输入通道数
            reduction: 降维比例，用于减少MLP的参数量
        """
        super(ChannelAttention, self).__init__()
        # 计算中间层维度
        hidden_dim = max(in_channels // reduction, 1)
        
        # 共享的MLP（使用1x1卷积实现，便于处理2D特征图）
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 共享的MLP：两层全连接层
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
            nn.ReLU6(),
            nn.Conv2d(hidden_dim, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: 输入特征图 [B, C, H, W]
        
        Returns:
            应用通道注意力后的特征图 [B, C, H, W]
        """
        # 1. 沿着空间维度做 AvgPool 和 MaxPool
        # x: [B, C, H, W] -> avg_out: [B, C, 1, 1], max_out: [B, C, 1, 1]
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        
        # 2. 相加并应用Sigmoid生成通道权重
        scale = self.sigmoid(avg_out + max_out)
        
        # 3. 施加注意力
        return x * scale
# ==========================================


# ==========================================
# 3. 新增： CBAM (Convolutional Block Attention Module) 机制
# ==========================================
class CBAM(nn.Module):
    """
    CBAM: Convolutional Block Attention Module
    结合通道注意力和空间注意力，先应用通道注意力，再应用空间注意力
    """
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        """
        Args:
            in_channels: 输入通道数
            reduction: 通道注意力的降维比例
            kernel_size: 空间注意力的卷积核大小
        """
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        """
        Args:
            x: 输入特征图 [B, C, H, W]
        
        Returns:
            应用CBAM后的特征图 [B, C, H, W]
        """
        # 先应用通道注意力
        x = self.channel_attention(x)
        # 再应用空间注意力
        x = self.spatial_attention(x)
        return x
# ==========================================


# ==========================================
# 4. 新增： 并行注意力机制 (Parallel Attention Module)
# ==========================================
class ParallelAttention(nn.Module):
    """
    Parallel Attention: 并行组合通道注意力和空间注意力
    通道注意力和空间注意力同时作用于输入，然后将结果融合
    """
    def __init__(self, in_channels, reduction=16, kernel_size=7, fusion='add'):
        """
        Args:
            in_channels: 输入通道数
            reduction: 通道注意力的降维比例
            kernel_size: 空间注意力的卷积核大小
            fusion: 融合方式，'add' 表示相加，'multiply' 表示相乘，'concat' 表示拼接
        """
        super(ParallelAttention, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        self.fusion = fusion
        
        # 如果使用拼接融合，需要额外的卷积来降维
        if fusion == 'concat':
            self.fusion_conv = nn.Sequential(
                nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU6()
            )
    
    def forward(self, x):
        """
        Args:
            x: 输入特征图 [B, C, H, W]
        
        Returns:
            应用并行注意力后的特征图 [B, C, H, W]
        """
        # 并行应用通道注意力和空间注意力
        x_ca = self.channel_attention(x)  # 通道注意力结果
        x_sa = self.spatial_attention(x)   # 空间注意力结果
        
        # 融合两种注意力结果
        if self.fusion == 'add':
            # 相加融合
            x = x_ca + x_sa
        elif self.fusion == 'multiply':
            # 相乘融合
            x = x_ca * x_sa
        elif self.fusion == 'concat':
            # 拼接融合
            x = torch.cat([x_ca, x_sa], dim=1)  # [B, 2C, H, W]
            x = self.fusion_conv(x)  # [B, C, H, W]
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion}")
        
        return x
# ==========================================


class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0., with_attn=False, use_sa=True, use_grn=True):
        super().__init__()
        mid_dim = mlp_ratio * dim
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # 空间注意力、GRN开关
        self.sa = SpatialAttention(kernel_size=7) if use_sa else nn.Identity()
        self.grn = GRN(mid_dim) if use_grn else nn.Identity()

    def forward(self, x):
        input = x
        # if self.with_attn:
        x = self.sa(x)  
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.grn(x)
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x


# --- CBAM Block: 使用CBAM机制的Block ---
class CBAMBlock(nn.Module):
    """
    StarNet Block with CBAM (Channel + Spatial Attention)
    位置参考空间注意力，在Block开始处应用CBAM
    """
    def __init__(self, dim, mlp_ratio=3, drop_path=0., with_attn=False):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # CBAM机制：先通道注意力，再空间注意力
        self.cbam = CBAM(in_channels=dim, reduction=16, kernel_size=7)

    def forward(self, x):
        input = x
        # 应用CBAM（位置参考空间注意力，在Block开始处）
        x = self.cbam(x)
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x

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


# --- Parallel Attention Block: 使用并行注意力机制的Block ---
class ParallelAttentionBlock(nn.Module):
    """
    StarNet Block with Parallel Attention (Channel + Spatial Attention in parallel)
    在Block开始处应用并行注意力（通道注意力和空间注意力并行处理，然后融合）
    """
    def __init__(self, dim, mlp_ratio=3, drop_path=0., fusion='add'):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # 并行注意力机制：通道注意力和空间注意力并行处理
        self.parallel_attn = ParallelAttention(in_channels=dim, reduction=16, kernel_size=7, fusion=fusion)

    def forward(self, x):
        input = x
        # 应用并行注意力（在Block开始处）
        x = self.parallel_attn(x)
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x

# --- 1. 完整的 Selective Kernel 单元 ---
class CompleteSKUnit(nn.Module):
    """
    严格遵循 SKNet 的 Split -> Fuse -> Select 流程
    """
    def __init__(self, dim, kernel_sizes=[3, 5], reduction=4):
        super().__init__()
        self.dim = dim
        self.num_branches = len(kernel_sizes)
        
        # --- Split 阶段: 构建多尺度分支 ---
        self.branches = nn.ModuleList()
        for ks in kernel_sizes:
            # 使用 dilation 来模拟大核，减少参数量
            # 1x1: 直接使用1x1卷积
            # 3x3 (dil=1) -> 3x3
            # 3x3 (dil=2) -> 5x5
            # 3x3 (dil=3) -> 7x7
            # 3x3 (dil=4) -> 9x9
            if ks == 1:
                # 1x1卷积，不需要dilation
                branch = nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=1, padding=0, groups=dim, bias=False),
                    nn.BatchNorm2d(dim),
                    nn.ReLU6()
                )
            else:
                # 使用3x3卷积 + dilation来模拟大核
                dilation = 1 if ks == 3 else (ks - 1) // 2
                padding = dilation
                
                branch = nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=3, padding=padding, dilation=dilation, groups=dim, bias=False),
                    nn.BatchNorm2d(dim),
                    nn.ReLU6()
                )
            self.branches.append(branch)
            
        # --- Fuse & Select 阶段: 自适应注意力 ---
        hidden_dim = max(dim // reduction, 32)
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc_reduce = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6()
        )
        
        # 输出维度是 branches * dim，用于给每个分支的每个通道生成权重
        self.fc_expand = nn.Conv2d(hidden_dim, dim * self.num_branches, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        
        # 1. Split: 计算各分支特征
        feats = [branch(x) for branch in self.branches] # List of [B, C, H, W]
        # 堆叠以便后续计算: [B, num_branches, C, H, W]
        feats_stack = torch.stack(feats, dim=1)
        
        # 2. Fuse: 元素级相加，汇聚信息
        U = torch.sum(feats_stack, dim=1) # [B, C, H, W]
        
        # 3. Squeeze: 全局描述符
        s = self.gap(U) # [B, C, 1, 1]
        
        # 4. Excitation: 生成权重
        z = self.fc_reduce(s) 
        weights = self.fc_expand(z) # [B, num_branches * C, 1, 1]
        
        # 变形为 [B, num_branches, C, 1, 1] 以便进行 Softmax
        weights = weights.view(batch_size, self.num_branches, self.dim, 1, 1)
        weights = self.softmax(weights)
        
        # 5. Select: 加权融合
        # feats_stack: [B, branches, C, H, W]
        # weights:     [B, branches, C, 1, 1]
        # 广播机制会自动处理 H, W 维度
        V = torch.sum(feats_stack * weights, dim=1) # [B, C, H, W]
        
        return V

# --- 2. SK 融合的 StarNet Block ---
class SKStarBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0., kernel_sizes=[3, 9], use_sa=True, use_grn=True):
        super().__init__()
        
        # 隐藏层维度
        mid_dim = int(dim * mlp_ratio)
        
        # ===========================
        # 分支 1: 静态内容分支 (Content)
        # 传统的 StarNet 做法，使用 7x7 DWConv 提供大感受野背景
        # ===========================
        self.dw_content = ConvBN(dim, dim, 7, 1, 3, groups=dim, with_bn=True)
        self.f_content = ConvBN(dim, mid_dim, 1, with_bn=False) # 升维
        
        # ===========================
        # 分支 2: 动态门控分支 (Gating)
        # 创新点：使用 SK Unit 来决定"关注什么尺度"
        # ===========================
        # 定义 SK 需要的尺度，例如 3x3 (细节) 和 5x5 (中等上下文)
        self.sk_unit = CompleteSKUnit(dim, kernel_sizes=kernel_sizes)
        self.f_gate = ConvBN(dim, mid_dim, 1, with_bn=False)    # 升维
        
        # ===========================
        # Star Operation 后处理
        # ===========================
        self.act = nn.ReLU6()
        # self.grn = GRN(mid_dim) # 在高维空间做 GRN 效果最好
        self.g = ConvBN(mid_dim, dim, 1, with_bn=True) # 降维回归
        
        # 最后的深度卷积（可选，增强局部性）
        self.dw_final = ConvBN(dim, dim, 7, 1, 3, groups=dim, with_bn=False)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.sa = SpatialAttention(kernel_size=7) if use_sa else nn.Identity()

        self.grn = GRN(mid_dim) if use_grn else nn.Identity()

    def forward(self, x):
        input = x
        x = self.sa(x)
        # --- 路径 1: 静态内容 ---
        x_c = self.dw_content(x)
        x_c = self.f_content(x_c) # [B, mid_dim, H, W]
        
        # --- 路径 2: SK 动态特征 ---
        x_g = self.sk_unit(x)     # SK 融合后的特征
        x_g = self.f_gate(x_g)    # [B, mid_dim, H, W]
        
        # --- Star Interaction ---
        # 逻辑：(激活后的内容) * (动态门控)
        # SK 模块生成的特征作为一种复杂的、多尺度的 Attention Map
        # 来加权筛选静态分支提取的特征
        x = self.act(x_c) * x_g
        
        # --- 后处理 ---
        x = self.grn(x)
        x = self.g(x)
        x = self.dw_final(x)
        
        x = input + self.drop_path(x)
        return x


class StarNet(nn.Module):
    def __init__(self, base_dim=32, depths=[3, 3, 12, 5], mlp_ratio=4, drop_path_rate=0.0, num_classes=1000, dropout_rate=0.1, use_attn=None, use_multi_head=False, cls_num_list=None, use_sa=True, use_grn=True, **kwargs):
        """
        Args:
            base_dim: 基础维度
            depths: 每个stage的block数量
            mlp_ratio: MLP比例
            drop_path_rate: DropPath率
            use_sa: 是否在Block/ SKStarBlock中使用空间注意力
            use_grn: 是否在Block/ SKStarBlock中使用GRN
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
            if i_layer == 3:
                blocks = [ SKStarBlock(self.in_channel, mlp_ratio, dpr[cur + i], use_sa=use_sa, use_grn=use_grn)  for i in range(depths[i_layer]) ]
            else:
                blocks = [ Block(self.in_channel, mlp_ratio, dpr[cur + i], use_sa=use_sa, use_grn=use_grn)  for i in range(depths[i_layer]) ]
            cur += depths[i_layer]
            self.stages.append(nn.Sequential(down_sampler, *blocks))
        # head
        self.norm = nn.BatchNorm2d(self.in_channel)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # Dropout已注释
        # if dropout_rate > 0:
        #     self.dropout = nn.Dropout(dropout_rate)
        # else:
        self.dropout = nn.Identity()
        
        # 特征维度
        feat_dim = self.in_channel
    
        # 单分类头模式
        self.head = nn.Linear(feat_dim, num_classes)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear or nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm or nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, labels=None):
        """
        Args:
            x: 输入图像 [B, 3, H, W]
            labels: 标签 [B]，用于 ArcFace 和 CosFace（训练时需要）
        
        Returns:
            如果 use_multi_head=True:
                dict: {
                    'features': [B, feat_dim],
                    'logits_softmax': [B, num_classes],
                    'logits_arcface': [B, num_classes],
                    'logits_cosface': [B, num_classes],
                    'logits_ldam': [B, num_classes]
                }
            如果 use_multi_head=False:
                [B, num_classes] - 单个分类头的 logits
        """
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        features = torch.flatten(self.avgpool(self.norm(x)), 1)
        # features = self.dropout(features)  # Dropout已注释
        
        return self.head(features)


@register_model
def starnet_s1_sa(pretrained=False, **kwargs):
    """
    StarNet S1 with Channel Attention module
    Block includes ChannelAttention module
    """
    model = StarNet(24, [2, 2, 8, 3], use_attn=None, **kwargs)
    return model


@register_model
def starnet_s1_all_grn(pretrained=False, **kwargs):
    """
    StarNet S1 variant: 所有Block开启GRN，关闭空间注意力
    """
    model = StarNet(24, [2, 2, 8, 3], use_attn=None, use_sa=False, use_grn=True, **kwargs)
    return model


@register_model
def starnet_s1_all_sa(pretrained=False, **kwargs):
    """
    StarNet S1 variant: 所有Block开启空间注意力，关闭GRN
    """
    model = StarNet(24, [2, 2, 8, 3], use_attn=None, use_sa=True, use_grn=False, **kwargs)
    return model


class StarNet_CBAM(nn.Module):
    """
    StarNet with CBAM (Convolutional Block Attention Module)
    使用 CBAMBlock，结合通道注意力和空间注意力
    """
    def __init__(self, base_dim=32, depths=[3, 3, 12, 5], mlp_ratio=4, drop_path_rate=0.0, num_classes=1000, dropout_rate=0.1, use_attn=None, use_multi_head=False, cls_num_list=None, **kwargs):
        """
        Args:
            base_dim: 基础维度
            depths: 每个stage的block数量
            mlp_ratio: MLP比例
            drop_path_rate: DropPath率
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
            # 使用 CBAMBlock 而不是 Block
            blocks = [ CBAMBlock(self.in_channel, mlp_ratio, dpr[cur + i])  for i in range(depths[i_layer]) ]
            cur += depths[i_layer]
            self.stages.append(nn.Sequential(down_sampler, *blocks))
        # head
        self.norm = nn.BatchNorm2d(self.in_channel)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Identity()
        
        # 特征维度
        feat_dim = self.in_channel
    
        # 单分类头模式
        self.head = nn.Linear(feat_dim, num_classes)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear or nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm or nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, labels=None):
        """
        Args:
            x: 输入图像 [B, 3, H, W]
            labels: 标签 [B]，用于 ArcFace 和 CosFace（训练时需要）
        
        Returns:
            [B, num_classes] - 单个分类头的 logits
        """
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        features = torch.flatten(self.avgpool(self.norm(x)), 1)
        
        return self.head(features)


@register_model
def starnet_s1_cbam(pretrained=False, **kwargs):
    """
    StarNet S1 with CBAM (Convolutional Block Attention Module)
    Block uses CBAM which combines Channel Attention and Spatial Attention
    """
    model = StarNet_CBAM(24, [2, 2, 8, 3], use_attn=None, **kwargs)
    return model


class StarNet_ParallelAttention(nn.Module):
    """
    StarNet with Parallel Attention
    使用 ParallelAttentionBlock，通道注意力和空间注意力并行处理然后融合
    """
    def __init__(self, base_dim=32, depths=[3, 3, 12, 5], mlp_ratio=4, drop_path_rate=0.0, num_classes=1000, dropout_rate=0.1, use_attn=None, use_multi_head=False, cls_num_list=None, fusion='add', **kwargs):
        """
        Args:
            base_dim: 基础维度
            depths: 每个stage的block数量
            mlp_ratio: MLP比例
            drop_path_rate: DropPath率
            fusion: 并行注意力的融合方式，'add'（相加）、'multiply'（相乘）或'concat'（拼接）
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
            # 使用 ParallelAttentionBlock
            blocks = [ ParallelAttentionBlock(self.in_channel, mlp_ratio, dpr[cur + i], fusion=fusion)  for i in range(depths[i_layer]) ]
            cur += depths[i_layer]
            self.stages.append(nn.Sequential(down_sampler, *blocks))
        # head
        self.norm = nn.BatchNorm2d(self.in_channel)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Identity()
        
        # 特征维度
        feat_dim = self.in_channel
    
        # 单分类头模式
        self.head = nn.Linear(feat_dim, num_classes)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear or nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm or nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, labels=None):
        """
        Args:
            x: 输入图像 [B, 3, H, W]
            labels: 标签 [B]，用于 ArcFace 和 CosFace（训练时需要）
        
        Returns:
            [B, num_classes] - 单个分类头的 logits
        """
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        features = torch.flatten(self.avgpool(self.norm(x)), 1)
        
        return self.head(features)


@register_model
def starnet_s1_parallel_attn(pretrained=False, fusion='add', **kwargs):
    """
    StarNet S1 with Parallel Attention
    Block uses Parallel Attention which combines Channel Attention and Spatial Attention in parallel
    
    Args:
        pretrained: 是否加载预训练权重
        fusion: 融合方式，'add'（相加）、'multiply'（相乘）或'concat'（拼接），默认'add'
        **kwargs: 其他参数
    """
    model = StarNet_ParallelAttention(24, [2, 2, 8, 3], use_attn=None, fusion=fusion, **kwargs)
    return model



