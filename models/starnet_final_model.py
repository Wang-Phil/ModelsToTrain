"""
StarNet 空间注意力变体模型
根据空间注意力在不同stage的使用情况，创建4个不同的模型变体：
- starnet_sa_s1: 所有stage都加空间注意力 (stage 0,1,2,3)
- starnet_sa_s2: 第一个stage不加注意力 (stage 1,2,3加注意力)
- starnet_sa_s3: 前两个stage不加注意力 (stage 2,3加注意力)
- starnet_sa_s4: 前三个stage不加注意力 (只有stage 3加注意力)
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


# class ConvBN(torch.nn.Sequential):
#     def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
#         super().__init__()
#         self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
#         if with_bn:
#             self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
#             torch.nn.init.constant_(self.bn.weight, 1)
#             torch.nn.init.constant_(self.bn.bias, 0)

# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.bn = nn.GroupNorm(1, 1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         # 1. 沿着通道维度做 AvgPool 和 MaxPool
#         # x: [B, C, H, W] -> avg_out: [B, 1, H, W], max_out: [B, 1, H, W]
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
        
#         # 2. 拼接
#         x_cat = torch.cat([avg_out, max_out], dim=1)
        
#         # 3. 卷积 + BN + Sigmoid 生成空间掩码
#         scale = self.sigmoid(self.bn(self.conv1(x_cat)))
        
#         # 4. 施加注意力
#         return x * scale

# class Block(nn.Module):
#     def __init__(self, dim, mlp_ratio=3, drop_path=0., with_attn=False):
#         super().__init__()
#         self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
#         self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
#         self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
#         self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
#         self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
#         self.act = nn.ReLU6()
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         # 3. 空间注意力模块（已注释）
#         self.with_attn = with_attn
#         self.sa = SpatialAttention(kernel_size=7)

#     def forward(self, x):
#         input = x
#         x = self.dwconv(x)
#         x1, x2 = self.f1(x), self.f2(x)
#         x = self.act(x1) * x2
#         x = self.dwconv2(self.g(x))
#         x = self.sa(x)
#         # [修改] 在 DropPath 和残差连接之前应用注意力（已注释）
#         # 这让网络在把特征加回主干之前，先"提炼"一次特征
#         # [修正] 只有开启开关才进行注意力计算
#         x = input + self.drop_path(x)
#         return x


# class StarNet_SA(nn.Module):
#     """
#     StarNet Model with configurable Spatial Attention across stages
    
#     Args:
#         base_dim: 基础维度
#         depths: 每个stage的block数量
#         mlp_ratio: MLP扩展比例
#         drop_path_rate: Drop path rate
#         num_classes: 分类类别数
#         sa_stages: 哪些stage使用空间注意力，例如 [0,1,2,3] 表示所有stage都使用
#     """
#     def __init__(self, base_dim=32, depths=[3, 3, 12, 5], mlp_ratio=4, drop_path_rate=0.0, num_classes=1000, dropout_rate=0.1, use_attn=None, **kwargs):
#         """
#         Args:
#             base_dim: 基础维度
#             depths: 每个stage的block数量
#             mlp_ratio: MLP比例
#             drop_path_rate: DropPath率
#             num_classes: 分类类别数
#             dropout_rate: Dropout比例（默认0.1，设置为0禁用Dropout）
#         """
#         super().__init__()
#         self.num_classes = num_classes
#         self.in_channel = 32
#         # stem layer
#         self.stem = nn.Sequential(ConvBN(3, self.in_channel, kernel_size=3, stride=2, padding=1),nn.ReLU6())
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] # stochastic depth
#         # build stages
#         self.stages = nn.ModuleList()
#         cur = 0
#         for i_layer in range(len(depths)):
#             embed_dim = base_dim * 2 ** i_layer
#             down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1)
#             self.in_channel = embed_dim
#             # 空间注意力机制配置
#             # use_attn=0: 所有stage都使用 (stage 0,1,2,3)
#             # use_attn=1: 从stage 1开始使用 (stage 1,2,3)
#             # use_attn=2: 从stage 2开始使用 (stage 2,3)
#             # use_attn=3: 只有stage 3使用
#             use_attn_here = False
#             if use_attn is not None:
#                 if use_attn == 0:  # 所有stage都使用
#                     use_attn_here = True
#                 elif i_layer >= use_attn:  # 从指定stage开始使用
#                     use_attn_here = True
#             blocks = [
#                 Block(self.in_channel, mlp_ratio, dpr[cur + i], with_attn=use_attn_here) 
#                 for i in range(depths[i_layer])
#             ]
#             cur += depths[i_layer]
#             self.stages.append(nn.Sequential(down_sampler, *blocks))
#         # head
#         self.norm = nn.BatchNorm2d(self.in_channel)
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         # Dropout已注释
#         # if dropout_rate > 0:
#         #     self.dropout = nn.Dropout(dropout_rate)
#         # else:
#         self.dropout = nn.Identity()
#         self.head = nn.Linear(self.in_channel, num_classes)
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, (nn.Linear, nn.Conv2d)):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)

#     def forward(self, x):
#         x = self.stem(x)
#         for stage in self.stages:
#             x = stage(x)
#         x = torch.flatten(self.avgpool(self.norm(x)), 1)
#         # x = self.dropout(x)  # Dropout已注释
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

class CrossStarBlock(nn.Module):
    """
    [D - 基线] Inception-style Cross-Star Block
    实现了 Y = Concat((x_{3A} * x_{7B}), (x_{7A} * x_{3B}))
    交叉星乘：局部细节调制全局语境，全局语境校正局部细节
    """
    def __init__(self, dim, mlp_ratio=3, drop_path=0.,with_attn=False):
        super().__init__()
        
        # 保持原有的输入处理 (7x7 Depthwise Conv)
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        
        # 定义中间维度。总扩展维度是 dim * mlp_ratio，交叉乘积需要平分通道
        self.mid_dim = (dim * mlp_ratio) // 2
        
        # --- Multi-Scale Branches for Cross-Star Operation ---
        
        # Branch 1 (Local): 3x3 Conv (捕捉局部细节)
        self.f3 = ConvBN(dim, self.mid_dim, 3, 1, 1, with_bn=False) # Conv_{3x3}
        
        # Branch 2 (Context): 7x7 Conv (捕捉全局语境和纹理)
        self.f7 = ConvBN(dim, self.mid_dim, 7, 1, 3, with_bn=False) # Conv_{7x7}
        
        # 融合与输出
        # 注意：如果只使用 y12（不concat y21），输入通道数应该是 mid_dim 而不是 dim * mlp_ratio
        self.g = ConvBN(self.mid_dim, dim, 1, with_bn=True) # 输入通道数改为 mid_dim
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.sa = SpatialAttention(kernel_size=7)
        # 为每个分支创建独立的 GRN（维度为 self.mid_dim）
        self.grn_y12 = GRN(dim=self.mid_dim)  # 用于 y12 分支
        self.grn_y21 = GRN(dim=self.mid_dim)  # 用于 y21 分支

        
    def forward(self, x):
        input = x
        x = self.sa(x)
        x = self.dwconv(x)
        
        # 1. 计算两个分支的特征（共享参数）
        x_3 = self.f3(x)
        x_7 = self.f7(x)

        
        # 2. 交叉星乘 (Cross-Star Operation) - D (基线)
        # 乘法 1: Point-wise (1A) 调制 Local (3B) -> 强调点卷积对局部细节的调制
        y12 = self.act(x_3) * x_7
        x_out = self.grn_y12(y12)
        
        # 4. 投影回输入维度
        x_out = self.dwconv2(self.g(x_out))
        x_out = input + self.drop_path(x_out)
        return x_out


class SelectiveKernelBlock(nn.Module):
    """
    通用自适应选择性卷积块 (Selective Kernel Block)
    支持任意数量的多尺度分支，动态选择最佳的感受野尺度
    
    Args:
        dim: 输入维度
        mlp_ratio: MLP扩展比例
        drop_path: DropPath率
        with_attn: 是否使用空间注意力（暂未实现）
        kernel_sizes: 卷积核尺寸列表，例如 [3, 7] 表示使用 3x3 和 7x7 两个分支
        use_dilation: 是否使用空洞卷积实现大感受野（True）或直接使用大卷积核（False）
    """
    def __init__(self, dim, mlp_ratio=3, drop_path=0., with_attn=False, 
                 kernel_sizes=None, use_dilation=True):
        super().__init__()
        # 使用 None 作为默认值，避免可变默认参数的问题
        if kernel_sizes is None:
            kernel_sizes = [3, 7]
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.num_branches = len(kernel_sizes)
        self.kernel_sizes = kernel_sizes
        
        # 1. 通道降维比例 (用于注意力计算的 bottleneck)
        reduction = 4
        hidden_dim = max(dim // reduction, 32)

        # 2. 定义多尺度分支 (全部使用 DWConv 以节省参数)
        self.branches = nn.ModuleList()
        for ks in kernel_sizes:
            if use_dilation and ks > 3:
                # 使用空洞卷积实现大感受野（更节省参数）
                # 3x3 dilation=1 -> 感受野 3x3
                # 3x3 dilation=2 -> 感受野 5x5
                # 3x3 dilation=3 -> 感受野 7x7
                # 3x3 dilation=4 -> 感受野 9x9
                dilation = (ks - 1) // 2
                padding = dilation
                branch = nn.Sequential(
                    nn.Conv2d(dim, dim, 3, padding=padding, dilation=dilation, groups=dim, bias=False),
                    nn.BatchNorm2d(dim),
                    nn.ReLU6()
                )
            else:
                # 直接使用指定大小的卷积核
                padding = ks // 2
                branch = nn.Sequential(
                    nn.Conv2d(dim, dim, ks, padding=padding, groups=dim, bias=False),
                    nn.BatchNorm2d(dim),
                    nn.ReLU6()
                )
            self.branches.append(branch)

        # 3. 自适应注意力生成器 (Adaptive Attention)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(),
            nn.Conv2d(hidden_dim, dim * self.num_branches, 1, bias=False) # 输出所有分支的权重
        )
        self.softmax = nn.Softmax(dim=1)

        # 4. 后续处理 (FFN / MLP)
        self.proj = ConvBN(dim, dim, 1, with_bn=True) # 融合一下
        
        # FFN 部分
        self.mlp = nn.Sequential(
            ConvBN(dim, int(dim * mlp_ratio), 1, with_bn=False),
            nn.ReLU6(),
            ConvBN(int(dim * mlp_ratio), dim, 1, with_bn=True)
        )

        self.sa = SpatialAttention(kernel_size=7)
        self.grn = GRN(dim=dim)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        # --- SK Unit Start ---
        # 1. 并行计算所有尺度的特征
        x = self.sa(x)
        branch_features = []
        for branch in self.branches:
            branch_features.append(branch(x))
        
        # 2. 特征相加，计算全局描述符
        feat_sum = sum(branch_features)
        attn_vec = self.gap(feat_sum) # [B, C, 1, 1]
        
        # 3. 生成权重 [B, num_branches*C, 1, 1] -> 变形为 [B, num_branches, C, 1, 1]
        attn_weights = self.fc(attn_vec)
        B, _, H, W = x.shape
        attn_weights = attn_weights.view(B, self.num_branches, self.dim, 1, 1)
        attn_weights = self.softmax(attn_weights) # 在所有分支间归一化
        
        # 4. 加权融合 (Soft Attention Fusion)
        x_sk = sum(attn_weights[:, i] * branch_features[i] for i in range(self.num_branches))
        # --- SK Unit End ---
        
        x = self.proj(x_sk)
        x = self.grn(x)
        # FFN
        x = x + self.drop_path(self.mlp(x))

        x = input + self.drop_path(x)
        return x


class LightSKBlock(nn.Module):
    """
    轻量级自适应选择性卷积块 (Lightweight Selective Kernel Block)
    核心功能：动态选择最佳的感受野尺度 (1x1 vs 3x3)
    使用最佳组合：1x1 + 3x3 双分支
    """
    def __init__(self, dim, mlp_ratio=3, drop_path=0., with_attn=False):
        super().__init__()
        # 使用最佳组合：1x1 + 3x3
        self.sk_block = SelectiveKernelBlock(
            dim=dim, 
            mlp_ratio=mlp_ratio, 
            drop_path=drop_path, 
            with_attn=with_attn,
            kernel_sizes=[1, 3],
            use_dilation=True
        )

    def forward(self, x):
        return self.sk_block(x)



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


class SKDWConv(nn.Module):
    """
    SK-style Multi-Scale Depthwise Convolution
    Designed for StarNet
    """
    def __init__(self, dim, kernel_sizes=(3, 5, 7), use_dilation=True, reduction=4):
        super().__init__()
        self.dim = dim
        self.num_branches = len(kernel_sizes)

        # 多尺度 DWConv 分支
        self.branches = nn.ModuleList()
        for ks in kernel_sizes:
            if use_dilation and ks > 3:
                dilation = (ks - 1) // 2
                padding = dilation
                self.branches.append(
                    nn.Sequential(
                        nn.Conv2d(dim, dim, 3, padding=padding,
                                  dilation=dilation, groups=dim, bias=False),
                        nn.BatchNorm2d(dim),
                        nn.ReLU6()
                    )
                )
            else:
                padding = ks // 2
                self.branches.append(
                    nn.Sequential(
                        nn.Conv2d(dim, dim, ks, padding=padding,
                                  groups=dim, bias=False),
                        nn.BatchNorm2d(dim),
                        nn.ReLU6()
                    )
                )

        # SK Attention（轻量化）
        hidden_dim = max(dim // reduction, 32)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, bias=False),
            nn.ReLU6(),
            nn.Conv2d(hidden_dim, dim * self.num_branches, 1, bias=False)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 并行多尺度特征
        feats = [b(x) for b in self.branches]

        # 全局描述
        feat_sum = sum(feats)
        attn = self.gap(feat_sum)
        attn = self.fc(attn)

        B, _, _, _ = attn.shape
        attn = attn.view(B, self.num_branches, self.dim, 1, 1)
        attn = self.softmax(attn)

        # 加权融合
        out = sum(attn[:, i] * feats[i] for i in range(self.num_branches))
        return out



class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0., with_attn=False):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # 3. 空间注意力模块（已注释）
        # self.with_attn = with_attn
        # self.sa = SpatialAttention(kernel_size=7)
        # mid_dim = mlp_ratio * dim
        # self.grn = GRN(dim=mid_dim)

    def forward(self, x):
        input = x
        # if self.with_attn:
        # x = self.sa(x)
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        # x = self.grn(x)
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
            # 3x3 (dil=1) -> 3x3
            # 3x3 (dil=2) -> 5x5
            # 3x3 (dil=3) -> 7x7
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
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
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
        self.sk_unit = CompleteSKUnit(dim, kernel_sizes=[3, 9])
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

        # self.sa = SpatialAttention(kernel_size=7)

    def forward(self, x):
        input = x
        # x = self.sa(x)
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
        # x = self.grn(x)
        x = self.g(x)
        x = self.dw_final(x)
        
        x = input + self.drop_path(x)
        return x

class StarNet_FINAL(nn.Module):
    def __init__(self, base_dim=32, depths=[3, 3, 12, 5], mlp_ratio=4, drop_path_rate=0.0, num_classes=1000, dropout_rate=0.1, use_attn=None, use_multi_head=False, cls_num_list=None, sk_kernel_sizes=None, sk_start_layer=None, **kwargs):
        """
        Args:
            base_dim: 基础维度
            depths: 每个stage的block数量
            mlp_ratio: MLP比例
            drop_path_rate: DropPath率
            num_classes: 分类类别数
            dropout_rate: Dropout比例（默认0.1，设置为0禁用Dropout）
            use_attn: 空间注意力使用策略
            use_multi_head: 是否使用多分类头（ArcFace, CosFace, LDAM, Softmax）
            cls_num_list: 类别数量列表（用于 LDAM），如果为 None 则使用均匀分布
            sk_kernel_sizes: SK Block 的卷积核尺寸列表，例如 [3, 7] 表示使用 3x3 和 7x7 两个分支
                            如果为 None，则使用默认的 LightSKBlock (3x3 + 7x7)
            sk_start_layer: 从哪个layer开始使用Block_SK
                           0: 所有layer都使用Block_SK (i_layer 0,1,2,3)
                           1: 从layer 1开始使用Block_SK (i_layer 1,2,3)
                           2: 从layer 2开始使用Block_SK (i_layer 2,3)
                           3: 只有layer 3使用Block_SK (i_layer 3)
                           None: 使用默认策略 (i_layer < 3 使用Block, i_layer >= 3 使用Block_SK)
        """
        super().__init__()
        self.num_classes = num_classes
        self.use_multi_head = use_multi_head
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
            # --- Strategy: Hybrid Block Usage ---
            if i_layer < 3:
                BlockType = Block
            else:
                BlockType = SKStarBlock

            blocks = [
                BlockType(self.in_channel, mlp_ratio, dpr[cur + i]) 
                for i in range(depths[i_layer])
            ]
            # blocks = [Block(self.in_channel, mlp_ratio, dpr[cur + i]) for i in range(depths[i_layer])]
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
        
        if use_multi_head:
            # 多分类头模式：ArcFace, CosFace, LDAM, Softmax
            self.head_softmax = nn.Linear(feat_dim, num_classes)
            self.head_arcface = ArcFace(feat_dim, num_classes, s=30.0, m=0.5)
            self.head_cosface = CosFace(feat_dim, num_classes, s=30.0, m=0.35)
            self.head_ldam = nn.Linear(feat_dim, num_classes)
            # 保存类别数量列表用于 LDAM Loss
            if cls_num_list is None:
                # 默认均匀分布
                self.cls_num_list = [1.0] * num_classes
            else:
                self.cls_num_list = cls_num_list
        else:
            # 单分类头模式（向后兼容）
            self.head = nn.Linear(feat_dim, num_classes)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
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
        
        if self.use_multi_head:
            # 多分类头模式
            logits_softmax = self.head_softmax(features)
            logits_arcface = self.head_arcface(features, labels)
            logits_cosface = self.head_cosface(features, labels)
            logits_ldam = self.head_ldam(features)
            
            return {
                'features': features,
                'logits_softmax': logits_softmax,
                'logits_arcface': logits_arcface,
                'logits_cosface': logits_cosface,
                'logits_ldam': logits_ldam
            }
        else:
            # 单分类头模式（向后兼容）
            return self.head(features)

@register_model
def starnet_s1_final(pretrained=False, **kwargs):
    """
    StarNet SA S1: 所有stage都加空间注意力 (stage 0,1,2,3)
    参数量: 约 2.9M (基于 S1 配置)
    """
    model = StarNet_FINAL(
        base_dim=24, 
        depths=[2, 2, 8, 3], 
        **kwargs
    )
    # 注意：SK 变体模型暂不支持预训练权重
    # if pretrained:
    #     url = model_urls['starnet_s1']
    #     checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
    #     model.load_state_dict(checkpoint["state_dict"], strict=False)
    return model


# ============================================
# SKNet 模型变体（仅保留最佳组合用于向后兼容）
# ============================================

@register_model
def starnet_sk_1_3(pretrained=False, **kwargs):
    """
    StarNet with SK Block: 1x1 + 3x3 双分支 (最佳组合)
    参数量: 约 2.9M (基于 S1 配置)
    注意：此模型与 starnet_s1_final 使用相同的配置，保留用于向后兼容
    """
    model = StarNet_FINAL(
        base_dim=24,
        depths=[2, 2, 8, 3],
        use_attn=0,
        **kwargs
    )
    return model


# ============================================
# 消融实验：不同i_layer上使用Block_SK的变体
# ============================================

@register_model
def starnet_sk_ablation_all(pretrained=False, **kwargs):
    """
    消融实验1：所有layer都使用Block_SK (i_layer 0,1,2,3)
    参数量: 约 2.9M (基于 S1 配置)
    """
    model = StarNet_FINAL(
        base_dim=24,
        depths=[2, 2, 8, 3],
        use_attn=0,
        sk_start_layer=0,  # 从layer 0开始使用Block_SK（即所有layer都使用）
        **kwargs
    )
    return model


@register_model
def starnet_sk_ablation_last3(pretrained=False, **kwargs):
    """
    消融实验2：后面三个layer使用Block_SK (i_layer 1,2,3)
    参数量: 约 2.9M (基于 S1 配置)
    """
    model = StarNet_FINAL(
        base_dim=24,
        depths=[2, 2, 8, 3],
        use_attn=0,
        sk_start_layer=1,  # 从layer 1开始使用Block_SK（后面三个layer）
        **kwargs
    )
    return model


@register_model
def starnet_sk_ablation_last2(pretrained=False, **kwargs):
    """
    消融实验3：后面两个layer使用Block_SK (i_layer 2,3)
    参数量: 约 2.9M (基于 S1 配置)
    """
    model = StarNet_FINAL(
        base_dim=24,
        depths=[2, 2, 8, 3],
        use_attn=0,
        sk_start_layer=2,  # 从layer 2开始使用Block_SK（后面两个layer）
        **kwargs
    )
    return model


@register_model
def starnet_sk_ablation_last1(pretrained=False, **kwargs):
    """
    消融实验4：只有最后一个layer使用Block_SK (i_layer 3)
    参数量: 约 2.9M (基于 S1 配置)
    """
    model = StarNet_FINAL(
        base_dim=24,
        depths=[2, 2, 8, 3],
        use_attn=0,
        sk_start_layer=3,  # 只有layer 3使用Block_SK
        **kwargs
    )
    return model



