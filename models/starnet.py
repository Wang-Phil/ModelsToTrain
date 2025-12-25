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


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0., with_attn=False, use_sa=True, use_grn=True):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # 空间注意力和GRN开关
        mid_dim = int(dim * mlp_ratio)
        self.sa = SpatialAttention(kernel_size=7) if use_sa else nn.Identity()
        self.grn = GRN(mid_dim) if use_grn else nn.Identity()

    def forward(self, x):
        input = x
        x = self.sa(x)
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.grn(x)
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x

# --- 1. 完整的 Selective Kernel 单元 ---
class CompleteSKUnit(nn.Module):
    """
    严格遵循 SKNet 的 Split -> Fuse -> Select 流程
    """
    def __init__(self, dim, kernel_sizes=[3, 9], reduction=4):
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
    def __init__(self, dim, mlp_ratio=3, drop_path=0., kernel_sizes=[3, 7], use_sa=True, use_grn=True):
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
        # 空间注意力和GRN开关
        # self.sa = SpatialAttention(kernel_size=7) if use_sa else nn.Identity()
        # self.grn = GRN(mid_dim) if use_grn else nn.Identity() # 在高维空间做 GRN 效果最好
        self.g = ConvBN(mid_dim, dim, 1, with_bn=True) # 降维回归
        
        # 最后的深度卷积（可选，增强局部性）
        self.dw_final = ConvBN(dim, dim, 7, 1, 3, groups=dim, with_bn=False)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

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


class StarNet(nn.Module):
    def __init__(self, base_dim=32, depths=[3, 3, 12, 5], mlp_ratio=4, drop_path_rate=0.0, num_classes=1000, dropout_rate=0.1, use_attn=None, use_multi_head=False, cls_num_list=None, sk_kernel_sizes=[3, 7], use_sa=True, use_grn=True, **kwargs):
        """
        Args:
            base_dim: 基础维度
            depths: 每个stage的block数量
            mlp_ratio: MLP比例
            drop_path_rate: DropPath率
            sk_kernel_sizes: SKStarBlock中使用的kernel sizes组合，默认[3, 7]
                            前3个stage使用Block，最后一个stage使用SKStarBlock
            use_sa: 是否使用空间注意力，默认True
            use_grn: 是否使用GRN，默认True
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
            # 默认前3个stage使用Block，最后一个stage使用SKStarBlock
            if i_layer == len(depths) - 1:
                # 最后一个stage使用SKStarBlock
                BlockType = SKStarBlock
                blocks = [
                    BlockType(self.in_channel, mlp_ratio, dpr[cur + i], kernel_sizes=sk_kernel_sizes, use_sa=use_sa, use_grn=use_grn) 
                    for i in range(depths[i_layer])
                ]
            else:
                # 前3个stage使用Block
                BlockType = Block
                blocks = [
                    BlockType(self.in_channel, mlp_ratio, dpr[cur + i], use_sa=use_sa, use_grn=use_grn) 
                    for i in range(depths[i_layer])
                ]
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
    
        # 单分类头模式（向后兼容）
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
def starnet_s1(pretrained=False, **kwargs):
    model = StarNet(24, [2, 2, 8, 3], use_attn=None, sk_kernel_sizes=[3, 9], **kwargs)
    return model


@register_model
def starnet_s2(pretrained=False, **kwargs):
    model = StarNet(24, [2, 2, 8, 3], use_attn=1, sk_kernel_sizes=[3, 7], **kwargs)
    return model


@register_model
def starnet_s3(pretrained=False, **kwargs):
    model = StarNet(24, [2, 2, 8, 3], use_attn=2, sk_kernel_sizes=[3, 7], **kwargs)
    return model


@register_model
def starnet_s4(pretrained=False, **kwargs):
    model = StarNet(24, [2, 2, 8, 3], use_attn=3, sk_kernel_sizes=[3, 7], **kwargs)
    return model


# 不同kernel_sizes组合的模型变体
@register_model
def starnet_s1_sk35(pretrained=False, **kwargs):
    """StarNet S1 with SK kernel sizes [3, 5]"""
    model = StarNet(24, [2, 2, 8, 3], use_attn=None, sk_kernel_sizes=[3, 5], **kwargs)
    return model


@register_model
def starnet_s1_sk37(pretrained=False, **kwargs):
    """StarNet S1 with SK kernel sizes [3, 7] (same as starnet_s1)"""
    model = StarNet(24, [2, 2, 8, 3], use_attn=None, sk_kernel_sizes=[3, 7], **kwargs)
    return model


@register_model
def starnet_s1_sk57(pretrained=False, **kwargs):
    """StarNet S1 with SK kernel sizes [5, 7]"""
    model = StarNet(24, [2, 2, 8, 3], use_attn=None, sk_kernel_sizes=[5, 7], **kwargs)
    return model


@register_model
def starnet_s1_sk13(pretrained=False, **kwargs):
    """StarNet S1 with SK kernel sizes [1, 3]"""
    model = StarNet(24, [2, 2, 8, 3], use_attn=None, sk_kernel_sizes=[1, 3], **kwargs)
    return model


@register_model
def starnet_s1_sk15(pretrained=False, **kwargs):
    """StarNet S1 with SK kernel sizes [1, 5]"""
    model = StarNet(24, [2, 2, 8, 3], use_attn=None, sk_kernel_sizes=[1, 5], **kwargs)
    return model


@register_model
def starnet_s1_sk17(pretrained=False, **kwargs):
    """StarNet S1 with SK kernel sizes [1, 7]"""
    model = StarNet(24, [2, 2, 8, 3], use_attn=None, sk_kernel_sizes=[1, 7], **kwargs)
    return model


@register_model
def starnet_s1_sk19(pretrained=False, **kwargs):
    """StarNet S1 with SK kernel sizes [1, 9]"""
    model = StarNet(24, [2, 2, 8, 3], use_attn=None, sk_kernel_sizes=[1, 9], **kwargs)
    return model


@register_model
def starnet_s1_sk39(pretrained=False, **kwargs):
    """StarNet S1 with SK kernel sizes [3, 9]"""
    model = StarNet(24, [2, 2, 8, 3], use_attn=None, sk_kernel_sizes=[3, 9],use_sa=True, use_grn=True,**kwargs)
    return model


@register_model
def starnet_s1_sk59(pretrained=False, **kwargs):
    """StarNet S1 with SK kernel sizes [5, 9]"""
    model = StarNet(24, [2, 2, 8, 3], use_attn=None, sk_kernel_sizes=[5, 9], **kwargs)
    return model


@register_model
def starnet_s1_sk79(pretrained=False, **kwargs):
    """StarNet S1 with SK kernel sizes [7, 9]"""
    model = StarNet(24, [2, 2, 8, 3], use_attn=None, sk_kernel_sizes=[7, 9], **kwargs)
    return model


# very small networks #
@register_model
def starnet_s050(pretrained=False, **kwargs):
    return StarNet(16, [1, 1, 3, 1], 3, **kwargs)


@register_model
def starnet_s100(pretrained=False, **kwargs):
    return StarNet(20, [1, 2, 4, 1], 4, **kwargs)


@register_model
def starnet_s150(pretrained=False, **kwargs):
    return StarNet(24, [1, 2, 4, 2], 3, **kwargs)


# 消融实验模型：只使用GRN，不使用空间注意力
@register_model
def starnet_s1_grn_only(pretrained=False, **kwargs):
    """
    StarNet S1 variant: 所有Block开启GRN，关闭空间注意力
    用于消融实验，验证GRN的作用
    """
    model = StarNet(24, [2, 2, 8, 3], use_attn=None, sk_kernel_sizes=[3, 9], use_sa=False, use_grn=True, **kwargs)
    return model


# 消融实验模型：只使用空间注意力，不使用GRN
@register_model
def starnet_s1_sa_only(pretrained=False, **kwargs):
    """
    StarNet S1 variant: 所有Block开启空间注意力，关闭GRN
    用于消融实验，验证空间注意力的作用
    """
    model = StarNet(24, [2, 2, 8, 3], use_attn=None, sk_kernel_sizes=[3, 9], use_sa=True, use_grn=False, **kwargs)
    return model