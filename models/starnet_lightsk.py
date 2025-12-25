"""
 LightSKBlock only (no attention, no GRN)
简化版本：只使用选择性卷积核机制，移除空间注意力和GRN
"""

import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model


class ConvBN(torch.nn.Sequential):
    """Conv + BN 组合"""
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)


class SelectiveKernelBlock(nn.Module):
    """
    简化版选择性卷积核块 (Selective Kernel Block)
    移除空间注意力和GRN，只保留核心的SK机制
    
    Args:
        dim: 输入维度
        mlp_ratio: MLP扩展比例
        drop_path: DropPath率
        kernel_sizes: 卷积核尺寸列表，例如 [1, 3] 表示使用 1x1 和 3x3 两个分支
        use_dilation: 是否使用空洞卷积实现大感受野
    """
    def __init__(self, dim, mlp_ratio=3, drop_path=0., 
                 kernel_sizes=None, use_dilation=True):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [1, 3]
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
                # 使用空洞卷积实现大感受野
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

        # 3. 自适应注意力生成器 (Adaptive Attention) - 用于SK权重计算
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(),
            nn.Conv2d(hidden_dim, dim * self.num_branches, 1, bias=False)  # 输出所有分支的权重
        )
        self.softmax = nn.Softmax(dim=1)

        # 4. 后续处理 (FFN / MLP)
        self.proj = ConvBN(dim, dim, 1, with_bn=True)  # 融合一下
        
        # FFN 部分
        # self.mlp = nn.Sequential(
        #     ConvBN(dim, int(dim * mlp_ratio), 1, with_bn=False),
        #     nn.ReLU6(),
        #     ConvBN(int(dim * mlp_ratio), dim, 1, with_bn=True)
        # )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        # --- SK Unit Start ---
        # 1. 并行计算所有尺度的特征（不使用空间注意力）
        branch_features = []
        for branch in self.branches:
            branch_features.append(branch(x))
        
        # 2. 特征相加，计算全局描述符
        feat_sum = sum(branch_features)
        attn_vec = self.gap(feat_sum)  # [B, C, 1, 1]
        
        # 3. 生成权重 [B, num_branches*C, 1, 1] -> 变形为 [B, num_branches, C, 1, 1]
        attn_weights = self.fc(attn_vec)
        B, _, H, W = x.shape
        attn_weights = attn_weights.view(B, self.num_branches, self.dim, 1, 1)
        attn_weights = self.softmax(attn_weights)  # 在所有分支间归一化
        
        # 4. 加权融合 (Soft Attention Fusion)
        x_sk = sum(attn_weights[:, i] * branch_features[i] for i in range(self.num_branches))
        # --- SK Unit End ---
        
        x = self.proj(x_sk)

        x = input + self.drop_path(x)
        return x


class LightSKBlock(nn.Module):
    """
    轻量级自适应选择性卷积块 (Lightweight Selective Kernel Block)
    核心功能：动态选择最佳的感受野尺度 (1x1 vs 3x3)
    使用最佳组合：1x1 + 3x3 双分支
    移除空间注意力和GRN
    """
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        # 使用最佳组合：1x1 + 3x3
        self.sk_block = SelectiveKernelBlock(
            dim=dim, 
            mlp_ratio=mlp_ratio, 
            drop_path=drop_path, 
            kernel_sizes=[1, 3],
            use_dilation=True
        )

    def forward(self, x):
        return self.sk_block(x)


class LightSK(nn.Module):
    """
     - 只使用 LightSKBlock（无注意力和GRN）
    
    Args:
        base_dim: 基础维度
        depths: 每个stage的block数量
        mlp_ratio: MLP比例
        drop_path_rate: DropPath率
        num_classes: 分类类别数
        dropout_rate: Dropout比例
    """
    def __init__(self, base_dim=24, depths=[2, 2, 8, 3], mlp_ratio=3, 
                 drop_path_rate=0.0, num_classes=1000, dropout_rate=0.1, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.in_channel = 32
        
        # stem layer
        self.stem = nn.Sequential(
            ConvBN(3, self.in_channel, kernel_size=3, stride=2, padding=1), 
            nn.ReLU6()
        )
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth
        
        # build stages
        self.stages = nn.ModuleList()
        cur = 0
        for i_layer in range(len(depths)):
            embed_dim = base_dim * 2 ** i_layer
            down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1)
            self.in_channel = embed_dim
            
            # 所有stage都使用 LightSKBlock
            blocks = [
                LightSKBlock(self.in_channel, mlp_ratio, dpr[cur + i]) 
                for i in range(depths[i_layer])
            ]
            cur += depths[i_layer]
            self.stages.append(nn.Sequential(down_sampler, *blocks))
        
        # head
        self.norm = nn.BatchNorm2d(self.in_channel)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.head = nn.Linear(self.in_channel, num_classes)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = torch.flatten(self.avgpool(self.norm(x)), 1)
        x = self.dropout(x)
        return self.head(x)


@register_model
def lightsk(pretrained=False, **kwargs):
    """
    LightSKBlock only (no attention, no GRN)
    参数量: 约 2.5M (基于 base_dim=24, depths=[2,2,8,3] 配置)
    """
    model = LightSK(
        base_dim=24, 
        depths=[2, 2, 8, 3], 
        mlp_ratio=3,
        drop_path_rate=0.0,
        **kwargs
    )
    return model


@register_model
def lightsk_small(pretrained=False, **kwargs):
    """
     LightSK - Small variant
    """
    model = LightSK(
        base_dim=16, 
        depths=[2, 2, 6, 2], 
        mlp_ratio=3,
        drop_path_rate=0.0,
        **kwargs
    )
    return model


@register_model
def lightsk_base(pretrained=False, **kwargs):
    """
    LightSK - Base variant
    """
    model = LightSK(
        base_dim=32, 
        depths=[2, 2, 8, 3], 
        mlp_ratio=3,
        drop_path_rate=0.0,
        **kwargs
    )
    return model

