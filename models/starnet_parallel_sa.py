import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model

# ==========================================
# 1. 优化后的空间注意力 (Cleaner & More Efficient)
# ==========================================
class SpatialAttentionBlock(nn.Module):
    """
    优化点：
    1. 代码逻辑清理。
    2. 明确卷积为由两通道到一通道的映射，用于生成空间掩码。
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttentionBlock, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        # 这里使用普通的Conv2d即可，因为输入通道只有2 (Max+Avg)，计算量极小，不需要DW-Conv
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        
        # 1. 压缩通道维度：提取空间结构信息
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        
        # 2. 拼接
        x_cat = torch.cat([avg_out, max_out], dim=1)   # [B, 2, H, W]
        
        # 3. 生成空间注意力图 [B, 1, H, W]
        attn_map = self.sigmoid(self.conv1(x_cat))
        
        # 4. 广播机制相乘
        return x * attn_map

# ==========================================
# 2. 基础组件 (ConvBN)
# ==========================================
class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups=groups, bias=False))
        if with_bn:
            self.add_module('bn', nn.BatchNorm2d(out_planes))
            nn.init.constant_(self.bn.weight, 1)
            nn.init.constant_(self.bn.bias, 0)

# ==========================================
# 3. StarNet Block (保持不变)
# ==========================================
class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x

# ==========================================
# 4. 优化后的并行空间分支 (Efficient & Aligned)
# ==========================================
class ParallelSpatialBranch(nn.Module):
    """
    优化点：
    1. 下采样策略：针对大 stride，使用 Depthwise Separable Conv 减少参数量。
    2. 特征对齐：在输出前加入 1x1 Conv，帮助将原始图像特征映射到深层特征空间。
    """
    def __init__(self, out_channels, total_stride):
        super().__init__()
        
        # 优化1：如果 stride 很大（>=8），直接用一个 Conv 可能会丢失太多信息且参数量大。
        # 这里我们采用两步走：先用 DW-Conv 下采样，再用 PW-Conv 升维。
        
        self.downsample = nn.Sequential(
            # 深度卷积负责下采样空间信息 (groups=3, 独立处理RGB)
            nn.Conv2d(3, 3, kernel_size=3, stride=total_stride, padding=1, groups=3, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU6(),
            
            # 点卷积负责将通道数从 3 映射到 out_channels
            nn.Conv2d(3, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6()
        )
        
        # 空间注意力
        self.spatial_attn = SpatialAttentionBlock(kernel_size=7)
        
        # 优化2：特征对齐 (Channel Mixer)
        # 在融合前，再次通过一个 1x1 卷积整理通道信息，使其更适合与主干特征相加
        self.align_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # 可学习的缩放因子 (Zero Init)
        self.scale = nn.Parameter(torch.zeros(1))
    
    def forward(self, raw_image):
        # 1. 高效下采样 + 升维
        x = self.downsample(raw_image)
        
        # 2. 空间注意力增强
        x = self.spatial_attn(x)
        
        # 3. 特征对齐
        x = self.align_conv(x)
        
        # 4. 缩放
        return x * self.scale

# ==========================================
# 5. StarNet 主体结构
# ==========================================
class StarNet_ParallelSA(nn.Module):
    def __init__(self, base_dim=32, depths=[3, 3, 12, 5], mlp_ratio=4, drop_path_rate=0.0, num_classes=1000, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.in_channel = 32
        
        # Stem
        self.stem = nn.Sequential(ConvBN(3, self.in_channel, kernel_size=3, stride=2, padding=1), nn.ReLU6())
        current_stride = 2
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        self.stages = nn.ModuleList()
        self.parallel_branches = nn.ModuleList()
        
        cur = 0
        for i_layer in range(len(depths)):
            embed_dim = base_dim * 2 ** i_layer
            
            # Stage Downsampler
            down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1)
            current_stride *= 2 
            
            # Stage Blocks
            blocks = [Block(embed_dim, mlp_ratio, dpr[cur + i]) for i in range(depths[i_layer])]
            self.stages.append(nn.Sequential(down_sampler, *blocks))
            
            # Parallel Branch (Note: input is raw image)
            # 这里的 total_stride 需要匹配当前 stage 输出的特征图尺寸
            parallel_branch = ParallelSpatialBranch(out_channels=embed_dim, total_stride=current_stride)
            self.parallel_branches.append(parallel_branch)
            
            self.in_channel = embed_dim
            cur += depths[i_layer]
            
        # Head
        self.norm = nn.BatchNorm2d(self.in_channel)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(self.in_channel, num_classes)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        raw_input = x
        
        # Stem
        x_main = self.stem(x)
        
        # Loop Stages
        for stage, parallel_branch in zip(self.stages, self.parallel_branches):
            # Main Branch
            x_main = stage(x_main)
            
            # Parallel Spatial Branch (Always from raw input)
            x_spatial = parallel_branch(raw_input)
            
            # Fusion
            x_main = x_main + x_spatial
            
        # Head
        x = self.norm(x_main)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.head(x)

@register_model
def starnet_s1_parallel_sa(pretrained=False, **kwargs):
    model = StarNet_ParallelSA(24, [2, 2, 8, 3], **kwargs)
    return model