import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model

# ==========================================
# 1. 基础组件 (风格与效率优化)
# ==========================================
class ConvBN(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups=groups, bias=False)
        modules = [conv]
        if with_bn:
            bn = nn.BatchNorm2d(out_planes)
            nn.init.constant_(bn.weight, 1)
            nn.init.constant_(bn.bias, 0)
            modules.append(bn)
        super().__init__(*modules)

class SpatialAttentionBlock(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.bn(self.conv1(x_cat)))
        return x * scale

class Block(nn.Module):
    """
    StarNet Block 优化版：
    使用合并卷积 (Merged Conv) 提升 GPU 推理效率
    """
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        
        # 1. Depthwise
        self.dwconv = ConvBN(dim, dim, 7, 1, 3, groups=dim, with_bn=True)
        
        # 2. Star Operation (Merged Conv)
        # 优化：一次卷积生成两个分支 (f1, f2)，减少 CUDA kernel 启动次数
        self.f1_f2 = nn.Conv2d(dim, hidden_dim * 2, 1) 
        
        # 3. Output Projection
        self.g = ConvBN(hidden_dim, dim, 1, with_bn=True)
        
        # 4. Second DW
        self.dwconv2 = ConvBN(dim, dim, 7, 1, 3, groups=dim, with_bn=False)
        
        self.act = nn.GELU()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        
        # 优化后的 Star 操作
        x_star = self.f1_f2(x)
        x1, x2 = x_star.chunk(2, dim=1) # 在通道维度分割
        x = self.act(x1) * x2
        
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x

# ==========================================
# 2. 核心模块：共享图像金字塔 (结构保持不变)
# ==========================================
class SharedImagePyramid(nn.Module):
    def __init__(self, base_channels=24):
        super().__init__()
        # ... (此处保持您的原始逻辑，仅简化写法，结构无逻辑错误) ...
        # Level 0: S4
        self.stem_s4 = nn.Sequential(
            ConvBN(3, base_channels, 3, 2, 1), nn.GELU(),
            ConvBN(base_channels, base_channels, 3, 2, 1, groups=base_channels), nn.GELU(),
            ConvBN(base_channels, base_channels, 1)
        )
        # Level 1: S8
        self.layer_s8 = nn.Sequential(
            ConvBN(base_channels, base_channels, 3, 2, 1, groups=base_channels), nn.GELU(),
            ConvBN(base_channels, base_channels*2, 1)
        )
        # Level 2: S16
        self.layer_s16 = nn.Sequential(
            ConvBN(base_channels*2, base_channels*2, 3, 2, 1, groups=base_channels*2), nn.GELU(),
            ConvBN(base_channels*2, base_channels*4, 1)
        )
        # Level 3: S32
        self.layer_s32 = nn.Sequential(
            ConvBN(base_channels*4, base_channels*4, 3, 2, 1, groups=base_channels*4), nn.GELU(),
            ConvBN(base_channels*4, base_channels*8, 1)
        )

    def forward(self, x):
        feat_s4 = self.stem_s4(x)
        feat_s8 = self.layer_s8(feat_s4)
        feat_s16 = self.layer_s16(feat_s8)
        feat_s32 = self.layer_s32(feat_s16)
        return [feat_s4, feat_s8, feat_s16, feat_s32]

# ==========================================
# 3. 核心模块：金字塔适配器 (修复双重归一化问题)
# ==========================================
class PyramidFeatureAdapter(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # 1. 投影: Pyramid Ch -> Backbone Ch
        self.project = ConvBN(in_channels, out_channels, 1, with_bn=True)
        
        # 2. 空间注意力
        self.spatial_attn = SpatialAttentionBlock(kernel_size=7)
        
        # 3. 归一化 (GroupNorm/LayerNorm2d)
        self.norm = nn.GroupNorm(1, out_channels)
        
        # 优化点：移除原来冗余的 out_conv (Conv+BN)，因为前面已经有 Project ConvBN 了
        # 且 GN 后接 BN 是不推荐的。
        # 如果确实需要进一步混合，仅使用 1x1 Conv (无BN, 无Bias)
        self.out_mix = nn.Conv2d(out_channels, out_channels, 1, bias=False)

        # 4. 学习缩放系数
        self.scale = nn.Parameter(torch.ones(1) * 1e-3)
        
    def forward(self, pyramid_feat):
        # Proj
        proj = self.project(pyramid_feat)
        
        # Attn & Residual
        attn = self.spatial_attn(proj)
        fused = attn + proj
        
        # Norm & Mix
        fused = self.norm(fused)
        fused = self.out_mix(fused)
        
        return fused * self.scale

# ==========================================
# 4. StarNet with Shared Pyramid (逻辑检查无误)
# ==========================================
class StarNet_PyramidSA(nn.Module):
    def __init__(self, base_dim=32, depths=[3, 3, 12, 5], mlp_ratio=4, drop_path_rate=0.0, num_classes=1000, dropout_rate=0.1, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.in_channel = 32
        
        # Stem
        self.stem = nn.Sequential(ConvBN(3, self.in_channel, 3, 2, 1), nn.GELU())
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        self.stages = nn.ModuleList()
        self.adapters = nn.ModuleList()
        
        # Pyramid
        pyramid_base_c = 24
        self.image_pyramid = SharedImagePyramid(base_channels=pyramid_base_c)
        pyramid_channels = [pyramid_base_c * (2**i) for i in range(4)]
        
        cur = 0
        for i_layer in range(len(depths)):
            embed_dim = base_dim * 2 ** i_layer
            
            # Backbone Stage
            # 注意：down_sampler 将 stride=2 应用于主干，使其分辨率从 S2->S4, S4->S8 等
            down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1) 
            blocks = [Block(embed_dim, mlp_ratio, dpr[cur + i]) for i in range(depths[i_layer])]
            self.stages.append(nn.Sequential(down_sampler, *blocks))
            
            # Adapter
            adapter = PyramidFeatureAdapter(
                in_channels=pyramid_channels[i_layer], 
                out_channels=embed_dim
            )
            self.adapters.append(adapter)
            
            self.in_channel = embed_dim
            cur += depths[i_layer]
            
        # Head
        self.norm = nn.BatchNorm2d(self.in_channel)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.in_channel, num_classes)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        pyramid_feats = self.image_pyramid(x)
        x_main = self.stem(x)
        
        for stage, adapter, p_feat in zip(self.stages, self.adapters, pyramid_feats):
            # 1. 主干下采样 + 卷积块
            x_main = stage(x_main)
            
            # 2. 金字塔特征融合
            # 此时 x_main 和 adapter(p_feat) 的分辨率和通道数应该完全一致
            x_main = x_main + adapter(p_feat)
            
        x = self.norm(x_main)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.head(x)

@register_model
def starnet_s1_pyramid(pretrained=False, **kwargs):
    model = StarNet_PyramidSA(24, [2, 2, 8, 3], **kwargs)
    return model