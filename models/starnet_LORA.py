"""
Modified StarNet with Texture-Aware LoRA Mechanism for Medical Imaging.
Specifically designed for Femoral Head/Hip Joint Classification.

Modifications:
1. Added `TextureBranch_LoRA`: A bottleneck large-kernel branch to capture texture and suppress artifacts.
2. Added `TextureStarBlock`: Replaces the standard f2 branch with the TextureBranch.
3. Modified `StarNet`: Applies TextureStarBlock in deeper stages (stages 3 & 4).
"""

import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model

# 预训练权重下载链接 (保持原样)
model_urls = {
    "starnet_s1": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s1.pth.tar",
    "starnet_s2": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s2.pth.tar",
    "starnet_s3": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s3.pth.tar",
    "starnet_s4": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s4.pth.tar",
}

class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)

# =========================================================================
#  Innovation Part: Texture-Aware Components
# =========================================================================

class TextureBranch_LoRA(nn.Module):
    """
    [创新核心] 纹理门控分支
    结构: Pointwise(降维) -> Depthwise Large Kernel(纹理上下文) -> Pointwise(升维) -> Sigmoid
    作用: 
    1. 低秩(LoRA)特性: 减少参数，防止在长尾类别上过拟合。
    2. 大核(Large Kernel): 捕捉骨小梁、填充块的低频纹理一致性。
    3. Sigmoid门控: 抑制高亮的金属伪影 (Gate -> 0)。
    """
    def __init__(self, in_dim, out_dim, reduction=4, kernel_size=7):
        super().__init__()
        # 内部瓶颈维度，至少保证有8个通道
        hidden_dim = max(in_dim // reduction, 8)
        
        self.net = nn.Sequential(
            # 1. LoRA A: 降维，压缩信息，聚焦主要纹理成分
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            
            # 2. Context: 大核卷积提取空间纹理 (Depthwise以节省参数)
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, 
                      padding=kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            
            # 3. LoRA B: 升维，对齐主分支维度
            nn.Conv2d(hidden_dim, out_dim, kernel_size=1, bias=True),
            
            # 4. Gating: 生成 [0, 1] 权重
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class TextureStarBlock(nn.Module):
    """
    [创新 Block] 用于替换深层的普通 Block
    """
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        # 深度卷积 (Spatial Mixing) - 保持不变
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        
        # Branch 1 (f1): 负责携带“形状/轮廓/高频”特征 (保持标准卷积)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        
        # Branch 2 (f2): 替换为 LoRA 纹理门控
        # 输入 dim, 输出必须与 f1 一致 (mlp_ratio * dim)
        self.f2_texture = TextureBranch_LoRA(dim, mlp_ratio * dim, reduction=4, kernel_size=7)
        
        # 融合与输出
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        
        # --- Star Operation Modified ---
        x1 = self.f1(x)             # Shape Features (Main)
        x2 = self.f2_texture(x)     # Texture/Artifact Weights (Gate)
        
        # Element-wise multiplication: 
        # 纹理权重去"雕刻"形状特征，抑制伪影区域
        x = self.act(x1) * x2       
        # -------------------------------
        
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x

# =========================================================================
#  Original Components (Preserved for shallow layers)
# =========================================================================

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

# =========================================================================
#  Main Model
# =========================================================================

class StarNet(nn.Module):
    def __init__(self, base_dim=32, depths=[3, 3, 12, 5], mlp_ratio=4, drop_path_rate=0.0, num_classes=1000, **kwargs):
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
            
            # --- Strategy: Hybrid Block Usage ---
            # Stage 0 & 1 (Shallow): Use Standard Block for basic edge/shape detection.
            # Stage 2 & 3 (Deep): Use TextureStarBlock for semantic texture & artifact suppression.
            if i_layer < 2:
                BlockType = Block
            else:
                BlockType = TextureStarBlock
            # ------------------------------------

            blocks = [BlockType(self.in_channel, mlp_ratio, dpr[cur + i]) for i in range(depths[i_layer])]
            cur += depths[i_layer]
            self.stages.append(nn.Sequential(down_sampler, *blocks))
        
        # head
        self.norm = nn.BatchNorm2d(self.in_channel)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
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
        return self.head(x)

# =========================================================================
#  Model Builders
# =========================================================================

@register_model
def starnet_s1_lora(pretrained=False, **kwargs):
    model = StarNet(24, [2, 2, 8, 3], **kwargs)
    if pretrained:
        # Load logic needs to be handled carefully if shapes change, 
        # but for fresh training this is fine.
        url = model_urls['starnet_s1']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        # filtering out keys might be needed if you load pre-trained weights due to architecture change
        model.load_state_dict(checkpoint["state_dict"], strict=False) 
    return model
