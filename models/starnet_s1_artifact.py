"""
Modified StarNet with Artifact-Suppressing Soft-Gating (抗伪影软门控)

Innovation: ArtifactStarBlock replaces the standard f2 branch with a thresholded 
Sigmoid gate designed to output near zero for high-intensity artifact regions.
"""

import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model

# 保持原有的 ConvBN 辅助类
class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)


# =========================================================================
#  Innovation Part: Artifact Suppressing Components
# =========================================================================

class ArtifactGate(nn.Module):
    """
    [创新核心] 伪影抑制门控
    公式实现: Y = Sigmoid( Tau - Conv(X) )
    其中 Tau 是可学习的通道级阈值。
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # 1x1 卷积，用于将输入特征映射到输出维度
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=True)
        
        # 可学习的阈值 Tau (通道级，非空间级)
        # 初始化为一个正值 (例如 2.0)，保证 Sigmoid 在初始化时不会全部趋于1或0
        self.tau = nn.Parameter(torch.ones(1, out_dim, 1, 1) * 2.0) 
        
        # 门控激活函数
        self.activation = nn.Sigmoid()

    def forward(self, x):
        # 1. 特征映射
        x_raw = self.conv(x)
        
        # 2. 反向注意力软门控
        # Sigmoid( Tau - X_raw )
        # - 如果 X_raw (代表高亮伪影) >> Tau, 则 (Tau - X_raw) << 0, Sigmoid -> 0 (抑制)
        # - 如果 X_raw (代表正常组织) << Tau, 则 (Tau - X_raw) > 0, Sigmoid -> 1 (放行)
        gate_output = self.activation(self.tau - x_raw)
        
        return gate_output


class ArtifactStarBlock(nn.Module):
    """
    [创新 Block] 用于替换深层的普通 Block
    """
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        # 保持原有的输入处理
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        
        # Branch 1 (f1): 负责携带主要特征 (保持标准 1x1 卷积)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        
        # Branch 2 (f2): 替换为伪影抑制门控
        self.f2_artifact_gate = ArtifactGate(dim, mlp_ratio * dim)
        
        # 融合与输出
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        
        # --- Star Operation Modified ---
        x1 = self.f1(x)             # Main Features
        x2 = self.f2_artifact_gate(x) # Artifact Suppression Gate
        
        # Element-wise multiplication: 
        # 当门控输出 x2 接近 0 时，x1 特征被抑制。
        x = self.act(x1) * x2       
        # -------------------------------
        
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x

# =========================================================================
#  Original Block (Preserved for shallow layers)
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
#  Main Model (Modified to integrate ArtifactStarBlock)
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
            # Stage 0 & 1 (Shallow): Use Standard Block (基本特征提取)
            # Stage 2 & 3 (Deep): Use ArtifactStarBlock (在语义层进行伪影抑制)
            if i_layer < 2:
                BlockType = Block
            else:
                BlockType = ArtifactStarBlock
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
#  Model Builders (Registration)
# =========================================================================

model_urls = {
    "starnet_s1": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s1.pth.tar",
    "starnet_s2": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s2.pth.tar",
    "starnet_s3": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s3.pth.tar",
    "starnet_s4": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s4.pth.tar",
}


@register_model
def starnet_artifact_s1(pretrained=False, **kwargs):
    # 使用 StarNet S1 的配置，但内部使用了 ArtifactStarBlock
    model = StarNet(24, [2, 2, 8, 3], **kwargs)
    if pretrained:
        url = model_urls['starnet_s1']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    return model

# ... (其他注册函数 starnet_s1, s2, s4 保持不变，可根据需要调整) ...


if __name__ == '__main__':
    # Simple Test to verify dimension compatibility
    model = starnet_artifact_s1(num_classes=10)
    input_tensor = torch.randn(2, 3, 224, 224)
    output = model(input_tensor)
    print(f"Model Output Shape: {output.shape}")
    print("Artifact-Suppressing StarNet initialized successfully.")