import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model

class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)

# ==========================================
# 核心改造：轻量级 SK 融合模块
# ==========================================
class LightSKFusion(nn.Module):
    """
    轻量级 Selective Kernel 融合
    不使用全连接层，改用 1x1 卷积和 GlobalAvgPool
    """
    def __init__(self, dim, output_dim=None):
        super().__init__()
        self.dim = dim
        self.output_dim = output_dim or dim
        
        # 分支 1: 小感受野 (3x3)
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU6()
        )
        
        # 分支 2: 大感受野 (7x7) -> 使用 3x3 dilation=3 模拟，减少参数
        self.branch7x7 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=3, dilation=3, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU6()
        )
        
        # 注意力生成器 (Channel Attention)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1, bias=False), # 降维
            nn.ReLU6(),
            nn.Conv2d(dim // 4, dim * 2, 1, bias=False), # 升维到 2*dim (两个分支的权重)
        )
        self.softmax = nn.Softmax(dim=1)
        
        # 融合后的投影 (可选)
        self.proj = ConvBN(dim, self.output_dim, 1, with_bn=True)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 1. 计算两个分支
        feat_3x3 = self.branch3x3(x)
        feat_7x7 = self.branch7x7(x)
        
        # 2. 叠加用于计算注意力
        feat_sum = feat_3x3 + feat_7x7
        
        # 3. 计算权重 [B, 2*C, 1, 1]
        attn = self.fc(self.gap(feat_sum))
        attn = attn.reshape(B, 2, C, 1, 1)
        attn = self.softmax(attn) # 在分支维度做 Softmax
        
        # 4. 加权融合
        # attn[:, 0] 是 3x3 的权重, attn[:, 1] 是 7x7 的权重
        feat_out = (feat_3x3 * attn[:, 0]) + (feat_7x7 * attn[:, 1])
        
        return self.proj(feat_out)

# ==========================================
# 改造后的 SK-StarNet Block
# ==========================================
class SKStarBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        
        # 1. 动态多尺度空间混合 (Dynamic Multi-Scale Spatial Mixing)
        # 这里替换了 StarNet 原始单一的 7x7 DWConv
        self.sk_mixer = LightSKFusion(dim)
        
        # 2. Star Operation (FFN)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False) # 分支1
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False) # 分支2
        
        # StarNet 核心：Act(x1) * x2
        self.act = nn.ReLU6()
        
        # 3. 输出投影
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        
        # 4. 第二次空间混合 (可选)
        # StarNet 原版这里有一个 7x7 DWConv。
        # 由于前面已经用了强力的 SK Mixer，这里可以用一个简单的 3x3 DWConv 做补充，或者直接去掉
        self.dwconv2 = ConvBN(dim, dim, 3, 1, 1, groups=dim, with_bn=False)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        
        # Step 1: SK 多尺度特征提取
        x = self.sk_mixer(x)
        
        # Step 2: 宽升维
        x1 = self.f1(x)
        x2 = self.f2(x)
        
        # Step 3: Star Operation (高维特征交互)
        x = self.act(x1) * x2
        
        # Step 4: 降维与后处理
        x = self.g(x)
        x = self.dwconv2(x)
        
        # Step 5: 残差
        x = input + self.drop_path(x)
        return x

# ==========================================
# 网络主体
# ==========================================
class SKStarNet(nn.Module):
    def __init__(self, base_dim=32, depths=[3, 3, 12, 5], mlp_ratio=4, drop_path_rate=0.0, num_classes=1000, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.in_channel = 32
        
        # Stem
        self.stem = nn.Sequential(ConvBN(3, self.in_channel, kernel_size=3, stride=2, padding=1), nn.ReLU6())
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        self.stages = nn.ModuleList()
        cur = 0
        for i_layer in range(len(depths)):
            embed_dim = base_dim * 2 ** i_layer
            down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1)
            self.in_channel = embed_dim
            
            blocks = []
            for i in range(depths[i_layer]):
                # 可以在这里做混合：
                # 比如：每隔一个 block 用一次 SK，其余用普通 block，以节省算力
                # 这里为了演示，全部使用 SKStarBlock
                blocks.append(SKStarBlock(self.in_channel, mlp_ratio, dpr[cur + i]))
                
            cur += depths[i_layer]
            self.stages.append(nn.Sequential(down_sampler, *blocks))
        
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
        features = torch.flatten(self.avgpool(self.norm(x)), 1)
        return self.head(features)

@register_model
def sk_starnet_s1(pretrained=False, **kwargs):
    model = SKStarNet(24, [2, 2, 8, 3], mlp_ratio=3, **kwargs)
    return model