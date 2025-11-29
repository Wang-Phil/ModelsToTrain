"""
StarNet with Residual Fusion
融合多个Stage的特征，结合残差网络思想
融合 Stage 1, 2, 3 的特征进行多尺度特征融合
"""
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model

# ==========================================
# ConvBN 模块（从starnet.py复制）
# ==========================================
class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)

# ==========================================
# Block 模块（从starnet.py复制，不带注意力）
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
# StarNet with Residual Fusion
# ==========================================
class StarNet_ResFusion(nn.Module):
    def __init__(self, base_dim=32, depths=[3, 3, 12, 5], mlp_ratio=4, drop_path_rate=0.0, num_classes=1000, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.in_channel = 32
        
        # Stem (保持不变)
        self.stem = nn.Sequential(ConvBN(3, self.in_channel, kernel_size=3, stride=2, padding=1), nn.ReLU6())
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.stages = nn.ModuleList()
        
        # 用于存储每个 Stage 输出通道数的列表
        self.stage_dims = [] 
        
        cur = 0
        for i_layer in range(len(depths)):
            embed_dim = base_dim * 2 ** i_layer
            down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1)
            self.in_channel = embed_dim
            blocks = [Block(embed_dim, mlp_ratio, dpr[cur + i]) for i in range(depths[i_layer])]
            self.stages.append(nn.Sequential(down_sampler, *blocks))
            
            # 记录每个 Stage 的维度
            self.stage_dims.append(embed_dim)
            cur += depths[i_layer]
            
        # --- 修改点 1: 多尺度融合头 ---
        # 我们将融合 Stage 1, Stage 2, Stage 3 的特征
        # Stage 0 (浅层边缘) 通常噪音太大，不融合
        # 融合后的维度 = dim(Stage1) + dim(Stage2) + dim(Stage3)
        fusion_dim = self.stage_dims[1] + self.stage_dims[2] + self.stage_dims[3]
        
        # 为了处理不同 Stage 的特征，我们需要独立的 Norm 层
        self.norm_s2 = nn.BatchNorm2d(self.stage_dims[1])
        self.norm_s3 = nn.BatchNorm2d(self.stage_dims[2])
        self.norm_s4 = nn.BatchNorm2d(self.stage_dims[3])
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # 分类头现在的输入维度变大了
        self.dropout = nn.Dropout(0.1)  # 添加Dropout提高泛化能力
        self.head = nn.Linear(fusion_dim, num_classes)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        # (修复isinstance语法)
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        x = self.stem(x)
        
        # 依次通过 Stage，并保存中间结果
        # Stage 0
        x0 = self.stages[0](x) 
        
        # Stage 1 (我们不仅通过它，还要保存它) -> 对应上面的 self.stage_dims[1]
        x1 = self.stages[1](x0)
        
        # Stage 2
        x2 = self.stages[2](x1)
        
        # Stage 3 (最后一层)
        x3 = self.stages[3](x2)
        
        # --- 修改点 2: 特征融合逻辑 ---
        
        # 1. 对不同尺度的特征进行 Global Average Pooling
        # x1: [B, C1, H/4, W/4] -> [B, C1]
        feat_s2 = self.avgpool(self.norm_s2(x1)).flatten(1)
        
        # x2: [B, C2, H/8, W/8] -> [B, C2]
        feat_s3 = self.avgpool(self.norm_s3(x2)).flatten(1)
        
        # x3: [B, C3, H/16, W/16] -> [B, C3]
        feat_s4 = self.avgpool(self.norm_s4(x3)).flatten(1)
        
        # 2. 拼接 (Concatenation) - 这是 ResNet 变体 DenseNet 的精髓
        # 融合了 中层纹理 + 深层语义
        combined_feat = torch.cat([feat_s2, feat_s3, feat_s4], dim=1)
        
        # 3. 直接分类（不使用Dropout）
        combined_feat = self.dropout(combined_feat)
        return self.head(combined_feat)

# ==========================================
# 模型注册（基于starnet_s1配置）
# ==========================================
@register_model
def starnet_s1_res_fusion(pretrained=False, **kwargs):
    """
    StarNet S1 with Residual Fusion
    融合 Stage 1, 2, 3 的特征进行多尺度特征融合
    """
    model = StarNet_ResFusion(24, [2, 2, 8, 3], **kwargs)
    return model

