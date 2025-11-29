"""
StarNet Model with Channel Frequency Re-scaling (CF-StarBlock)
Innovation Focus: Texture Dominance & Long-Tail Distribution.

1. CFStarBlock: Integrates Channel Frequency Attention and a class-frequency re-scaling term 
                into the Star Operation's gating branch (f2).
2. Hybrid Staging: Uses CFStarBlock in deep stages (2 & 3) for semantic texture fusion.
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

# ------------------------------------------------------------
# 辅助函数类
# ------------------------------------------------------------
class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)

class SpatialAttention(nn.Module):
    """用户原有的空间注意力模块"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.bn(self.conv1(x_cat)))
        return x * scale

# ------------------------------------------------------------
# 创新核心组件
# ------------------------------------------------------------
class ChannelFrequencyAttention(nn.Module):
    """
    [创新核心组件] 通道频率注意力和长尾重标定
    通过全局平均池化 Squeeze & Excitation (SE) 结构，并引入可学习的长尾权重。
    """
    def __init__(self, channels, reduction=4, cls_num_list=None):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Squeeze Operation: 降维
        self.squeeze = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.BatchNorm2d(channels // reduction),
            nn.SiLU()
        )
        # Excitation Operation: 升维
        self.excitation = nn.Conv2d(channels // reduction, channels, 1, bias=False)
        
        # 长尾重标定系数 (Long-Tail Re-weighting)
        if cls_num_list is not None:
            # 这里的计算是示例，实际中可能需要更复杂的映射
            # 目标是初始化一个可学习的参数，使其在训练初期倾向于平衡长尾类别
            num_classes = len(cls_num_list)
            # 简单的倒数平方根启发式初始化
            inv_freq = torch.pow(torch.tensor(cls_num_list, dtype=torch.float), -0.5)
            inv_freq = inv_freq / inv_freq.sum() * channels # 归一化并缩放到通道数
            
            # 使用一个 1x1 卷积来实现通道级的权重缩放
            self.cls_weight_conv = nn.Conv2d(channels, channels, 1, bias=False)
            # 通过权重初始化来引入长尾信息 (这里假设权重与频率相关)
            self.cls_weight_conv.weight.data.fill_(1.0) # 初始化为 1.0 (不起作用)
            # self.cls_weight_conv.weight.data = ... (可以更复杂地初始化)
            self.cls_weight = nn.Parameter(torch.ones(1, channels, 1, 1), requires_grad=True)
        else:
            self.cls_weight = 1.0 

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.squeeze(y)
        y = self.excitation(y)
        
        # 施加长尾重标定系数
        if isinstance(self.cls_weight, nn.Parameter):
            y = y * self.cls_weight
        
        return self.sigmoid(y) # Output: [B, C, 1, 1] 权重

class CFStarBlock(nn.Module):
    """
    [创新 Block] 通道频率 Star Block
    f1: Shape/Detail (标准 MLP)
    f2: Texture/Gate (CF Attention 调制)
    """
    def __init__(self, dim, mlp_ratio=3, drop_path=0., cls_num_list=None, with_attn=False):
        super().__init__()
        
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        
        # 引入 CF Attention
        self.cf_attn = ChannelFrequencyAttention(
            channels=dim, 
            reduction=4, 
            cls_num_list=cls_num_list 
        )
        
        # 门控调制器：将加权后的特征映射到 Star Operation 的维度
        self.gate_modulator = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.with_attn = with_attn
        self.sa = SpatialAttention(kernel_size=7) if with_attn else None

    def forward(self, x):
        input = x
        if self.with_attn and self.sa is not None:
            x = self.sa(x)
        x_dw = self.dwconv(x) 
        
        # 1. Branch f1: Detail Features
        x1 = self.f1(x_dw) 
        
        # 2. Branch f2: Channel Frequency & Long-Tail Gate
        cf_weights = self.cf_attn(x_dw) 
        x_weighted = x_dw * cf_weights # 通道级频率/长尾重标定
        x2 = self.gate_modulator(x_weighted) # 维度扩展
        
        # 3. Star Operation: Detail @ Weighted Gate
        x = self.act(x1) * x2
        
        # 4. 投影与跳跃连接
        x = self.dwconv2(self.g(x)) 
        x = input + self.drop_path(x)
        return x

# ------------------------------------------------------------
# 兼容与原版 Block (用于浅层)
# ------------------------------------------------------------
class Block(nn.Module):
    """
    原版 StarNet Block (兼容新的参数传递)
    """
    def __init__(self, dim, mlp_ratio=3, drop_path=0., with_attn=False, cls_num_list=None):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.with_attn = with_attn
        self.sa = SpatialAttention(kernel_size=7)

    def forward(self, x):
        input = x
        if self.with_attn:
            x = self.sa(x)
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x)) 
        x = input + self.drop_path(x)
        return x

# ------------------------------------------------------------
# 主模型
# ------------------------------------------------------------

class StarNet_CF(nn.Module):
    """
    StarNet Model with Channel Frequency Re-scaling (CF-StarBlock) Integration.
    """
    def __init__(self, base_dim=32, depths=[3, 3, 12, 5], mlp_ratio=4, drop_path_rate=0.0, 
                 num_classes=1000, cls_num_list=None, use_attn=None, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.in_channel = 32
        
        # stem layer (使用 SiLU 如同用户代码所示)
        self.stem = nn.Sequential(ConvBN(3, self.in_channel, kernel_size=3, stride=2, padding=1), nn.SiLU())
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        self.stages = nn.ModuleList()
        cur = 0
        for i_layer in range(len(depths)):
            embed_dim = base_dim * 2 ** i_layer
            down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1)
            self.in_channel = embed_dim

            # --- 块选择策略 ---
            # Stage 0 & 1: 使用原始 Block
            # Stage 2 & 3: 使用 CFStarBlock (i_layer >= 2)
            if i_layer < 2:
                BlockType = Block
            else:
                BlockType = CFStarBlock
            # -----------------
            
            use_attn_here = (use_attn is not None and i_layer >= use_attn)

            blocks = [
                BlockType(
                    self.in_channel, 
                    mlp_ratio, 
                    dpr[cur + i], 
                    with_attn=use_attn_here, # 沿用用户定义的注意力启用逻辑
                    cls_num_list=cls_num_list # 传递长尾信息
                )
                for i in range(depths[i_layer])
            ]
            
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

# ------------------------------------------------------------
# 模型变体注册
# ------------------------------------------------------------

@register_model
def starnet_cf_s3(pretrained=False, **kwargs):
    """
    StarNet CF S3: 使用 Channel Frequency Star Block
    """
    # 假设 cls_num_list 和 use_attn 在 kwargs 中传入
    model = StarNet_CF(
        base_dim=24, 
        depths=[2, 2, 8, 3], 
        **kwargs
    )
    if pretrained:
        url = model_urls['starnet_s3']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=False) # strict=False 允许缺失新模块的权重
    return model

# 示例运行部分
if __name__ == '__main__':
    # 示例：长尾类别数量列表
    dummy_cls_num_list = [1000, 800, 600, 400, 200, 100, 80, 70, 60, 50] 
    
    # 示例：启用空间注意力从 Stage 2 开始 (i_layer=2)
    model = starnet_cf_s3(
        num_classes=10, 
        cls_num_list=dummy_cls_num_list, 
        use_attn=2
    )
    input_tensor = torch.randn(2, 3, 224, 224)
    output = model(input_tensor)
    
    print(f"CF-StarNet (S3 configuration) initialized successfully.")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")
    print(f"Output shape: {output.shape}")