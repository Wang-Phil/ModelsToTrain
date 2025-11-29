"""
Implementation of Prof-of-Concept Network: StarNet with Dynamic Inter-Stage Gating.

The Inter-Stage Gating (StarGate) is added to dynamically modulate the input feature 
flow between consecutive stages, utilizing a lightweight Star Operation for gating weight calculation.
"""
import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model

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
        x = self.act(x1) * x2 # Star Operation (Element-wise Multiplication)
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x


# !!! 新增 StarGate 模块 !!!
class StarGate(nn.Module):
    """
    轻量级 StarNet 门控模块，用于生成通道权重 alpha (Channel-wise Gating)。
    门控权重用于动态调制相邻 Stage 间的特征流。
    """
    def __init__(self, in_channels, reduce_ratio=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        reduced_channels = max(in_channels // reduce_ratio, 1) # 确保通道数至少为1
        
        # 降维的 1x1 卷积
        self.conv_reduce = ConvBN(in_channels, reduced_channels, 1, with_bn=False)
        self.act = nn.ReLU6()
        
        # 升维的 1x1 卷积 F1 和 F2，用于 Star Operation
        self.conv_expand_f1 = ConvBN(reduced_channels, in_channels, 1, with_bn=False)
        self.conv_expand_f2 = ConvBN(reduced_channels, in_channels, 1, with_bn=False)
        
        # 最终的 Sigmoid 激活函数，将权重限制在 (0, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. 空间信息压缩：全局平均池化 [B, C, H, W] -> [B, C, 1, 1]
        y = self.avg_pool(x)
        
        # 2. 特征降维与激活
        y = self.act(self.conv_reduce(y))
        
        # 3. Star Operation (元素级乘法) 生成权重
        y1 = self.conv_expand_f1(y)
        y2 = self.conv_expand_f2(y)
        
        # 核心的 Star Operation
        gate_features = y1 * y2 
        
        # 4. Sigmoid 激活得到门控权重 alpha [B, C, 1, 1]
        alpha = self.sigmoid(gate_features)
        
        return alpha


class StarNet(nn.Module):
    def __init__(self, base_dim=32, depths=[3, 3, 12, 5], mlp_ratio=4, drop_path_rate=0.0, num_classes=1000, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.base_dim = base_dim # 存储 base_dim 供 gate 初始化使用
        self.in_channel = 32
        
        # stem layer
        self.stem = nn.Sequential(ConvBN(3, self.in_channel, kernel_size=3, stride=2, padding=1), nn.ReLU6())
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] # stochastic depth
        
        # build stages
        self.stages = nn.ModuleList()
        cur = 0
        
        for i_layer in range(len(depths)):
            embed_dim = base_dim * 2 ** i_layer
            # Stage 的 down_sampler 会将输入通道从 self.in_channel 变为 embed_dim
            down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1)
            self.in_channel = embed_dim
            
            blocks = [Block(self.in_channel, mlp_ratio, dpr[cur + i]) for i in range(depths[i_layer])]
            cur += depths[i_layer]
            self.stages.append(nn.Sequential(down_sampler, *blocks))
            
        # !!! 新增：为每个 Stage 的输入添加一个 StarGate !!!
        # Stage i 的输入 (即 Stage i-1 的输出) 通道数是 base_dim * 2 ** (i-1)
        # 考虑到 stages[0] 的输入是 stem 的输出 (self.in_channel=32)，需要特殊处理
        
        self.gates = nn.ModuleList()
        # 1. 第一个 Stage 的门控 (输入是 stem 的输出，通道数 32)
        self.gates.append(StarGate(32)) 
        
        # 2. 剩余 Stage 的门控 (输入是上一个 Stage 的输出)
        for i in range(1, len(depths)):
            # Stage i 的输入通道数是 base_dim * 2 ** (i-1)
            gate_in_channels = base_dim * 2 ** (i-1) 
            self.gates.append(StarGate(gate_in_channels)) 
            
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
        
        # !!! 门控机制在 Stage 间应用 !!!
        for i, stage in enumerate(self.stages):
            # 1. 计算门控权重 alpha
            # x 是当前 Stage 的输入特征
            alpha = self.gates[i](x) 
            
            # 2. 动态调制 Stage 输入
            # F_in = alpha * F_in
            x = x * alpha
            
            # 3. 执行当前 Stage
            x = stage(x)
            
        x = torch.flatten(self.avgpool(self.norm(x)), 1)
        return self.head(x)


@register_model
def starnet_s1_gated(pretrained=False, **kwargs):
    model = StarNet(24, [2, 2, 8, 3], **kwargs)
    if pretrained:
        # Note: Pretrained weights are for the original StarNet, not the gated version.
        # This part is kept for compatibility but needs re-training.
        url = model_urls['starnet_s1']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        # May need careful key mapping or ignore keys for self.gates
        model.load_state_dict(checkpoint["state_dict"], strict=False) 
    return model
