"""
Implementation of Prof-of-Concept Network: StarNet.

We make StarNet as simple as possible [to show the key contribution of element-wise multiplication]:
    - like NO layer-scale in network design,
    - and NO EMA during training,
    - which would improve the performance further.

Created by: Xu Ma (Email: ma.xu1@northeastern.edu)
Modified Date: Mar/29/2024
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model

model_urls = {
    "starnet_s1": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s1.pth.tar",
    "starnet_s2": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s2.pth.tar",
    "starnet_s3": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s3.pth.tar",
    "starnet_s4": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s4.pth.tar",
}

class DynamicConv1x1(nn.Module):
    """
    简化版 ODConv/CondConv 1x1 实现。
    通过动态加权 N 个专家（Expert）卷积核，实现动态适应性。
    """
    def __init__(self, in_channels, out_channels, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        
        # 1. N 个静态专家卷积核（Expert Kernels）
        # 形状: (N * out_c, in_c, 1, 1)
        # 注意: 这里我们将 N 个核堆叠在输出通道维度上
        self.weights = nn.Parameter(
            torch.randn(num_experts * out_channels, in_channels, 1, 1)
        )
        # 初始化
        nn.init.kaiming_uniform_(self.weights, a=5) 
        
        # 2. 动态注意力生成模块（Attention Module）
        # 用于生成 N 个专家权重 (α_i)
        # 输入: C -> 中间 FC -> 输出: N
        self.attn_generator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # GAP: (B, C, 1, 1)
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1), # 降维
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, num_experts, kernel_size=1) # 输出 N 个通道
        )

    def forward(self, x):
        # x.shape: (B, C_in, H, W)
        B, C_in, H, W = x.shape
        
        # 1. 动态生成 N 个专家权重 (α_i)
        # attn_weights shape: (B, N, 1, 1)
        attn_weights = self.attn_generator(x)
        attn_weights = F.softmax(attn_weights, dim=1) 
        
        # 2. 生成最终的动态卷积核 (W_dynamic)
        # W_dynamic shape: (B, N, C_out, C_in, 1, 1) -> 需要重塑
        
        # 将静态权重复制 B 次
        static_weights = self.weights.unsqueeze(0).repeat(B, 1, 1, 1, 1)
        # 重塑专家权重: (B, N, 1, 1) -> (B, N, 1, 1, 1, 1)
        attn_weights = attn_weights.view(B, self.num_experts, 1, 1, 1, 1) 
        
        # 将静态权重按 N 分组: (B, N, C_out, C_in, 1, 1)
        static_weights = static_weights.view(
            B, self.num_experts, -1, C_in, 1, 1
        )
        
        # 动态加权融合: W_dynamic = Sum(α_i * W_i)
        # W_dynamic_fused shape: (B, C_out, C_in, 1, 1)
        W_dynamic_fused = (static_weights * attn_weights).sum(dim=1)

        # 3. 执行动态卷积
        # 传统 Conv2d 不支持 batch 维度的动态权重
        # 必须手动实现 batch-wise 卷积，通常使用 group convolution (非常快)
        
        # 重塑输入 x: (1, B*C_in, H, W)
        x_reshaped = x.view(1, B * C_in, H, W)
        # 重塑动态核: (B*C_out, C_in, 1, 1)
        W_reshaped = W_dynamic_fused.view(B * W_dynamic_fused.shape[1], C_in, 1, 1)

        # 执行 grouped convolution，group 数量 = Batch Size (B)
        # 结果 shape: (1, B*C_out, H, W)
        out = F.conv2d(
            input=x_reshaped,
            weight=W_reshaped,
            bias=None, # 简化，没有 bias
            stride=1,
            padding=0,
            groups=B # 核心：B组卷积
        )
        
        # 恢复输出 shape: (B, C_out, H, W)
        out = out.view(B, -1, H, W)
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
    def __init__(self, dim, mlp_ratio=3, drop_path=0., num_experts=4):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mid_dim = mlp_ratio * dim
        self.f1 = DynamicConv1x1(dim, mid_dim, num_experts) # 动态生成 x1
        self.f2 = DynamicConv1x1(dim, mid_dim, num_experts) # 动态生成 x2
        self.bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.bn(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x


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
            blocks = [Block(self.in_channel, mlp_ratio, dpr[cur + i]) for i in range(depths[i_layer])]
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


@register_model
def starnet_s1_odconv(pretrained=False, **kwargs):
    model = StarNet(24, [2, 2, 8, 3], **kwargs)
    if pretrained:
        url = model_urls['starnet_s1']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
    return model