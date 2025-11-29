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
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model


model_urls = {
    "starnet_s1": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s1.pth.tar",
    "starnet_s2": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s2.pth.tar",
    "starnet_s3": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s3.pth.tar",
    "starnet_s4": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s4.pth.tar",
}

class LSKModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # -------------------------------------------------
        # 1. 定义两个并行的深度可分离卷积 (DW-Conv)
        # -------------------------------------------------
        
        # 分支 A: 5x5 卷积，捕获中等范围信息
        self.conv_5x5 = nn.Conv2d(
            dim, dim, 
            kernel_size=5, padding=2, groups=dim
        )
        
        # 分支 B: 7x7 膨胀卷积，dilation=3
        # padding = dilation * (kernel_size - 1) // 2 = 3 * 6 // 2 = 9
        self.conv_7x7_dilated = nn.Conv2d(
            dim, dim, 
            kernel_size=7, padding=9, groups=dim, dilation=3
        )
        
        # -------------------------------------------------
        # 2. 空间注意力融合机制 (Spatial Selection)
        # -------------------------------------------------
        # 用于生成两个分支的权重。
        # 简单的做法是：Concat -> Conv1x1 -> Sigmoid
        self.norm_fusion = nn.BatchNorm2d(dim) # 融合前的归一化（可选，但推荐）
        
        # 这里的 2 表示为两个分支各生成一个通道的空间注意力图
        # 为了轻量化，我们先降维再生成
        self.attention_conv = nn.Sequential(
            nn.Conv2d(dim, dim // 4, kernel_size=1), # 降维
            nn.ReLU(),
            nn.Conv2d(dim // 4, 2, kernel_size=1)    # 生成 2 个通道的 Attention Map
        )

    def forward(self, x):
        # 1. 并行计算
        feat_5x5 = self.conv_5x5(x)
        feat_7x7 = self.conv_7x7_dilated(x)
        
        # 2. 融合特征以计算注意力
        # (也可以选择 element-wise add，这里为了信息最大化选择 element-wise add)
        feat_sum = feat_5x5 + feat_7x7
        
        # 3. 生成空间注意力图
        # output shape: [B, 2, H, W]
        attn = self.attention_conv(feat_sum)
        # 在通道维度 1 上做 Softmax，使得两个分支的权重之和为 1
        attn = torch.softmax(attn, dim=1) 
        
        # 分离权重: att_5x5 和 att_7x7
        att_5x5 = attn[:, 0:1, :, :]
        att_7x7 = attn[:, 1:2, :, :]
        
        # 4. 动态加权融合 (Selective Kernel)
        out = feat_5x5 * att_5x5 + feat_7x7 * att_7x7
        
        return out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.bn = nn.GroupNorm(1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.bn(self.conv1(x_cat)))
        return x * scale

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
        # self.sa = SpatialAttention(kernel_size=7)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.spatial_mix = LSKModule(dim)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        input = x
        # x = self.sa(x)  # SpatialAttention is the spatial attention mechanism
        x = self.spatial_mix(x)
        x = self.norm(x)
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
        if isinstance(m, nn.Linear or nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm or nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = torch.flatten(self.avgpool(self.norm(x)), 1)
        return self.head(x)


@register_model
def starnet_s1_lsk(pretrained=False, **kwargs):
    model = StarNet(24, [2, 2, 8, 3], **kwargs)
    if pretrained:
        url = model_urls['starnet_s1']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
    return model
