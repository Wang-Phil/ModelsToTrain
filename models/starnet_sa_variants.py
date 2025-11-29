"""
StarNet 空间注意力变体模型
根据空间注意力在不同stage的使用情况，创建4个不同的模型变体：
- starnet_sa_s1: 所有stage都加空间注意力 (stage 0,1,2,3)
- starnet_sa_s2: 第一个stage不加注意力 (stage 1,2,3加注意力)
- starnet_sa_s3: 前两个stage不加注意力 (stage 2,3加注意力)
- starnet_sa_s4: 前三个stage不加注意力 (只有stage 3加注意力)
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


class ConvBN(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=0, g=1, bn=True):
        modules = [nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)]
        if bn: modules.append(nn.BatchNorm2d(out_ch))
        super().__init__(*modules)

class SpatialAttention(nn.Module):
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

class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.0, with_attn=False):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.dwconv = ConvBN(dim, dim, 7, 1, 3, g=dim, bn=True)
        self.f1 = ConvBN(dim, hidden, 1, bn=False)
        self.f2 = ConvBN(dim, hidden, 1, bn=False)
        self.g = ConvBN(hidden, dim, 1, bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, 3, g=dim, bn=False)
        self.act = nn.RelU()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.with_attn = with_attn
        self.norm = nn.BatchNorm2d(dim)
        self.sa = SpatialAttention(kernel_size=7)

    def forward(self, x):
        input = x
        # if self.with_attn:
        x = self.sa(x)
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = self.norm(x)    # 添加归一化层 可能这个比较重要
        return input + self.drop_path(x)


class StarNet_SA(nn.Module):
    """
    StarNet Model with configurable Spatial Attention across stages
    
    Args:
        base_dim: 基础维度
        depths: 每个stage的block数量
        mlp_ratio: MLP扩展比例
        drop_path_rate: Drop path rate
        num_classes: 分类类别数
        sa_stages: 哪些stage使用空间注意力，例如 [0,1,2,3] 表示所有stage都使用
    """
    def __init__(self, base_dim=32, depths=[3, 3, 12, 5], mlp_ratio=4, drop_path_rate=0.1, num_classes=1000, dropout_rate=0.1, use_attn=None,cls_num_list=None, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.in_channel = 32
        
        # stem layer
        self.stem = nn.Sequential(ConvBN(3, self.in_channel, k=3, s=2, p=1), nn.RelU())
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth
        
        # build stages
        self.stages = nn.ModuleList()
        cur = 0
        for i_layer in range(len(depths)):
            embed_dim = base_dim * 2 ** i_layer
            down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1)
            self.in_channel = embed_dim

            # 空间注意力机制已注释
            use_attn_here = True
            if use_attn is not None:
                if i_layer < use_attn:
                    use_attn_here = False
            else:
                use_attn_here = False
            
            
            blocks = [
                Block(self.in_channel, mlp_ratio, dpr[cur + i], with_attn=use_attn_here) 
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


# ============================================
# 模型变体注册
# ============================================

@register_model
def starnet_sa_s1(pretrained=False, **kwargs):
    """
    StarNet SA S1: 所有stage都加空间注意力 (stage 0,1,2,3)
    参数量: 约 2.9M (基于 S1 配置)
    """
    model = StarNet_SA(
        base_dim=24, 
        depths=[2, 2, 8, 3], 
        use_attn=0,  # 所有stage都使用空间注意力
        **kwargs
    )
    if pretrained:
        url = model_urls['starnet_s1']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    return model


@register_model
def starnet_sa_s2(pretrained=False, **kwargs):
    """
    StarNet SA S2: 第一个stage不加注意力 (stage 1,2,3加注意力)
    参数量: 约 2.9M (基于 S1 配置)
    """
    model = StarNet_SA(
        base_dim=24, 
        depths=[2, 2, 8, 3], 
        use_attn=1,  # stage 0不使用，stage 1,2,3使用
        **kwargs
    )
    if pretrained:
        url = model_urls['starnet_s1']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    return model


@register_model
def starnet_sa_s3(pretrained=False, **kwargs):
    """
    StarNet SA S3: 前两个stage不加注意力 (stage 2,3加注意力)
    参数量: 约 2.9M (基于 S1 配置)
    """
    model = StarNet_SA(
        base_dim=24, 
        depths=[2, 2, 8, 3], 
        use_attn=2,  # stage 0,1不使用，stage 2,3使用
        **kwargs
    )
    if pretrained:
        url = model_urls['starnet_s1']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    return model


@register_model
def starnet_sa_s4(pretrained=False, **kwargs):
    """
    StarNet SA S4: 前三个stage不加注意力 (只有stage 3加注意力)
    参数量: 约 2.9M (基于 S1 配置)
    """
    model = StarNet_SA(
        base_dim=24, 
        depths=[2, 2, 8, 3], 
        use_attn=3,  # 只有stage 3使用空间注意力
        **kwargs
    )
    if pretrained:
        url = model_urls['starnet_s1']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    return model

