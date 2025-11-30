"""
StarNet 空间注意力变体模型
根据空间注意力在不同stage的使用情况，创建4个不同的模型变体：
- starnet_sa_s1: 所有stage都加空间注意力 (stage 0,1,2,3)
- starnet_sa_s2: 第一个stage不加注意力 (stage 1,2,3加注意力)
- starnet_sa_s3: 前两个stage不加注意力 (stage 2,3加注意力)
- starnet_sa_s4: 前三个stage不加注意力 (只有stage 3加注意力)
"""

# import torch
# import torch.nn as nn
# from timm.models.layers import DropPath, trunc_normal_
# from timm.models.registry import register_model

# model_urls = {
#     "starnet_s1": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s1.pth.tar",
#     "starnet_s2": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s2.pth.tar",
#     "starnet_s3": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s3.pth.tar",
#     "starnet_s4": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s4.pth.tar",
# }


# class ConvBN(torch.nn.Sequential):
#     def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
#         super().__init__()
#         self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
#         if with_bn:
#             self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
#             torch.nn.init.constant_(self.bn.weight, 1)
#             torch.nn.init.constant_(self.bn.bias, 0)

# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#         assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
#         padding = 3 if kernel_size == 7 else 1
#         self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
#         self.bn = nn.GroupNorm(1, 1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         # 1. 沿着通道维度做 AvgPool 和 MaxPool
#         # x: [B, C, H, W] -> avg_out: [B, 1, H, W], max_out: [B, 1, H, W]
#         avg_out = torch.mean(x, dim=1, keepdim=True)
#         max_out, _ = torch.max(x, dim=1, keepdim=True)
        
#         # 2. 拼接
#         x_cat = torch.cat([avg_out, max_out], dim=1)
        
#         # 3. 卷积 + BN + Sigmoid 生成空间掩码
#         scale = self.sigmoid(self.bn(self.conv1(x_cat)))
        
#         # 4. 施加注意力
#         return x * scale

# class Block(nn.Module):
#     def __init__(self, dim, mlp_ratio=3, drop_path=0., with_attn=False):
#         super().__init__()
#         self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
#         self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
#         self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
#         self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
#         self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
#         self.act = nn.ReLU6()
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         # 3. 空间注意力模块（已注释）
#         self.with_attn = with_attn
#         self.sa = SpatialAttention(kernel_size=7)

#     def forward(self, x):
#         input = x
#         x = self.dwconv(x)
#         x1, x2 = self.f1(x), self.f2(x)
#         x = self.act(x1) * x2
#         x = self.dwconv2(self.g(x))
#         x = self.sa(x)
#         # [修改] 在 DropPath 和残差连接之前应用注意力（已注释）
#         # 这让网络在把特征加回主干之前，先"提炼"一次特征
#         # [修正] 只有开启开关才进行注意力计算
#         x = input + self.drop_path(x)
#         return x


# class StarNet_SA(nn.Module):
#     """
#     StarNet Model with configurable Spatial Attention across stages
    
#     Args:
#         base_dim: 基础维度
#         depths: 每个stage的block数量
#         mlp_ratio: MLP扩展比例
#         drop_path_rate: Drop path rate
#         num_classes: 分类类别数
#         sa_stages: 哪些stage使用空间注意力，例如 [0,1,2,3] 表示所有stage都使用
#     """
#     def __init__(self, base_dim=32, depths=[3, 3, 12, 5], mlp_ratio=4, drop_path_rate=0.0, num_classes=1000, dropout_rate=0.1, use_attn=None, **kwargs):
#         """
#         Args:
#             base_dim: 基础维度
#             depths: 每个stage的block数量
#             mlp_ratio: MLP比例
#             drop_path_rate: DropPath率
#             num_classes: 分类类别数
#             dropout_rate: Dropout比例（默认0.1，设置为0禁用Dropout）
#         """
#         super().__init__()
#         self.num_classes = num_classes
#         self.in_channel = 32
#         # stem layer
#         self.stem = nn.Sequential(ConvBN(3, self.in_channel, kernel_size=3, stride=2, padding=1),nn.ReLU6())
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] # stochastic depth
#         # build stages
#         self.stages = nn.ModuleList()
#         cur = 0
#         for i_layer in range(len(depths)):
#             embed_dim = base_dim * 2 ** i_layer
#             down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1)
#             self.in_channel = embed_dim
#             # 空间注意力机制配置
#             # use_attn=0: 所有stage都使用 (stage 0,1,2,3)
#             # use_attn=1: 从stage 1开始使用 (stage 1,2,3)
#             # use_attn=2: 从stage 2开始使用 (stage 2,3)
#             # use_attn=3: 只有stage 3使用
#             use_attn_here = False
#             if use_attn is not None:
#                 if use_attn == 0:  # 所有stage都使用
#                     use_attn_here = True
#                 elif i_layer >= use_attn:  # 从指定stage开始使用
#                     use_attn_here = True
#             blocks = [
#                 Block(self.in_channel, mlp_ratio, dpr[cur + i], with_attn=use_attn_here) 
#                 for i in range(depths[i_layer])
#             ]
#             cur += depths[i_layer]
#             self.stages.append(nn.Sequential(down_sampler, *blocks))
#         # head
#         self.norm = nn.BatchNorm2d(self.in_channel)
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         # Dropout已注释
#         # if dropout_rate > 0:
#         #     self.dropout = nn.Dropout(dropout_rate)
#         # else:
#         self.dropout = nn.Identity()
#         self.head = nn.Linear(self.in_channel, num_classes)
#         self.apply(self._init_weights)

#     def _init_weights(self, m):
#         if isinstance(m, (nn.Linear, nn.Conv2d)):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)

#     def forward(self, x):
#         x = self.stem(x)
#         for stage in self.stages:
#             x = stage(x)
#         x = torch.flatten(self.avgpool(self.norm(x)), 1)
#         # x = self.dropout(x)  # Dropout已注释
#         return self.head(x)

import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from .multi_head_classifiers import ArcFace, CosFace

# ==========================================
# 1. 新增： 空间注意力机制模块
# ==========================================
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. 沿着通道维度做 AvgPool 和 MaxPool
        # x: [B, C, H, W] -> avg_out: [B, 1, H, W], max_out: [B, 1, H, W]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # 2. 拼接
        x_cat = torch.cat([avg_out, max_out], dim=1)
        
        # 3. 卷积 + Sigmoid 生成空间掩码
        scale = self.sigmoid(self.conv1(x_cat))
        
        # 4. 施加注意力
        return x * scale
# ==========================================



class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)


class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0., with_attn=False):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # 3. 空间注意力模块（已注释）
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

        # [修改] 在 DropPath 和残差连接之前应用注意力（已注释）
        # 这让网络在把特征加回主干之前，先"提炼"一次特征
        # [修正] 只有开启开关才进行注意力计算
        

        x = input + self.drop_path(x)
        return x


class StarNet_SA(nn.Module):
    def __init__(self, base_dim=32, depths=[3, 3, 12, 5], mlp_ratio=4, drop_path_rate=0.0, num_classes=1000, dropout_rate=0.1, use_attn=None, use_multi_head=False, cls_num_list=None, **kwargs):
        """
        Args:
            base_dim: 基础维度
            depths: 每个stage的block数量
            mlp_ratio: MLP比例
            drop_path_rate: DropPath率
            num_classes: 分类类别数
            dropout_rate: Dropout比例（默认0.1，设置为0禁用Dropout）
            use_attn: 空间注意力使用策略
            use_multi_head: 是否使用多分类头（ArcFace, CosFace, LDAM, Softmax）
            cls_num_list: 类别数量列表（用于 LDAM），如果为 None 则使用均匀分布
        """
        super().__init__()
        self.num_classes = num_classes
        self.use_multi_head = use_multi_head
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
            # blocks = [Block(self.in_channel, mlp_ratio, dpr[cur + i]) for i in range(depths[i_layer])]
            cur += depths[i_layer]
            self.stages.append(nn.Sequential(down_sampler, *blocks))
        # head
        self.norm = nn.BatchNorm2d(self.in_channel)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # Dropout已注释
        # if dropout_rate > 0:
        #     self.dropout = nn.Dropout(dropout_rate)
        # else:
        self.dropout = nn.Identity()
        
        # 特征维度
        feat_dim = self.in_channel
        
        if use_multi_head:
            # 多分类头模式：ArcFace, CosFace, LDAM, Softmax
            self.head_softmax = nn.Linear(feat_dim, num_classes)
            self.head_arcface = ArcFace(feat_dim, num_classes, s=30.0, m=0.5)
            self.head_cosface = CosFace(feat_dim, num_classes, s=30.0, m=0.35)
            self.head_ldam = nn.Linear(feat_dim, num_classes)
            # 保存类别数量列表用于 LDAM Loss
            if cls_num_list is None:
                # 默认均匀分布
                self.cls_num_list = [1.0] * num_classes
            else:
                self.cls_num_list = cls_num_list
        else:
            # 单分类头模式（向后兼容）
            self.head = nn.Linear(feat_dim, num_classes)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear or nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm or nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, labels=None):
        """
        Args:
            x: 输入图像 [B, 3, H, W]
            labels: 标签 [B]，用于 ArcFace 和 CosFace（训练时需要）
        
        Returns:
            如果 use_multi_head=True:
                dict: {
                    'features': [B, feat_dim],
                    'logits_softmax': [B, num_classes],
                    'logits_arcface': [B, num_classes],
                    'logits_cosface': [B, num_classes],
                    'logits_ldam': [B, num_classes]
                }
            如果 use_multi_head=False:
                [B, num_classes] - 单个分类头的 logits
        """
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        features = torch.flatten(self.avgpool(self.norm(x)), 1)
        # features = self.dropout(features)  # Dropout已注释
        
        if self.use_multi_head:
            # 多分类头模式
            logits_softmax = self.head_softmax(features)
            logits_arcface = self.head_arcface(features, labels)
            logits_cosface = self.head_cosface(features, labels)
            logits_ldam = self.head_ldam(features)
            
            return {
                'features': features,
                'logits_softmax': logits_softmax,
                'logits_arcface': logits_arcface,
                'logits_cosface': logits_cosface,
                'logits_ldam': logits_ldam
            }
        else:
            # 单分类头模式（向后兼容）
            return self.head(features)

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

