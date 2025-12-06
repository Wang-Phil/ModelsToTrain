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

class GRN(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        
        # gamma (γ) 和 beta (β) 是通道维度的可学习参数
        # 初始化为 1 和 0，形状为 (1, C, 1, 1)，便于广播
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.gamma.data.fill_(1.0) # γ 初始化为 1
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 计算响应范数（Response Normalization）
        # across spatial dimensions (H, W)
        
        # 计算 x^2 在 H, W 维度上的均值
        # 结果 shape: (B, C, 1, 1)
        Gx = x.pow(2).mean(dim=[2, 3], keepdim=True)
        
        # 开根号得到 L2 范数（Response Norm）
        # R_norm = sqrt(Gx) + epsilon
        Rx = torch.sqrt(Gx + self.eps)

        # 2. 归一化和竞争增强
        # Response = X / R_norm
        NormX = x / Rx
        
        # 3. 最终输出（根据用户提供的公式）
        # Output = gamma * NormX + beta + X
        out = self.gamma * NormX + self.beta + x
        
        return out

class CrossStarBlock(nn.Module):
    """
    [D - 基线] Inception-style Cross-Star Block
    实现了 Y = Concat((x_{3A} * x_{7B}), (x_{7A} * x_{3B}))
    交叉星乘：局部细节调制全局语境，全局语境校正局部细节
    """
    def __init__(self, dim, mlp_ratio=3, drop_path=0.,with_attn=False):
        super().__init__()
        
        # 保持原有的输入处理 (7x7 Depthwise Conv)
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        
        # 定义中间维度。总扩展维度是 dim * mlp_ratio，交叉乘积需要平分通道
        self.mid_dim = (dim * mlp_ratio) // 2
        
        # --- Multi-Scale Branches for Cross-Star Operation ---
        
        # Branch 1 (Local): 3x3 Conv (捕捉局部细节)
        self.f3 = ConvBN(dim, self.mid_dim, 3, 1, 1, with_bn=False) # Conv_{3x3}
        
        # Branch 2 (Context): 7x7 Conv (捕捉全局语境和纹理)
        self.f7 = ConvBN(dim, self.mid_dim, 7, 1, 3, with_bn=False) # Conv_{7x7}
        
        # 融合与输出
        # 注意：如果只使用 y12（不concat y21），输入通道数应该是 mid_dim 而不是 dim * mlp_ratio
        self.g = ConvBN(self.mid_dim, dim, 1, with_bn=True) # 输入通道数改为 mid_dim
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.sa = SpatialAttention(kernel_size=7)
        # 为每个分支创建独立的 GRN（维度为 self.mid_dim）
        self.grn_y12 = GRN(dim=self.mid_dim)  # 用于 y12 分支
        self.grn_y21 = GRN(dim=self.mid_dim)  # 用于 y21 分支

        
    def forward(self, x):
        input = x
        x = self.sa(x)
        x = self.dwconv(x)
        
        # 1. 计算两个分支的特征（共享参数）
        x_3 = self.f3(x)
        x_7 = self.f7(x)

        
        # 2. 交叉星乘 (Cross-Star Operation) - D (基线)
        # 乘法 1: Point-wise (1A) 调制 Local (3B) -> 强调点卷积对局部细节的调制
        y12 = self.act(x_3) * x_7
        x_out = self.grn_y12(y12)
        
        # 4. 投影回输入维度
        x_out = self.dwconv2(self.g(x_out))
        x_out = input + self.drop_path(x_out)
        return x_out


class LightSKBlock(nn.Module):
    """
    轻量级自适应选择性卷积块 (Lightweight Selective Kernel Block)
    核心功能：动态选择最佳的感受野尺度 (3x3 vs 7x7)
    """
    def __init__(self, dim, mlp_ratio=3, drop_path=0., with_attn=False):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        
        # 1. 通道降维比例 (用于注意力计算的 bottleneck)
        reduction = 4
        hidden_dim = max(dim // reduction, 32)

        # 2. 定义多尺度分支 (全部使用 DWConv 以节省参数)
        # 分支 A: 3x3 (Local)
        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU6()
        )
        # 分支 B: 3x3 dilation=3 (Context, 感受野=7x7)
        self.conv_7x7 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=3, dilation=3, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU6()
        )

        # 3. 自适应注意力生成器 (Adaptive Attention)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(),
            nn.Conv2d(hidden_dim, dim * 2, 1, bias=False) # 输出两个分支的权重
        )
        self.softmax = nn.Softmax(dim=1)

        # 4. 后续处理 (FFN / MLP)
        self.proj = ConvBN(dim, dim, 1, with_bn=True) # 融合一下
        
        # FFN 部分
        self.mlp = nn.Sequential(
            ConvBN(dim, int(dim * mlp_ratio), 1, with_bn=False),
            nn.ReLU6(),
            ConvBN(int(dim * mlp_ratio), dim, 1, with_bn=True)
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # 5. 空间注意力精炼 (依然保留，作为辅助)
        self.sa = SpatialAttention(kernel_size=7)

    def forward(self, x):
        input = x
        x = self.sa(x)
        # --- SK Unit Start ---
        # 1. 并行计算两个尺度
        feat_3 = self.conv_3x3(x)
        feat_7 = self.conv_7x7(x)
        
        # 2. 特征相加，计算全局描述符
        feat_sum = feat_3 + feat_7
        attn_vec = self.gap(feat_sum) # [B, C, 1, 1]
        
        # 3. 生成权重 [B, 2*C, 1, 1] -> 变形为 [B, 2, C, 1, 1]
        attn_weights = self.fc(attn_vec)
        B, _, H, W = x.shape
        attn_weights = attn_weights.view(B, 2, self.dim, 1, 1)
        attn_weights = self.softmax(attn_weights) # 在两个分支间归一化
        
        # 4. 加权融合 (Soft Attention Fusion)
        # weight[0] * feat_3 + weight[1] * feat_7
        x_sk = (attn_weights[:, 0] * feat_3) + (attn_weights[:, 1] * feat_7)
        # --- SK Unit End ---

        x = self.proj(x_sk)

        # FFN
        x = x + self.drop_path(self.mlp(x))

        x = input + self.drop_path(x)
        return x



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
        mid_dim = mlp_ratio * dim
        self.grn = GRN(dim=mid_dim)

    def forward(self, x):
        input = x
        # if self.with_attn:
        x = self.sa(x)
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.grn(x)
        x = self.dwconv2(self.g(x))
        
        x = input + self.drop_path(x)
        return x

class StarNet_FINAL(nn.Module):
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
            
            # --- Strategy: Hybrid Block Usage ---
            # Stage 0 & 1 (Shallow): Use Standard Block (低级特征提取)
            # Stage 2 & 3 (Deep): Use specified Block type (多尺度语义融合)
            if i_layer < 2:
                BlockType = Block
            else:
                BlockType = LightSKBlock

            blocks = [
                BlockType(self.in_channel, mlp_ratio, dpr[cur + i], with_attn=use_attn_here) 
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

@register_model
def starnet_s1_final(pretrained=False, **kwargs):
    """
    StarNet SA S1: 所有stage都加空间注意力 (stage 0,1,2,3)
    参数量: 约 2.9M (基于 S1 配置)
    """
    model = StarNet_FINAL(
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



