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

class GRN(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        
        # gamma (γ) 和 beta (β) 是通道维度的可学习参数
        # 初始化为 1 和 0，形状为 (1, C, 1, 1)，便于广播
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
        # [优化] 建议初始化为 0，这样初始状态下该层为恒等映射 (Identity)，利于深层网络收敛
        # 如果你确实想要初始就有归一化效果，可以改回 1.0，但通常 0.0 是 ConvNeXt V2 的做法
        nn.init.constant_(self.gamma, 0.0)
        # self.gamma.data.fill_(1.0) # γ 初始化为 1
        
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
    """
    Standard StarNet Block with GRN and Spatial Attention
    """
    def __init__(self, dim, mlp_ratio=3, drop_path=0., use_sa=False):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        
        # ⭐ 添加 GRN 模块（在 Star Operation 之后）
        mid_dim = mlp_ratio * dim
        # self.grn = GRN(dim=mid_dim)
        
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.SiLU()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # ⭐ 可选的空间注意力机制
        self.use_sa = use_sa
        self.sa = SpatialAttention(kernel_size=7)

    def forward(self, x):
        input = x
        
        # ⭐ 可选：在输入处应用空间注意力
        if self.use_sa:
            x = self.sa(x)
        
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2  # Star Operation
        
        # ⭐ 应用 GRN（在投影之前）
        # x = self.grn(x)
        
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x

class CrossStarBlock(nn.Module):
    """
    [D - 基线] Inception-style Cross-Star Block
    实现了 Y = Concat((x_{3A} * x_{7B}), (x_{7A} * x_{3B}))
    交叉星乘：局部细节调制全局语境，全局语境校正局部细节
    """
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        
        # 保持原有的输入处理 (7x7 Depthwise Conv)
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        
        # 定义中间维度。总扩展维度是 dim * mlp_ratio，交叉乘积需要平分通道
        self.mid_dim = (dim * mlp_ratio) // 2
        
        # --- Multi-Scale Branches for Cross-Star Operation ---
        
        # Branch 1 (Local): 3x3 Convs (捕捉局部细节)
        self.f3_A = ConvBN(dim, self.mid_dim, 3, 1, 1, with_bn=False) # Conv_{3x3} A
        self.f3_B = ConvBN(dim, self.mid_dim, 3, 1, 1, with_bn=False) # Conv_{3x3} B
        
        # Branch 2 (Context): 7x7 Convs (捕捉全局语境和纹理)
        self.f7_A = ConvBN(dim, self.mid_dim, 7, 1, 3, with_bn=False) # Conv_{7x7} A
        self.f7_B = ConvBN(dim, self.mid_dim, 7, 1, 3, with_bn=False) # Conv_{7x7} B
        
        # 融合与输出
        self.g = ConvBN(dim * mlp_ratio, dim, 1, with_bn=True) # 融合后的总通道是 mid_dim * 2
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.sa = SpatialAttention(kernel_size=7)

    def forward(self, x):
        input = x
        x = self.sa(x) # 空间注意力
        x = self.dwconv(x)
        
        # 1. 计算四个子分支的特征
        x_3A, x_3B = self.f3_A(x), self.f3_B(x)
        x_7A, x_7B = self.f7_A(x), self.f7_B(x)
        
        # 2. 交叉星乘 (Cross-Star Operation) - D (基线)
        # 乘法 1: Local (3A) 调制 Context (7B) -> 强调局部细节在全局语境中的作用
        y12 = self.act(x_3A) * x_7B 
        
        # 乘法 2: Context (7A) 调制 Local (3B) -> 强调全局语境对局部细节的校正
        y21 = self.act(x_7A) * x_3B 
        
        # 3. Concatenate (Inception Style)
        x_out = torch.cat((y12, y21), dim=1) # 沿着通道维度拼接
        
        # 4. 投影回输入维度
        x_out = self.dwconv2(self.g(x_out))
        x_out = input + self.drop_path(x_out)
        return x_out




class StarNet(nn.Module): 
    """ Final StarNet Model with integrated features: -
        GRN (Global Response Normalization) in all blocks - 
        Spatial Attention in shallow layers (Stage 0, 1) - 
        CrossStarBlock in deep layers (Stage 2, 3) """ 
    def __init__(self, base_dim=32, depths=[3, 3, 12, 5], mlp_ratio=4, drop_path_rate=0.0, num_classes=1000, **kwargs): 
        super().__init__() 
        self.num_classes = num_classes 
        self.in_channel = 32 # stem layer 
        self.stem = nn.Sequential(ConvBN(3, self.in_channel, kernel_size=3, stride=2, padding=1), nn.GELU()) 
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] # stochastic depth 
        # build stages 
        self.stages = nn.ModuleList() 
        cur = 0 
        for i_layer in range(len(depths)): 
            embed_dim = base_dim * 2 ** i_layer 
            down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1) 
            self.in_channel = embed_dim 
            # ⭐ Strategy: Hybrid Block Usage 
            # # Stage 0 & 1 (Shallow): Use Block with Spatial Attention (低级特征提取)
            #  # Stage 2 & 3 (Deep): Use CrossStarBlock (多尺度语义融合) 
            if i_layer < 2: # 浅层使用标准 Block + 空间注意力 
                blocks = [Block(self.in_channel, mlp_ratio, dpr[cur + i], use_sa=True) for i in range(depths[i_layer])] 
            else: # 深层使用 CrossStarBlock 
                blocks = [CrossStarBlock(self.in_channel, mlp_ratio, dpr[cur + i]) for i in range(depths[i_layer])] 
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
def starnet_s1_final(pretrained=False, **kwargs):
    model = StarNet(24, [2, 2, 8, 3], **kwargs)
    return model


@register_model
def starnet_s2_final(pretrained=False, **kwargs):
    model = StarNet(32, [1, 2, 6, 2], **kwargs)
    return model


@register_model
def starnet_s3_final(pretrained=False, **kwargs):
    model = StarNet(32, [2, 2, 8, 4], **kwargs)
    return model


