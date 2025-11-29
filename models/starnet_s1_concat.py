"""
Modified StarNet with Inception-style Cross-Star Operation.

Innovation: Cross-Star Block replaces the standard Star Operation with 
a multi-scale, cross-modulation mechanism, addressing the local-global challenge.
"""

import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model

# 保持原有的 ConvBN 辅助类
class ConvBN(torch.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_module('conv', torch.nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module('bn', torch.nn.BatchNorm2d(out_planes))
            torch.nn.init.constant_(self.bn.weight, 1)
            torch.nn.init.constant_(self.bn.bias, 0)


# =========================================================================
#  Innovation Part: Cross-Star Block
# =========================================================================

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

    def forward(self, x):
        input = x
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


class CrossStarBlock_Add(nn.Module):
    """
    [D1 - 消融实验] Cross-Star Block with Addition (替代乘法)
    实现了 Y = Concat((x_{3A} + x_{7B}), (x_{7A} + x_{3B}))
    目的：证明乘法的非线性调制效果（对比基线D）
    """
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        
        # 保持原有的输入处理 (7x7 Depthwise Conv)
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        
        # 定义中间维度
        self.mid_dim = (dim * mlp_ratio) // 2
        
        # --- Multi-Scale Branches ---
        self.f3_A = ConvBN(dim, self.mid_dim, 3, 1, 1, with_bn=False)
        self.f3_B = ConvBN(dim, self.mid_dim, 3, 1, 1, with_bn=False)
        self.f7_A = ConvBN(dim, self.mid_dim, 7, 1, 3, with_bn=False)
        self.f7_B = ConvBN(dim, self.mid_dim, 7, 1, 3, with_bn=False)
        
        # 融合与输出
        self.g = ConvBN(dim * mlp_ratio, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        
        # 1. 计算四个子分支的特征
        x_3A, x_3B = self.f3_A(x), self.f3_B(x)
        x_7A, x_7B = self.f7_A(x), self.f7_B(x)
        
        # 2. 交叉加法 (Cross-Add Operation) - D1
        # 加法 1: Local (3A) + Context (7B)
        y12 = self.act(x_3A) + x_7B 
        
        # 加法 2: Context (7A) + Local (3B)
        y21 = self.act(x_7A) + x_3B 
        
        # 3. Concatenate
        x_out = torch.cat((y12, y21), dim=1)
        
        # 4. 投影回输入维度
        x_out = self.dwconv2(self.g(x_out))
        x_out = input + self.drop_path(x_out)
        return x_out


class CrossStarBlock_SameScale(nn.Module):
    """
    [D2 - 消融实验] Cross-Star Block with Same-Scale Star (同尺度相乘)
    实现了 Y = Concat((x_{3A} * x_{3B}), (x_{7A} * x_{7B}))
    目的：证明交叉融合多尺度的优势（对比基线D）
    """
    def __init__(self, dim, mlp_ratio=3, drop_path=0.):
        super().__init__()
        
        # 保持原有的输入处理 (7x7 Depthwise Conv)
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        
        # 定义中间维度
        self.mid_dim = (dim * mlp_ratio) // 2
        
        # --- Multi-Scale Branches ---
        self.f3_A = ConvBN(dim, self.mid_dim, 3, 1, 1, with_bn=False)
        self.f3_B = ConvBN(dim, self.mid_dim, 3, 1, 1, with_bn=False)
        self.f7_A = ConvBN(dim, self.mid_dim, 7, 1, 3, with_bn=False)
        self.f7_B = ConvBN(dim, self.mid_dim, 7, 1, 3, with_bn=False)
        
        # 融合与输出
        self.g = ConvBN(dim * mlp_ratio, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        
        # 1. 计算四个子分支的特征
        x_3A, x_3B = self.f3_A(x), self.f3_B(x)
        x_7A, x_7B = self.f7_A(x), self.f7_B(x)
        
        # 2. 同尺度星乘 (Same-Scale Star Operation) - D2
        # Star 1: Local (3A) * Local (3B) -> 同尺度局部特征
        y3 = self.act(x_3A) * x_3B 
        
        # Star 2: Context (7A) * Context (7B) -> 同尺度全局特征
        y7 = self.act(x_7A) * x_7B 
        
        # 3. Concatenate
        x_out = torch.cat((y3, y7), dim=1)
        
        # 4. 投影回输入维度
        x_out = self.dwconv2(self.g(x_out))
        x_out = input + self.drop_path(x_out)
        return x_out


# =========================================================================
#  Original Block (Preserved for shallow layers)
# =========================================================================

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

# =========================================================================
#  Main Model (Modified to integrate CrossStarBlock)
# =========================================================================

class StarNet(nn.Module):
    def __init__(self, base_dim=32, depths=[3, 3, 12, 5], mlp_ratio=4, drop_path_rate=0.0, num_classes=1000, 
                 deep_block_type=None, **kwargs):
        """
        Args:
            deep_block_type: 深层使用的Block类型
                - None 或 'cross_star': 使用 CrossStarBlock (基线 D)
                - 'cross_star_add': 使用 CrossStarBlock_Add (D1)
                - 'cross_star_samescale': 使用 CrossStarBlock_SameScale (D2)
        """
        super().__init__()
        self.num_classes = num_classes
        self.in_channel = 32
        
        # stem layer
        self.stem = nn.Sequential(ConvBN(3, self.in_channel, kernel_size=3, stride=2, padding=1), nn.ReLU6())
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] # stochastic depth
        
        # 确定深层使用的Block类型
        if deep_block_type is None or deep_block_type == 'cross_star':
            DeepBlockType = CrossStarBlock
        elif deep_block_type == 'cross_star_add':
            DeepBlockType = CrossStarBlock_Add
        elif deep_block_type == 'cross_star_samescale':
            DeepBlockType = CrossStarBlock_SameScale
        else:
            raise ValueError(f"Unknown deep_block_type: {deep_block_type}")
        
        # build stages
        self.stages = nn.ModuleList()
        cur = 0
        for i_layer in range(len(depths)):
            embed_dim = base_dim * 2 ** i_layer
            down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1)
            self.in_channel = embed_dim
            
            # --- Strategy: Hybrid Block Usage ---
            # Stage 0 & 1 (Shallow): Use Standard Block (低级特征提取)
            # Stage 2 & 3 (Deep): Use specified Block type (多尺度语义融合)
            if i_layer < 2:
                BlockType = Block
            else:
                BlockType = DeepBlockType
            # ------------------------------------

            blocks = [BlockType(self.in_channel, mlp_ratio, dpr[cur + i]) for i in range(depths[i_layer])]
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


# =========================================================================
#  Model Builders (Registration)
# =========================================================================

# (注册函数保持原样，它们会调用上面修改后的 StarNet 类)

model_urls = {
    "starnet_s1": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s1.pth.tar",
    "starnet_s2": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s2.pth.tar",
    "starnet_s3": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s3.pth.tar",
    "starnet_s4": "https://github.com/ma-xu/Rewrite-the-Stars/releases/download/checkpoints_v1/starnet_s4.pth.tar",
}


# =========================================================================
#  Model Registration: Baseline and Ablation Studies
# =========================================================================

@register_model
def starnet_s1_cross_star(pretrained=False, **kwargs):
    """
    [D - 基线] Cross-Star Block with Cross Multiplication
    Y = Concat((x_{3A} * x_{7B}), (x_{7A} * x_{3B}))
    """
    model = StarNet(24, [2, 2, 8, 3], deep_block_type='cross_star', **kwargs)
    return model


@register_model
def starnet_s1_cross_star_add(pretrained=False, **kwargs):
    """
    [D1 - 消融实验] Cross-Star Block with Addition (替代乘法)
    Y = Concat((x_{3A} + x_{7B}), (x_{7A} + x_{3B}))
    目的：证明乘法的非线性调制效果
    """
    model = StarNet(24, [2, 2, 8, 3], deep_block_type='cross_star_add', **kwargs)
    return model


@register_model
def starnet_s1_cross_star_samescale(pretrained=False, **kwargs):
    """
    [D2 - 消融实验] Cross-Star Block with Same-Scale Star (同尺度相乘)
    Y = Concat((x_{3A} * x_{3B}), (x_{7A} * x_{7B}))
    目的：证明交叉融合多尺度的优势
    """
    model = StarNet(24, [2, 2, 8, 3], deep_block_type='cross_star_samescale', **kwargs)
    return model


if __name__ == '__main__':
    # Simple Test to verify dimension compatibility
    model = StarNet(base_dim=32, depths=[1, 1, 2, 1], num_classes=10)
    input_tensor = torch.randn(2, 3, 224, 224)
    output = model(input_tensor)
    print(f"Model Output Shape: {output.shape}")
    print("Cross-StarNet initialized and passed forward test successfully.")