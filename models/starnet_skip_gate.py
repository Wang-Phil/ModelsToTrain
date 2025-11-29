"""
Implementation of StarNet with Dynamic Skip Gating (Stage 2 -> Stage 4).

Adds a dynamic, StarNet-based gated skip connection from the output of Stage 2 
to the input of Stage 4 to enable adaptive long-range feature fusion.
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
        x = self.act(x1) * x2  # Star Operation
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x


# ----------------------------------------------------------------------
# ！！！新增模块 1：跳跃特征投影模块 (SkipProjection) ！！！
# ----------------------------------------------------------------------
class SkipProjection(nn.Module):
    """用于将 Stage i 的特征降采样/投影到 Stage i+k 的尺度。"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 从 Stage 2 (例如 H/8) 到 Stage 4 (例如 H/32) 需要 4x 降采样 (2个步长为2的Conv)
        self.proj = nn.Sequential(
            ConvBN(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            ConvBN(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x):
        return self.proj(x)

# ----------------------------------------------------------------------
# ！！！新增模块 2：动态跳跃门控模块 (DynamicSkipGate) ！！！
# ----------------------------------------------------------------------
class DynamicSkipGate(nn.Module):
    """
    基于 StarNet 的动态门控模块，根据源特征 (F2) 和上下文特征 (F3) 计算权重 beta。
    输出通道权重 [B, C_target, 1, 1]。
    """
    def __init__(self, skip_channels, context_channels, target_channels, reduce_ratio=4):
        super().__init__()
        
        # 门控网络的输入通道数是全局池化后的 skip 和 context 特征之和
        gate_in_channels = skip_channels + context_channels
        reduced_channels = max(gate_in_channels // reduce_ratio, 1)
        
        # 空间信息压缩
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 门控分支 (基于 StarNet 思想)
        self.conv_reduce = ConvBN(gate_in_channels, reduced_channels, 1, with_bn=False)
        self.act = nn.ReLU6()
        
        # Star Operation 分支：映射回目标通道数
        self.conv_expand_f1 = ConvBN(reduced_channels, target_channels, 1, with_bn=False)
        self.conv_expand_f2 = ConvBN(reduced_channels, target_channels, 1, with_bn=False)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, skip_feat, context_feat):
        # 1. 空间压缩并拼接
        skip_pooled = self.avg_pool(skip_feat)       
        context_pooled = self.avg_pool(context_feat) 
        
        y = torch.cat([skip_pooled, context_pooled], dim=1) # 聚合全局信息
        
        # 2. 门控网络计算
        y = self.act(self.conv_reduce(y))
        
        # 3. Star Operation
        y1 = self.conv_expand_f1(y)
        y2 = self.conv_expand_f2(y)
        beta_features = y1 * y2 
        
        # 4. Sigmoid 激活得到门控权重 beta
        beta = self.sigmoid(beta_features)
        
        return beta


# ----------------------------------------------------------------------
# ！！！修改 StarNet 类 ！！！
# ----------------------------------------------------------------------
class StarNet(nn.Module):
    def __init__(self, base_dim=32, depths=[3, 3, 12, 5], mlp_ratio=4, drop_path_rate=0.0, num_classes=1000, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.base_dim = base_dim # 保存 base_dim
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
            
        # ！！！动态跳跃门控初始化 (Stage 2 -> Stage 4) ！！！
        C2 = base_dim * 2 ** 1  # Stage 2 输出通道数 (stages[1]的输出)
        C3 = base_dim * 2 ** 2  # Stage 3 输出通道数 (stages[2]的输出)
        C4_in = base_dim * 2 ** 3 # Stage 4 输入通道数 (DownSampler之后，即 C3)

        # C4_proj_out 是 Stage 4 Blocks 的输入通道数，即 C4_in
        
        # 1. 跳跃特征投影 (C2 -> C4_in)
        self.skip_proj_2_to_4 = SkipProjection(C2, C4_in)
        
        # 2. 动态门控模块 (C2_out 作为 skip_channels, C3_out 作为 context_channels, C4_in 作为 target_channels)
        self.dynamic_gate_2_to_4 = DynamicSkipGate(C2, C3, C4_in) 
        
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
        
        F2_out = None # 存储 Stage 2 输出
        
        for i, stage in enumerate(self.stages):
            if i == 0: # Stage 1
                x = stage(x) 
            
            elif i == 1: # Stage 2
                x = stage(x) 
                F2_out = x # 记录 Stage 2 输出
            
            elif i == 2: # Stage 3
                F3_in = x # F3_in = F2_out
                x = stage(x) # x 变为 F3_out
                F3_out = x
            
            elif i == 3: # Stage 4 (关键的融合点)
                # Stage 4 = [DownSampler_4, Block_4_1, ...]
                downsample_4 = stage[0] 
                block_sequence_4 = stage[1:] 
                
                # 1. 运行 DownSampler_4 (x 变为 F4_in_aligned)
                x = downsample_4(x)
                F4_in_main = x # 主分支特征

                # 2. 对齐跳跃特征
                F2_aligned = self.skip_proj_2_to_4(F2_out)
                
                # 3. 计算动态门控权重 beta
                # 使用 F2_out (跳跃源) 和 F3_out (当前上下文) 计算门控权重
                beta = self.dynamic_gate_2_to_4(F2_out, F3_out)
                
                # 4. 动态跳跃门控融合
                gated_skip_feat = beta * F2_aligned
                
                # 5. 融合：将门控后的跳跃特征加到 Stage 4 的输入中
                x = F4_in_main + gated_skip_feat 
                
                # 6. 执行 Stage 4 的 Blocks
                for block in block_sequence_4:
                    x = block(x)
                
        x = torch.flatten(self.avgpool(self.norm(x)), 1)
        return self.head(x)


# ----------------------------------------------------------------------
# ！！！注册新模型 ！！！
# ----------------------------------------------------------------------
@register_model
def starnet_s1_gated_skip(pretrained=False, **kwargs):
    # 使用 StarNet-S4 的参数作为基准
    model = StarNet(24, [3, 3, 8, 3], **kwargs) 
    if pretrained:
        # 注意：预训练权重不包含新的门控模块，加载时需要设置 strict=False
        url = model_urls['starnet_s1']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"], strict=False) 
    return model

