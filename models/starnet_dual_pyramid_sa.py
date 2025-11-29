import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model

# ======================================================
# Position Encoding (2D Sin-Cos)
# ======================================================
def get_2d_sincos_pos_embed(H, W, C, device=None):
    """
    生成2D正弦余弦位置编码 (MAE/ViT style)
    每个方向 C//2，sin + cos 合计 C。两个方向concat后是 2*C。
    C: 单轴编码维度 (总维度为 2*C)
    返回: [H*W, 2*C]
    """
    embed_dim = C // 2
    
    grid_h = torch.arange(H, dtype=torch.float32, device=device)
    grid_w = torch.arange(W, dtype=torch.float32, device=device)
    grid = torch.meshgrid(grid_h, grid_w, indexing="ij")

    pos_embed = []
    
    # Base for 2*k / C_half part
    dim_t = torch.arange(embed_dim, dtype=torch.float32, device=device)
    dim_t = 10000.0 ** (2 * dim_t / embed_dim) # 确保浮点
    
    for g in grid:
        pos = g[:, :, None] / dim_t[None, None, :]
        pe = torch.cat([torch.sin(pos), torch.cos(pos)], dim=-1)  # [H, W, C]
        pos_embed.append(pe)

    pos_embed = torch.cat(pos_embed, dim=-1)  # [H, W, 2*C]
    return pos_embed.flatten(0, 1)  # [HW, 2*C]


# ======================================================
# Spatial Attention (from starnet.py)
# ======================================================
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.bn = nn.GroupNorm(1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. 沿着通道维度做 AvgPool 和 MaxPool
        # x: [B, C, H, W] -> avg_out: [B, 1, H, W], max_out: [B, 1, H, W]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # 2. 拼接
        x_cat = torch.cat([avg_out, max_out], dim=1)
        
        # 3. 卷积 + Sigmoid 生成空间掩码
        scale = self.sigmoid(self.bn(self.conv1(x_cat)))
        
        # 4. 施加注意力
        return x * scale


# ======================================================
# Basic ConvBN (Optimized Style)
# ======================================================
class ConvBN(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=0, g=1, bn=True):
        modules = [
            nn.Conv2d(in_ch, out_ch, k, s, p, groups=g, bias=False)
        ]
        if bn:
            modules.append(nn.BatchNorm2d(out_ch))
        super().__init__(*modules)


# ======================================================
# ⭐ Upgraded StarBlock with Spatial Attention
# ======================================================
class StarBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.0, with_attn=False):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, g=dim, bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, g=dim, bn=False)
        self.act = nn.ReLU()
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


# ======================================================
# Pyramid Adapter (Optimized with Residual)
# ======================================================
class PyramidAdapter(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # 1. 投影层: 替换 ConvBN 为 Conv + GroupNorm
        self.project = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(1, out_ch)
        )
        # 使用统一的 SpatialAttention 替代 PSA_v2
        self.attn = SpatialAttention(kernel_size=7)
        # 2. 融合后的 1x1 卷积 + BN (用于整合注意力输出和残差)
        self.output_conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.GroupNorm(1, out_ch) # <--- 修复
        )
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, p):
        shortcut = p # Transformer特征作为shortcut
        
        # Attn Path: Proj -> Attn
        attn_out = self.attn(self.project(p))
        
        # 融合: Attn_out + Shortcut, 再通过 1x1 Conv+BN
        fused_out = self.output_conv(attn_out + shortcut) 
        
        return fused_out * self.scale


# ======================================================
# Local Pyramid (StarNet Pyramid) with Spatial Attention
# ======================================================
class LocalPyramid(nn.Module):
    def __init__(self, base=32, depths=[3,3,12,5], mlp_ratio=4, dpr=[], use_attn=None):
        super().__init__()
        # Stem: Conv + BN + GELU
        self.stem = nn.Sequential(ConvBN(3, base, 3, 2, 1), nn.GELU()) 
        stages = []
        in_ch = base
        idx = 0
        dpr_len = len(dpr)
        
        for i, d in enumerate(depths):
            out_ch = base * 2 ** i
            
            # 决定是否使用空间注意力
            use_attn_here = False
            if use_attn is not None:
                if use_attn == 0:  # 所有stage都使用
                    use_attn_here = True
                elif i >= use_attn:  # 从指定stage开始使用
                    use_attn_here = True
            
            blocks = [
                StarBlock(out_ch, mlp_ratio, dpr[idx + j] if idx + j < dpr_len else 0.0, with_attn=use_attn_here) 
                for j in range(d)
            ]
            
            # 优化点：在下采样后添加 GELU 激活函数
            downsample = nn.Sequential(ConvBN(in_ch, out_ch, 3, 2, 1), nn.GELU())
            stages.append(nn.Sequential(downsample, *blocks))
            
            idx += d
            in_ch = out_ch
        self.stages = nn.ModuleList(stages)

    def forward(self, x):
        feats = []
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats  # S4 S8 S16 S32


# ======================================================
# Global Transformer Pyramid (Added LayerNorm for stability)
# ======================================================
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*mlp_ratio),
            nn.GELU(),
            nn.Linear(dim*mlp_ratio, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class GlobalPyramid(nn.Module):
    def __init__(self, base=32, depths=[1,1,1,1]):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, base, 3, 2, 1, bias=False),
            nn.BatchNorm2d(base), nn.GELU()
        )

        stages = []
        in_ch = base
        for i, d in enumerate(depths):
            out_ch = base * 2 ** i
            blocks = [TransformerBlock(out_ch) for _ in range(d)]

            stages.append(nn.ModuleDict({
                "down": nn.Conv2d(in_ch, out_ch, 3, 2, 1),
                "pos_proj": nn.Linear(out_ch * 2, out_ch), # 投影 2C -> C
                "norm_in": nn.LayerNorm(out_ch), # 优化点：Transformer输入前的 Norm
                "blocks": nn.ModuleList(blocks)
            }))
            in_ch = out_ch

        self.stages = nn.ModuleList(stages)

    def forward(self, x):
        feats = []
        x = self.stem(x)

        for st in self.stages:
            x = st["down"](x)
            B, C, H, W = x.shape
            t = x.flatten(2).transpose(1, 2)  # [B, HW, C]
            
            # 修复点：传入完整的通道数 C
            pos_embed = get_2d_sincos_pos_embed(H, W, C, device=x.device) # [HW, 2*C]
            
            # pos_proj: [HW, 2*C] -> [HW, C]
            pos_embed = st["pos_proj"](pos_embed)  # [HW, C]
            t = t + pos_embed.unsqueeze(0)  # Add Positional Embedding
            
            # LayerNorm before Transformer Blocks (您的优化)
            t = st["norm_in"](t) 

            for blk in st["blocks"]:
                t = blk(t)

            x = t.transpose(1,2).reshape(B,C,H,W)
            feats.append(x)
        return feats


# ======================================================
# ⭐ Dual-Pyramid Fusion Network with Spatial Attention
# ======================================================
class StarNet_DualPyramid_SA(nn.Module):
    def __init__(self, base=32, depths=[3,3,12,5], num_classes=1000,
                 mlp_ratio=4, drop_path=0.1, global_depths=None, use_attn=0):
        """
        Args:
            base: 基础通道数
            depths: Local Pyramid 每个stage的block数量
            num_classes: 分类类别数
            mlp_ratio: MLP扩展比例
            drop_path: DropPath率
            global_depths: Global Pyramid 每个stage的Transformer block数量
            use_attn: 空间注意力使用策略
                - 0: 所有stage都使用 (类似starnet_s2)
                - 1: 从Stage 1开始使用
                - 2: 从Stage 2开始使用
                - 3: 仅Stage 3使用
                - None: 不使用空间注意力
        """
        super().__init__()

        total = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path, total)]

        if global_depths is None:
            global_depths = [1, 1, 1, 1]

        # 🌟 关键: 引入可学习的融合权重
        self.fuse_weights = nn.ParameterList([
            nn.Parameter(torch.ones(1) * 0.5) for _ in range(4) 
        ])
            
        self.local = LocalPyramid(base, depths, mlp_ratio, dpr, use_attn=use_attn)
        self.global_pyr = GlobalPyramid(base, depths=global_depths)

        self.adapters = nn.ModuleList([
            PyramidAdapter(base*2**i, base*2**i) for i in range(4)
        ])

        self.norm = nn.BatchNorm2d(base * 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base*8, num_classes)

        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)

    def forward(self, x):
        L = self.local(x)
        G = self.global_pyr(x)

        for i in range(4):
            A = self.adapters[i](G[i]) # 适配器输出
            # 🌟 融合: L[i] = L[i] * (1 - w_i) + A * w_i
            # 这里的 L[i] 现在是 Local Feature (BN style)，A 是 Adapter Feature (GN style)
            # 通过可学习的 w_i 来控制融合比例
            w = torch.sigmoid(self.fuse_weights[i]) # 限制 w 在 (0, 1)
            L[i] = L[i] * (1 - w) + A * w
        

        x = self.pool(self.norm(L[-1]))
        x = torch.flatten(x, 1)
        return self.fc(x)


# ======================================================
# Register to timm
# ======================================================
@register_model
def starnet_dual_pyramid_sa(pretrained=False, **kwargs):
    """
    StarNet Dual-Pyramid with Spatial Attention
    在Local Pyramid的所有StarBlock中使用空间注意力机制
    """
    return StarNet_DualPyramid_SA(base=24, depths=[2,2,8,3], global_depths=[1,1,1,1], use_attn=0, **kwargs)

