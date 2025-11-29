import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model

# 定义全局 Dropout 率
TRANSFORMER_ATTN_DROP = 0.0
TRANSFORMER_PROJ_DROP = 0.1 # 常规 FFN Dropout 率

# ======================================================
# Position Encoding (2D Sin-Cos)
# ======================================================
def get_2d_sincos_pos_embed(H, W, C, device=None):
    """
    生成2D正弦余弦位置编码 (MAE/ViT style)
    C: 期望的输出维度 (例如，模型通道数)。该实现输出的维度是 C。
    返回: [H*W, C]
    """
    # 保持原代码中的 C 逻辑，但调整注释以匹配 ViT/MAE 风格的标准实现
    # H轴和W轴分别编码 C/2 维, 最终拼接为 C
    
    embed_dim = C // 2
    
    grid_h = torch.arange(H, dtype=torch.float32, device=device)
    grid_w = torch.arange(W, dtype=torch.float32, device=device)
    
    # 维度计算
    dim_t_size = embed_dim // 2
    dim_t = torch.arange(dim_t_size, dtype=torch.float32, device=device)
    dim_t = 10000.0 ** (2 * dim_t / dim_t_size)

    # H 轴编码 (sin + cos -> C/2 维)
    pos_h = grid_h[:, None] / dim_t[None, :]
    pe_h = torch.cat([torch.sin(pos_h), torch.cos(pos_h)], dim=-1) # [H, C/2]
    
    # W 轴编码 (sin + cos -> C/2 维)
    pos_w = grid_w[:, None] / dim_t[None, :]
    pe_w = torch.cat([torch.sin(pos_w), torch.cos(pos_w)], dim=-1) # [W, C/2]
    
    # 扩展并在空间维度拼接 [H, W, C]
    pe_h = pe_h[:, None, :].expand(-1, W, -1)
    pe_w = pe_w[None, :, :].expand(H, -1, -1)
    
    pos_embed = torch.cat([pe_h, pe_w], dim=-1)  # [H, W, C]
    return pos_embed.flatten(0, 1) # [HW, C]


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

class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation (SE) Block 风格的通道注意力
    """
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        scale = self.fc(y)
        return x * scale 


# ======================================================
# ⭐ Upgraded StarBlock (保持不变)
# ======================================================
class StarBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)

        self.dw = ConvBN(dim, dim, 7, 1, 3, g=dim)
        self.f1 = nn.Sequential(nn.Conv2d(dim, hidden, 1), nn.GELU())
        self.f2 = nn.Sequential(nn.Conv2d(dim, hidden, 1), nn.Sigmoid())
        self.mix = ConvBN(hidden, dim, 1)
        self.dw2 = ConvBN(dim, dim, 7, 1, 3, g=dim, bn=False)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.dw(x)
        x = self.f1(x) * self.f2(x)
        x = self.dw2(self.mix(x))
        return shortcut + self.drop_path(x)


class PyramidAdapter(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.GroupNorm(1, out_ch)
        )
        self.output_conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1, bias=False),
            nn.GroupNorm(1, out_ch)
        )
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, p):
        shortcut = p
        proj = self.project(p)
        fused = self.output_conv(proj + shortcut)
        return fused * self.scale

# ======================================================
# Local Pyramid (StarNet Pyramid) (保持不变)
# ======================================================
class LocalPyramid(nn.Module):
    def __init__(self, base=32, depths=[3,3,12,5], mlp_ratio=4, dpr=[]):
        super().__init__()
        self.stem = nn.Sequential(ConvBN(3, base, 3, 2, 1), nn.GELU()) 
        stages = []
        in_ch = base
        idx = 0
        dpr_len = len(dpr)
        
        for i, d in enumerate(depths):
            out_ch = base * 2 ** i
            blocks = [
                StarBlock(out_ch, mlp_ratio, dpr[idx + j] if idx + j < dpr_len else 0.0) 
                for j in range(d)
            ]
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
        return feats 


# ======================================================
# ⭐ Global Transformer Pyramid (Dropout/DropPath 集成)
# ======================================================
class TransformerBlock(nn.Module):
    # ⭐ 引入 drop_path, attn_drop, proj_drop
    def __init__(self, dim, heads=4, mlp_ratio=4, drop_path=0.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.attn_dropout = nn.Dropout(attn_drop) # Attn 输出后的 Dropout
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(proj_drop) # FFN 输出后的 Dropout
        )

    def forward(self, x):
        # 1. Attention + Dropout + Residual
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.drop_path(self.attn_dropout(attn_out)) 
        
        # 2. FFN + Dropout + Residual
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class GlobalPyramid(nn.Module):
    # ⭐ 接收 dpr 列表
    def __init__(self, base=32, depths=[1,1,1,1], dpr=[]): 
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, base, 3, 2, 1, bias=False),
            nn.BatchNorm2d(base), nn.GELU()
        )

        stages = []
        in_ch = base
        dpr_idx = 0
        dpr_len = len(dpr)
        
        for i, d in enumerate(depths):
            out_ch = base * 2 ** i
            
            # ⭐ 分配 DropPath rate
            blocks = []
            for j in range(d):
                 dpr_rate = dpr[dpr_idx] if dpr_idx < dpr_len else 0.0
                 blocks.append(TransformerBlock(
                    dim=out_ch, 
                    drop_path=dpr_rate, # DropPath rate
                    attn_drop=TRANSFORMER_ATTN_DROP, 
                    proj_drop=TRANSFORMER_PROJ_DROP 
                 ))
                 dpr_idx += 1


            stages.append(nn.ModuleDict({
                "down": nn.Conv2d(in_ch, out_ch, 3, 2, 1),
                "pos_proj": nn.Linear(out_ch, out_ch), # 2D-SinCos 已经输出 C 维，无需 x2
                "norm_in": nn.LayerNorm(out_ch), 
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
            t = x.flatten(2).transpose(1, 2) 
            
            # get_2d_sincos_pos_embed 返回 [HW, C]
            pos_embed = get_2d_sincos_pos_embed(H, W, C, device=x.device) 
            
            # pos_proj: [HW, C] -> [HW, C] (用于可学习的PE)
            pos_embed = st["pos_proj"](pos_embed) 
            t = t + pos_embed.unsqueeze(0)
            
            t = st["norm_in"](t) 

            for blk in st["blocks"]:
                t = blk(t)

            x = t.transpose(1,2).reshape(B,C,H,W)
            feats.append(x)
        return feats


# ======================================================
# Dual-Pyramid Fusion Network (保持不变)
# ======================================================
class StarNet_DualPyramid(nn.Module):
    def __init__(self, base=32, depths=[3,3,12,5], num_classes=1000,
                 mlp_ratio=4, drop_path=0.1, global_depths=None):
        super().__init__()

        # --- Local Pyramid Setup ---
        local_total = sum(depths)
        local_dpr = [x.item() for x in torch.linspace(0, drop_path, local_total)]
        self.local = LocalPyramid(base, depths, mlp_ratio, local_dpr)

        # --- Global Pyramid Setup ---
        if global_depths is None:
            global_depths = [1, 1, 1, 1]
        global_total = sum(global_depths)
        global_dpr_rates = [x.item() for x in torch.linspace(0, drop_path, global_total)]
            
        self.global_pyr = GlobalPyramid(base, depths=global_depths, dpr=global_dpr_rates)

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
        # 确保 Adapter Scale 初始化
        if isinstance(m, PyramidAdapter):
             nn.init.constant_(m.scale, 0.01)

    def forward(self, x):
        L = self.local(x)
        G = self.global_pyr(x)

        for i in range(4):
            L[i] = L[i] + self.adapters[i](G[i])

        x = self.pool(self.norm(L[-1]))
        x = torch.flatten(x, 1)
        return self.fc(x)


# ======================================================
# Register to timm
# ======================================================
@register_model
def starnet_dual_pyramid(pretrained=False, **kwargs):
    # 保持参数一致性
    return StarNet_DualPyramid(base=24, depths=[2,2,8,3], global_depths=[1,1,1,1], **kwargs)