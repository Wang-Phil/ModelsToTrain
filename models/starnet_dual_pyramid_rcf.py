import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model

# ======================================================
# Position Encoding (2D Sin-Cos)
# (保持不变)
# ======================================================
def get_2d_sincos_pos_embed(H, W, C, device=None):
    # Standard ViT/MAE style 2D SinCos positional embedding
    embed_dim = C // 2
    grid_h = torch.arange(H, dtype=torch.float32, device=device)
    grid_w = torch.arange(W, dtype=torch.float32, device=device)
    
    dim_t_size = embed_dim // 2
    dim_t = torch.arange(dim_t_size, dtype=torch.float32, device=device)
    dim_t = 10000.0 ** (2 * dim_t / dim_t_size)

    pos_h = grid_h[:, None] / dim_t[None, :]
    pe_h = torch.cat([torch.sin(pos_h), torch.cos(pos_h)], dim=-1) # [H, C/2]
    
    pos_w = grid_w[:, None] / dim_t[None, :]
    pe_w = torch.cat([torch.sin(pos_w), torch.cos(pos_w)], dim=-1) # [W, C/2]
    
    pe_h = pe_h[:, None, :].expand(-1, W, -1)
    pe_w = pe_w[None, :, :].expand(H, -1, -1)
    
    pos_embed = torch.cat([pe_h, pe_w], dim=-1)  # [H, W, C]
    return pos_embed.flatten(0, 1) # [HW, C]


# ======================================================
# Basic ConvBN, SpatialAttention, StarBlock (保持不变)
# ======================================================
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
        self.bn = nn.GroupNorm(1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.bn(self.conv1(x_cat)))
        return x * scale

class StarBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.0, with_attn=False):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.dwconv = ConvBN(dim, dim, 7, 1, 3, g=dim, bn=True)
        self.f1 = ConvBN(dim, hidden, 1, bn=False)
        self.f2 = ConvBN(dim, hidden, 1, bn=False)
        self.g = ConvBN(hidden, dim, 1, bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, 3, g=dim, bn=False)
        self.act = nn.ReLU()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # self.with_attn = with_attn
        # self.sa = SpatialAttention(kernel_size=7)

    def forward(self, x):
        input = x
        # if self.with_attn:
        #      x = self.sa(x)
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        return input + self.drop_path(x)

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

# 空间注意力机制的PyramidAdapter
# class PyramidAdapter(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.project = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.GroupNorm(1, out_ch))
#         self.attn = SpatialAttention(kernel_size=7)
#         self.output_conv = nn.Sequential(nn.Conv2d(out_ch, out_ch, 1, bias=False), nn.GroupNorm(1, out_ch))
#         self.scale = nn.Parameter(torch.zeros(1))

#     def forward(self, p):
#         shortcut = p
#         attn_out = self.attn(self.project(p))
#         fused_out = self.output_conv(attn_out + shortcut) 
#         return fused_out * self.scale


# ======================================================
# ⭐ Local Pyramid (解耦 Stage)
# ======================================================
class LocalPyramid(nn.Module):
    # stage 拆分为 downsamples 和 block_lists
    def __init__(self, base=32, depths=[3,3,12,5], mlp_ratio=4, dpr=[], use_attn=None):
        super().__init__()
        self.stem = nn.Sequential(ConvBN(3, base, 3, 2, 1), nn.GELU()) 
        
        self.downsamples = nn.ModuleList()
        self.blocks_list = nn.ModuleList()
        
        in_ch = base
        idx = 0
        dpr_len = len(dpr)
        
        for i, d in enumerate(depths):
            out_ch = base * 2 ** i
            
            # 空间注意力判断
            use_attn_here = False
            if use_attn is not None and (use_attn == 0 or i >= use_attn):
                use_attn_here = True
            
            blocks = [
                StarBlock(out_ch, mlp_ratio, dpr[idx + j] if idx + j < dpr_len else 0.0, with_attn=use_attn_here) 
                for j in range(d)
            ]
            
            # 存储 Downsample 和 Blocks
            downsample = nn.Sequential(ConvBN(in_ch, out_ch, 3, 2, 1), nn.GELU())
            self.downsamples.append(downsample)
            self.blocks_list.append(nn.Sequential(*blocks))
            
            idx += d
            in_ch = out_ch
            
    # 注意：这里移除了 forward 函数，其逻辑将在 StarNet_DualPyramid_RCF 中执行


# ======================================================
# Global Transformer Pyramid (保持不变)
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
    # 保持 GlobalPyramid 不变
    def __init__(self, base=32, depths=[1,1,1,1]):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, base, 3, 2, 1, bias=False), nn.BatchNorm2d(base), nn.GELU())
        self.stages = nn.ModuleList()
        in_ch = base
        for i, d in enumerate(depths):
            out_ch = base * 2 ** i
            blocks = [TransformerBlock(out_ch) for _ in range(d)]
            self.stages.append(nn.ModuleDict({
                "down": nn.Conv2d(in_ch, out_ch, 3, 2, 1),
                "pos_proj": nn.Linear(out_ch, out_ch), # 修复为 C -> C
                "norm_in": nn.LayerNorm(out_ch), 
                "blocks": nn.ModuleList(blocks)
            }))
            in_ch = out_ch

    def forward(self, x):
        feats = []
        x = self.stem(x)
        for st in self.stages:
            x = st["down"](x)
            B, C, H, W = x.shape
            t = x.flatten(2).transpose(1, 2)
            pos_embed = get_2d_sincos_pos_embed(H, W, C, device=x.device)
            pos_embed = st["pos_proj"](pos_embed)
            t = t + pos_embed.unsqueeze(0)
            t = st["norm_in"](t) 
            for blk in st["blocks"]:
                t = blk(t)
            x = t.transpose(1,2).reshape(B,C,H,W)
            feats.append(x)
        return feats


# ======================================================
# ⭐ StarNet Dual-Pyramid with Residual Cascaded Fusion (RCF)
# ======================================================
class StarNet_DualPyramid_RCF(nn.Module):
    def __init__(self, base=32, depths=[3,3,12,5], num_classes=1000,
                 mlp_ratio=4, drop_path=0.1, global_depths=None, use_attn=0, dropout_rate=0.1):
        super().__init__()

        total = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path, total)]

        if global_depths is None:
            global_depths = [1, 1, 1, 1]
            
        self.local = LocalPyramid(base, depths, mlp_ratio, dpr, use_attn=use_attn)
        self.global_pyr = GlobalPyramid(base, depths=global_depths)

        self.adapters = nn.ModuleList([
            PyramidAdapter(base*2**i, base*2**i) for i in range(4)
        ])

        self.fuse_weights = nn.ParameterList()
        for i in range(4):
            channels = base * (2 ** i)
            # 通道数越多，越依赖全局上下文（StarNet权重略降低）
            if channels <= base * 2:  # 早期阶段，低通道
                init_val = 0.3  # StarNet占70%
            elif channels <= base * 4:  # 中期阶段
                init_val = 0.4  # StarNet占60%
            else:  # 后期阶段，高通道
                init_val = 0.45  # StarNet占55%
            self.fuse_weights.append(nn.Parameter(torch.ones(1) * init_val))
        
        # self.fuse_weights = nn.ParameterList([
        #     nn.Parameter(torch.ones(1) * 0.5) for _ in range(4) 
        # ])
        
        # ⭐ 新增: 残差级联融合权重 gamma_i (初始化为 0)
        self.gamma_weights = nn.ParameterList([
            nn.Parameter(torch.zeros(1)) for _ in range(4) 
        ])

        # self.dropout = nn.Dropout(dropout_rate)

        self.norm = nn.BatchNorm2d(base * 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base*8, num_classes)

        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
        if isinstance(m, PyramidAdapter):
             nn.init.constant_(m.scale, 0.01)

    def forward(self, x):
        # 1. Global Pyramid 独立运行
        G = self.global_pyr(x)
        
        # 2. Local Pyramid 初始化
        L_current = self.local.stem(x) # L_{-1}
        fused_features = []
        
        # 3. 逐层迭代，执行 RCF 逻辑
        for i in range(4):
            # L_{i+1} = Stage_{i+1} (L_i)
            # L_i = Downsample_i(L_{i-1}) + Blocks_i(Downsample_i(L_{i-1}))
            
            # 3.1. Downsample (L_{i-1} -> L_i_proj)
            L_proj = self.local.downsamples[i](L_current)
            
            # ⭐ 3.2. 残差注入 L_i_proj = L_i_proj + gamma_{i-1} * F_{i-1}
            # F_i 是 L_i 与 G_i 的融合。注入到 L_{i+1} 的输入。
            # 为了实现 L_{i+1} = L_{i+1} + gamma_i * F_i, 我们将 F_i 残差注入 L_{i+1} 的输入 (即 Stage i 的输出)。
            # 
            # 修正：我们应该在 L_i (即 L_current) 产生后，残差注入 F_{i-1}，用于增强 L_i。
            # 由于 L_{i+1} 的特征图大小和通道数可能与 L_i 不同，我们不能直接注入 F_{i-1}。
            #
            # 采用 Stage i 的输出 L_i 注入 Stage i+1 的输入 (L_proj) 的方法更稳定。
            # 然而，您期望的公式是 L_{i+1} = L_{i+1} + gamma_i * F_i。这要求 F_i 与 L_{i+1} 形状匹配。
            
            # --- 采用最接近您公式的实现 (需要 L_i 和 L_{i+1} 尺寸对齐) ---
            
            # L_i_out = Local Pyramid Stage i 的输出 (即 L_i)
            L_i_out = self.local.blocks_list[i](L_proj) # L_i
            
            # F_i = L_i * (1-w) + Adapter(G_i) * w
            A = self.adapters[i](G[i])
            w = torch.sigmoid(self.fuse_weights[i])
            F_i = L_i_out * (1 - w) + A * w
            # F_i = A
            
            # ⭐ Fused_i 存储融合后的特征 (用于下一 Stage 增强和最终分类)
            fused_features.append(F_i)

            # ============================================================
            # ⭐ 修改：只在 Stage3 进行级联残差融合
            # ============================================================
            # 原始逻辑（已注释，方便回退）：
            if i < 3:
                gamma = self.gamma_weights[i]
                L_current = L_i_out + gamma * F_i 
            else:
                L_current = L_i_out # 最后一层不需要级联
            
        # 4. 最终输出
        # 使用最后一个 Stage 的融合特征 F_3
        x = fused_features[-1] 
        
        x = self.pool(self.norm(x))
        x = torch.flatten(x, 1)
        # ⭐ 在全连接层之前加入 Dropout
        # x = self.dropout(x)
        return self.fc(x)


# ======================================================
# Register to timm
# ======================================================
@register_model
def starnet_dual_pyramid_rcf(pretrained=False, **kwargs):
    """
    StarNet Dual-Pyramid with Residual Cascaded Fusion (RCF)
    F_i 残差注入 L_{i+1} 的输入
    """
    return StarNet_DualPyramid_RCF(base=24, depths=[2,2,8,3], global_depths=[1,1,1,1], use_attn=None, **kwargs)