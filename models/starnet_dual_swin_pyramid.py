import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model

# 定义窗口大小
WINDOW_SIZE = 7

# ==========================================
# 1. 新增： 空间注意力机制模块
# ==========================================
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
        scale = self.sigmoid(self.conv1(x_cat))
        
        # 4. 施加注意力
        return x * scale
# ==========================================

# ======================================================
# Position Encoding (2D Sin-Cos)
# ======================================================
def get_2d_sincos_pos_embed(H, W, C, device=None):
    """
    生成2D正弦余弦位置编码 (ViT/MAE Style: H轴和W轴分别编码 C/2 维, 最终拼接为 C)
    输出形状: [H*W, C]
    """
    embed_dim = C // 2
    
    grid_h = torch.arange(H, dtype=torch.float32, device=device)
    grid_w = torch.arange(W, dtype=torch.float32, device=device)
    
    # 每个轴编码 C/2 维度，分为 C/4 的 sin 和 C/4 的 cos
    if embed_dim % 2 != 0:
        # Handle odd C/2 dimensions (rare, but for robustness)
        dim_t_size = embed_dim // 2
    else:
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
# 辅助函数: Window Partition/Reverse/Mask (Swin Style)
# ======================================================
def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W, C):
    num_windows_h = H // window_size
    num_windows_w = W // window_size
    B = int(windows.shape[0] / (num_windows_h * num_windows_w))
    x = windows.view(B, num_windows_h, num_windows_w, window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)
    return x

def create_mask(H, W, window_size, shift_size, device):
    img_mask = torch.zeros((1, H, W, 1), device=device)
    h_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    w_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.view(-1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    # 避免跨窗口计算
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


# ======================================================
# Window Attention (W-MSA / SW-MSA)
# ======================================================
class WindowAttention(nn.Module):
    # ⭐ 添加 attn_drop/proj_drop
    def __init__(self, dim, heads, window_size, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        # x: [N_win, WsWs, C]
        N_win, WsWs, C = x.shape
        
        qkv = self.qkv(x).reshape(N_win, WsWs, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # q, k, v: [N_win, heads, WsWs, head_dim]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # 应用 Mask
        if mask is not None:
            nW = mask.shape[0]
            # 确保 mask 在 heads 和 batch 维度广播正确
            attn = attn.view(N_win // nW, nW, self.heads, WsWs, WsWs) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.heads, WsWs, WsWs)
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Output = Attn @ V
        x = (attn @ v).transpose(1, 2).reshape(N_win, WsWs, C)
        
        # Output Projection
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


# ======================================================
# Swin Transformer Block (W-MSA / SW-MSA)
# ======================================================
class SwinTransformerBlock(nn.Module):
    # ⭐ 启用 Dropout 参数
    def __init__(self, dim, heads=4, mlp_ratio=4, drop_path=0.0, 
                 window_size=WINDOW_SIZE, shift_size=0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.shift_size = shift_size
        self.window_size = window_size
        
        self.norm1 = nn.LayerNorm(dim)
        # 传入 Dropouts
        self.attn = WindowAttention(dim, heads, window_size, attn_drop=attn_drop, proj_drop=proj_drop) 
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, H, W, attn_mask=None): # 必须传入 H, W, Mask
        shortcut = x
        x = self.norm1(x)
        B, L, C = x.shape
        
        x = x.view(B, H, W, C)

        # 1. 循环位移 (Cyclic Shift)
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        
        # 2. Window Partition & Flatten
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # 3. W-MSA / SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # 4. Window Reverse & Reshape
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, H, W, C)

        # 5. 反向循环位移 (Reverse Cyclic Shift)
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        
        # 6. 残差连接 + DropPath
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        
        # 7. FFN + DropPath
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ======================================================
# Basic ConvBN (Local Pyramid Helper)
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
# StarBlock (Local Pyramid Core)
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
        # self.sa = SpatialAttention(kernel_size=7)

    def forward(self, x):
        shortcut = x
        # x = self.sa(x)
        x = self.dw(x)
        x = self.f1(x) * self.f2(x)
        x = self.dw2(self.mix(x))
        return shortcut + self.drop_path(x)


# ======================================================
# Pyramid Adapter (Feature Fusion)
# ======================================================
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
        # ⭐ 修正 Adapter scale，初始化时通过 _init 统一设置为 0.01
        self.scale = nn.Parameter(torch.zeros(1)) 

    def forward(self, p):
        shortcut = p
        proj = self.project(p)
        fused = self.output_conv(proj + shortcut)
        return fused * self.scale


# ======================================================
# Local Pyramid (CNN based StarBlock stages)
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
# Global Transformer Pyramid (Swin style stages)
# ======================================================
class GlobalPyramid(nn.Module):
    # ⭐ 接受 dpr 列表
    def __init__(self, base=32, depths=[2, 2, 2, 2], dpr=[]):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, base, 3, 2, 1, bias=False),
            nn.BatchNorm2d(base), nn.GELU()
        )
        stages = []
        in_ch = base
        dpr_idx = 0
        
        for i, d in enumerate(depths):
            out_ch = base * 2 ** i
            
            # W-MSA 和 SW-MSA 交替
            blocks = []
            for j in range(d):
                shift = 0 if j % 2 == 0 else WINDOW_SIZE // 2
                dpr_rate = dpr[dpr_idx] if dpr_idx < len(dpr) else 0.0
                
                blocks.append(SwinTransformerBlock(
                    dim=out_ch, 
                    heads=4, 
                    mlp_ratio=4, 
                    drop_path=dpr_rate, # 使用分配的 DropPath rate
                    window_size=WINDOW_SIZE, 
                    shift_size=shift,
                    attn_drop=0.0,      # 可根据需要调整，0.0 或 0.1
                    proj_drop=0.0       # 可根据需要调整，0.0 或 0.1
                ))
                dpr_idx += 1

            # 修正 3: downsample 增加 BN + GELU
            downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, 2, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.GELU()
            )

            # 修正 1: pos_proj 维度 C -> C
            pos_proj = nn.Linear(out_ch, out_ch) 

            stages.append(nn.ModuleDict({
                "down": downsample,
                "pos_proj": pos_proj,
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
            
            # 确保输入尺寸可被窗口整除
            if H % WINDOW_SIZE != 0 or W % WINDOW_SIZE != 0:
                 raise ValueError(f"Feature map size ({H}x{W}) must be divisible by WINDOW_SIZE ({WINDOW_SIZE}).")

            t = x.flatten(2).transpose(1, 2)  # [B, HW, C]
            
            # Positional Embedding
            pos_embed = get_2d_sincos_pos_embed(H, W, C, device=x.device)
            pos_embed = st["pos_proj"](pos_embed)
            t = t + pos_embed.unsqueeze(0)
            t = st["norm_in"](t) 
            
            # SW-MSA Mask
            attn_mask = None
            if len(st["blocks"]) > 1 and st["blocks"][1].shift_size > 0:
                # 理论上只需要计算一次，但为确保 H, W 匹配，在此处计算
                shift_size = st["blocks"][1].shift_size
                attn_mask = create_mask(H, W, WINDOW_SIZE, shift_size, x.device)

            for j, blk in enumerate(st["blocks"]):
                current_mask = attn_mask if (j % 2 == 1 and attn_mask is not None) else None
                t = blk(t, H, W, current_mask) 

            x = t.transpose(1,2).reshape(B,C,H,W)
            feats.append(x)
        return feats


# ======================================================
# StarNet Dual-SwinPyramid (Main Model)
# ======================================================
class StarNet_DualSwinPyramid(nn.Module):
    def __init__(self, base=32, depths=[3,3,12,5], num_classes=1000,
                 mlp_ratio=4, drop_path=0.1, global_depths=None):
        super().__init__()

        # --- Local Pyramid Setup ---
        local_total = sum(depths)
        local_dpr = [x.item() for x in torch.linspace(0, drop_path, local_total)]
        self.local = LocalPyramid(base, depths, mlp_ratio, local_dpr)

        # --- Global Pyramid Setup ---
        if global_depths is None:
            global_depths = [2, 2, 6, 2] # Swin-Tiny/Base 默认 Block 数量
        
        global_total = sum(global_depths)
        global_dpr_rates = [x.item() for x in torch.linspace(0, drop_path, global_total)]
        
        # ⭐ 传入 global_dpr_rates
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
        # ⭐ 修正 3: PyramidAdapter scale 初始化 (设置为 0.01)
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
def starnet_dual_swin_pyramid(pretrained=False, **kwargs):
    """
    StarNet Dual-Pyramid with Swin Transformer (Base configuration)
    Local Pyramid: StarBlock (CNN)
    Global Pyramid: Swin Transformer (Window-based Attention)
    """
    return StarNet_DualSwinPyramid(
        base=24, 
        depths=[2, 2, 8, 3], 
        global_depths=[2, 2, 6, 2],
        **kwargs
    )