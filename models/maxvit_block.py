# models/maxvit_block.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class MAXViTBlock(nn.Module):
    def __init__(self, dim, window_size=7, num_heads=8, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.window_size = window_size
        
        # 1. 3x3 Depthwise Conv
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm1 = nn.LayerNorm(dim)
        
        # 2. Grid Attention (长条注意力)
        self.grid_attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        
        # 3. Window Attention (方块注意力)
        self.window_attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm3 = nn.LayerNorm(dim)
        
        # 4. FFN
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x
        
        # 1. 3x3 DWConv + Norm
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # B H W C
        x = self.norm1(x)
        
        # 2. Grid Attention (先划成长条)
        x_grid = x.view(B, H//self.window_size, self.window_size, W, C)
        x_grid = x_grid.permute(0, 1, 3, 2, 4).reshape(B * (H//self.window_size), W, -1)  # [B*gH, W, ws*C]
        x_grid = self.grid_attn(x_grid, x_grid, x_grid)[0]
        x_grid = x_grid.view(B, H//self.window_size, W, self.window_size, C)
        x_grid = x_grid.permute(0, 1, 3, 2, 4).reshape(B, H, W, C)
        x = x + x_grid
        
        # 3. Window Attention
        x_win = window_partition(x, self.window_size)  # nW, ws, ws, C
        x_win = x_win.view(-1, self.window_size * self.window_size, C)
        attn_win = self.window_attn(x_win, x_win, x_win)[0]
        x = x_win + attn_win
        x = window_reverse(x, self.window_size, H, W)
        
        x = x + self.mlp(self.norm3(x))
        x = x.permute(0, 3, 1, 2)  # B C H W
        x = shortcut + x
        return x