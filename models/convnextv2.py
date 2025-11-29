# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from .utils import LayerNorm, GRN
from torch.cpu.amp import autocast

import os
from pathlib import Path

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
LOCAL_WEIGHTS_DIR = PROJECT_ROOT / "pretrain_weights"
DATA_WEIGHTS_DIR = Path("/data/wangweicheng/ModelsToTrains/pretrainModels")

# 优先使用本地权重，如果不存在则使用/data目录
def get_weights_path(filename):
    """获取权重文件路径，优先使用本地路径"""
    local_path = LOCAL_WEIGHTS_DIR / filename
    data_path = DATA_WEIGHTS_DIR / filename
    
    if local_path.exists():
        return str(local_path)
    elif data_path.exists():
        return str(data_path)
    else:
        # 如果都不存在，返回本地路径（用于下载）
        return str(local_path)

model_urls = {
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_base_1k_224_ema.pt",
    "convnext_large_1k": get_weights_path("convnextv2_large_1k_224_ema.pt"),
    "convnext_tiny_1k": get_weights_path("convnextv2_tiny_1k_224_ema.pt"),
    "convnext_nano_1k": get_weights_path("convnextv2_nano_1k_224_ema.pt"),
    "convnext_base_2k": get_weights_path("convnextv2_base_22k_384_ema.pt"),
    "convnext_tiny_2k": get_weights_path("convnextv2_tiny_22k_224_ema.pt"),
}


class Block(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXtV2(nn.Module):
    """ ConvNeXt V2
        
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., head_init_scale=1.
                 ):
        super().__init__()
        self.depths = depths
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)
    @autocast()
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

def convnextv2_atto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model

def convnextv2_femto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model

def convnext_pico(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model

def convnextv2_nano(pretrained = False,**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    if pretrained:
        url = model_urls['convnext_nano_1k']
        # checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        checkpoint = torch.load(url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

def convnextv2_tiny(pretrained = False,**kwargs):
    # 先创建模型（使用指定的num_classes）
    model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    
    if pretrained:
        url = model_urls['convnext_tiny_2k']
        # 检查是否是URL还是本地路径
        if url.startswith('http'):
            # 从URL下载
            checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        else:
            # 从本地路径加载
            if not os.path.exists(url):
                raise FileNotFoundError(f"预训练权重文件不存在: {url}\n请先运行: python download_pretrained_weights.py convnextv2_tiny_22k_224_ema")
            checkpoint = torch.load(url, map_location="cpu")
        
        # 获取预训练权重
        checkpoint_model = checkpoint["model"] if "model" in checkpoint else checkpoint
        
        # 移除分类头权重（因为预训练模型是1000类，而我们的模型可能是其他类别数）
        keys_to_remove = []
        for key in checkpoint_model.keys():
            if 'head' in key or 'classifier' in key or 'fc' in key:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            if key in checkpoint_model:
                print(f"移除不匹配的分类头权重: {key} (形状: {checkpoint_model[key].shape})")
                del checkpoint_model[key]
        
        # 加载权重，忽略不匹配的键
        load_result = model.load_state_dict(checkpoint_model, strict=False)
        
        if load_result.missing_keys:
            print(f"警告: 以下权重未加载（缺失）: {load_result.missing_keys}")
        if load_result.unexpected_keys:
            print(f"警告: 以下权重未使用（多余）: {load_result.unexpected_keys}")
        
        print(f"✓ 预训练权重加载完成（已排除分类头）")

    return model

def convnextv2_base(pretrained=False,**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        url = model_urls['convnext_base_2k']
        # checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        checkpoint = torch.load(url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])

    return model

def convnextv2_large(pretrained=False,**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        url = model_urls['convnext_large_1k']
        checkpoint = torch.load(url, map_location="cpu")
        # checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

def convnextv2_huge(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
    return model