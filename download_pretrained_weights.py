"""
下载ConvNeXtV2预训练权重
"""

import os
import torch
from pathlib import Path

# 预训练权重URL
PRETRAINED_URLS = {
    "convnextv2_tiny_22k_224_ema": "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_tiny_22k_224_ema.pt",
    "convnextv2_tiny_1k_224_ema": "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_tiny_1k_224_ema.pt",
}

# 保存目录（优先使用/data目录，如果不存在则使用项目目录）
DATA_DIR = Path("/data/wangweicheng/ModelsToTrains/pretrainModels")
LOCAL_DIR = Path(__file__).parent / "pretrain_weights"

# 检查哪个目录可用
if DATA_DIR.parent.exists() and os.access(DATA_DIR.parent, os.W_OK):
    SAVE_DIR = DATA_DIR
else:
    SAVE_DIR = LOCAL_DIR
    print(f"注意: /data目录不可用，使用本地目录: {SAVE_DIR}")

SAVE_DIR.mkdir(parents=True, exist_ok=True)

def download_weights(model_name="convnextv2_tiny_22k_224_ema"):
    """下载预训练权重"""
    if model_name not in PRETRAINED_URLS:
        print(f"错误: 未知的模型名称: {model_name}")
        print(f"可用的模型: {list(PRETRAINED_URLS.keys())}")
        return False
    
    url = PRETRAINED_URLS[model_name]
    save_path = SAVE_DIR / f"{model_name}.pt"
    
    # 检查文件是否已存在
    if save_path.exists():
        print(f"预训练权重已存在: {save_path}")
        print(f"文件大小: {save_path.stat().st_size / 1024 / 1024:.2f} MB")
        return True
    
    print(f"开始下载: {model_name}")
    print(f"URL: {url}")
    print(f"保存路径: {save_path}")
    print("这可能需要几分钟，请耐心等待...")
    
    try:
        # 下载权重
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url,
            map_location="cpu",
            check_hash=True
        )
        
        # 保存权重
        torch.save(checkpoint, save_path)
        
        print(f"✓ 下载完成!")
        print(f"保存路径: {save_path}")
        print(f"文件大小: {save_path.stat().st_size / 1024 / 1024:.2f} MB")
        return True
        
    except Exception as e:
        print(f"✗ 下载失败: {e}")
        return False

if __name__ == '__main__':
    import sys
    
    # 默认下载tiny_22k版本（更好的预训练权重）
    model_name = "convnextv2_tiny_22k_224_ema"
    
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    
    print("=" * 60)
    print("ConvNeXtV2 预训练权重下载工具")
    print("=" * 60)
    print()
    
    success = download_weights(model_name)
    
    if success:
        print("\n" + "=" * 60)
        print("下载完成！现在可以使用 --pretrained 参数训练模型了")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("下载失败，请检查网络连接或手动下载")
        print("=" * 60)

