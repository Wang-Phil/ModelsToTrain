#!/usr/bin/env python3
"""
下载 CoOp 模型所需的预训练权重
"""

import os
import requests
from tqdm import tqdm
from pathlib import Path
import socket

# 检查点目录
CHECKPOINT_DIR = Path("clip/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# 检测网络连接并选择镜像站点
def check_network_and_get_mirror():
    """检测网络连接，如果无法连接 huggingface.co，则使用镜像站点"""
    try:
        # 尝试连接 huggingface.co
        socket.create_connection(("huggingface.co", 443), timeout=3)
        print("✓ 可以连接到 huggingface.co，使用官方站点")
        return "https://huggingface.co"
    except (socket.timeout, socket.error, OSError):
        print("⚠ 无法连接到 huggingface.co，使用镜像站点: hf-mirror.com")
        return "https://hf-mirror.com"

# 获取基础 URL
BASE_URL = check_network_and_get_mirror()

# PMC-CLIP 文件
PMC_FILES = {
    "text_encoder.pth": f"{BASE_URL}/datasets/axiong/pmc_oa/resolve/main/text_encoder.pth",
    "image_encoder(resnet50).pth": f"{BASE_URL}/datasets/axiong/pmc_oa/resolve/main/image_encoder(resnet50).pth",
    "text_projection_layer.pth": f"{BASE_URL}/datasets/axiong/pmc_oa/resolve/main/text_projection_layer.pth",
}

# PubMedCLIP 文件
PUBMED_FILES = {
    "PubMedCLIP_ViT32.pth": f"{BASE_URL}/sarahESL/PubMedCLIP/resolve/main/PubMedCLIP_ViT32.pth?download=true",
}


def download_file(url, filepath, max_retries=3):
    """下载文件并显示进度条，支持重试和镜像站点"""
    if filepath.exists():
        print(f"✓ {filepath.name} 已存在，跳过下载")
        return True
    
    print(f"下载 {filepath.name}...")
    print(f"  URL: {url}")
    
    # 如果第一次失败，尝试使用镜像站点
    urls_to_try = [url]
    if "huggingface.co" in url:
        mirror_url = url.replace("https://huggingface.co", "https://hf-mirror.com")
        urls_to_try.append(mirror_url)
    
    for attempt, try_url in enumerate(urls_to_try, 1):
        try:
            if attempt > 1:
                print(f"  尝试镜像站点: {try_url}")
            
            response = requests.get(try_url, stream=True, timeout=60)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=filepath.name) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            print(f"✓ {filepath.name} 下载完成")
            return True
        except Exception as e:
            if attempt < len(urls_to_try):
                print(f"  ✗ 尝试 {attempt} 失败: {e}")
                if filepath.exists():
                    filepath.unlink()  # 删除不完整的文件
                continue
            else:
                print(f"✗ {filepath.name} 下载失败（所有尝试都失败）: {e}")
                if filepath.exists():
                    filepath.unlink()  # 删除不完整的文件
                return False
    
    return False


def main():
    print("="*80)
    print("下载 CoOp 模型预训练权重")
    print("="*80)
    print()
    
    print("【1. CoOp_CLIP (标准 CLIP)】")
    print("-" * 80)
    print("✓ 自动下载，无需手动操作")
    print("  首次运行训练时会自动下载到 CLIP 库的默认缓存目录")
    print()
    
    print("【2. CoOp_PMCCLIP】")
    print("-" * 80)
    print(f"下载目录: {CHECKPOINT_DIR.absolute()}")
    print()
    
    success_count = 0
    for filename, url in PMC_FILES.items():
        filepath = CHECKPOINT_DIR / filename
        if download_file(url, filepath):
            success_count += 1
        print()
    
    print("【3. CoOp_PubMedCLIP】")
    print("-" * 80)
    print(f"下载目录: {CHECKPOINT_DIR.absolute()}")
    print()
    
    for filename, url in PUBMED_FILES.items():
        filepath = CHECKPOINT_DIR / filename
        if download_file(url, filepath):
            success_count += 1
        print()
    
    print("="*80)
    print("下载完成！")
    print("="*80)
    print(f"成功下载: {success_count}/{len(PMC_FILES) + len(PUBMED_FILES)} 个文件")
    print(f"文件位置: {CHECKPOINT_DIR.absolute()}")
    print("="*80)


if __name__ == "__main__":
    main()

