# -*- coding: utf-8 -*-
"""
Step 1: 离线特征固化（只做一次）
从髋关节验证集中抽样，用 CLIP Image Encoder 提取特征并保存为 I。
"""

import json
import torch
from pathlib import Path
from tqdm import tqdm

from config import (
    CLASS_NAMES,
    VALID_IMAGE_DIR,
    FEATURES_PATH,
    CLIP_MODEL_NAME,
    CACHE_DIR,
)

# 尝试 openai/clip，若无则用 open_clip
try:
    import clip
    USE_OPEN_CLIP = False
except ImportError:
    import open_clip
    USE_OPEN_CLIP = True

# 常见图片后缀
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_clip(device="cuda"):
    if USE_OPEN_CLIP:
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        return model, preprocess, tokenizer, device
    else:
        device = device if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load(CLIP_MODEL_NAME, device=device)
        return model, preprocess, None, device


def collect_image_paths(root: Path):
    """收集 root 下按类别子目录存放的图片路径。返回 [(path, class_idx), ...]"""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"验证集目录不存在: {root}")
    pairs = []
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        folder = root / cls_name
        if not folder.is_dir():
            continue
        for p in folder.iterdir():
            if p.suffix.lower() in IMAGE_EXT:
                pairs.append((str(p), cls_idx))
    return pairs


def extract_and_save(
    image_root=None,
    output_path=None,
    device="cuda",
    max_per_class=100,
):
    """
    提取验证集图像特征并保存。
    image_root: 验证集根目录，其下为类别名子目录。
    output_path: 保存的 .pt 文件路径。
    max_per_class: 每类最多使用图片数。
    """
    image_root = Path(image_root or VALID_IMAGE_DIR)
    output_path = output_path or FEATURES_PATH
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pairs = collect_image_paths(image_root)
    if not pairs:
        raise ValueError(
            f"在 {image_root} 下未找到按类别子目录存放的图片。"
            f"请确保存在如: {image_root}/Good Place/*.jpg"
        )

    # 每类截断，保持 (path, label) 对齐
    from collections import defaultdict
    by_class = defaultdict(list)
    for path, c in pairs:
        by_class[c].append(path)
    paths, labels = [], []
    for c in range(len(CLASS_NAMES)):
        for path in by_class[c][:max_per_class]:
            paths.append(path)
            labels.append(c)

    model, preprocess, _, dev = load_clip(device)
    model.eval()
    features_list = []
    labels_list = []

    for path, label in tqdm(
        list(zip(paths, labels)),
        desc="Extracting image features",
    ):
        from PIL import Image
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Skip {path}: {e}")
            continue
        x = preprocess(img).unsqueeze(0).to(dev)
        with torch.no_grad():
            if USE_OPEN_CLIP:
                feat = model.encode_image(x)
            else:
                feat = model.encode_image(x)
        features_list.append(feat.cpu())
        labels_list.append(label)

    I = torch.cat(features_list, dim=0)
    labels_tensor = torch.tensor(labels_list, dtype=torch.long)
    meta = {
        "class_names": CLASS_NAMES,
        "num_classes": len(CLASS_NAMES),
        "num_images": I.shape[0],
    }
    torch.save(
        {"I": I, "labels": labels_tensor, "meta": meta},
        output_path,
    )
    print(f"Saved features: {I.shape} -> {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", default=VALID_IMAGE_DIR, help="验证集图片根目录")
    parser.add_argument("--output", default=str(FEATURES_PATH), help="特征输出 .pt 路径")
    parser.add_argument("--max_per_class", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    extract_and_save(
        image_root=args.image_root,
        output_path=args.output,
        device=args.device,
        max_per_class=args.max_per_class,
    )
