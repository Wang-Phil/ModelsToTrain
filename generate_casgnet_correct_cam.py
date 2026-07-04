#!/usr/bin/env python3
"""
CasGNet（GAP + Linear 分类头）经典 CAM（非 Grad-CAM）：
在最后一层空间特征图（CASGNet 为 encoder.norm 输出）上用分类层权重做通道加权求和，
ReLU、归一化后对 JET 着色，再叠加到与推理一致的 resize 原图上。

默认在训练划分验证集 old_data/val 上扫描；每个类别在「真实标签=预测标签」前提下，
按文件名排序取最多 --per-class 张（默认 10），分别导出 PNG。

用法（项目根）:
  python generate_casgnet_correct_cam.py \\
    --checkpoint checkpoints/casgnet_supcon_olddata_resplit_new/best_auc_model.pth \\
    --data-dir old_data/val \\
    --per-class 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from compare_models_on_eltra_test import _infer_proj_dims
from train_casgnet_contrastive_newdata import SupConClassifierNet
from train_multiclass import ImageFolderDataset, get_data_augmentation


def _compute_gap_cam_map(
    encoder: nn.Module,
    x: torch.Tensor,
    class_idx: int,
    out_hw: tuple[int, int],
) -> np.ndarray:
    """
    经典 CAM：CAM_c = ReLU( sum_k w_{c,k} * F_{k,:,:} )，F 为 GAP 前特征图；
    归一化到 [0,1] 并双线性缩放到 out_hw。
    返回 float32 (H,W)，取值约 [0,1]。
    """
    head = encoder.head
    if not isinstance(head, nn.Linear):
        raise TypeError(f"CAM 需要 Linear 分类头，当前为 {type(head).__name__}")

    hook_mod = getattr(encoder, "norm", None)
    if hook_mod is None:
        raise ValueError("encoder 缺少 norm（CASGNet 预期在 GAP 前有 BN 输出），无法用经典 CAM")

    feats: list[torch.Tensor] = []

    def _hook(_m, _inp, out):
        feats.append(out.detach())

    h = hook_mod.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            _ = encoder.forward_features(x)
    finally:
        h.remove()

    if not feats:
        raise RuntimeError("未捕获到空间特征图")
    fmap = feats[0]
    if fmap.dim() != 4:
        raise RuntimeError(f"期望 4D 特征图，得到 shape={tuple(fmap.shape)}")

    with torch.no_grad():
        w = head.weight[class_idx].detach().view(1, -1, 1, 1)
        cam = (w * fmap).sum(dim=1).squeeze(0)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam_np = cam.float().cpu().numpy()
    cam_rs = cv2.resize(cam_np, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_LINEAR)
    return cam_rs.astype(np.float32)


def _overlay_jet_save(
    rgb_uint8: np.ndarray,
    cam_01: np.ndarray,
    *,
    alpha: float,
    save_path: Path,
) -> None:
    """rgb_uint8: HWC RGB；cam_01: HW 与 rgb 同尺寸；保存 BGR PNG。"""
    img_f = rgb_uint8.astype(np.float32) / 255.0
    heat_bgr = cv2.applyColorMap(np.uint8(255 * np.clip(cam_01, 0, 1)), cv2.COLORMAP_JET)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blend = (1.0 - alpha) * img_f + alpha * heat_rgb
    out_u8 = np.clip(blend * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(save_path), cv2.cvtColor(out_u8, cv2.COLOR_RGB2BGR))


def _pick_k_correct_per_class(
    data_root: Path,
    net: SupConClassifierNet,
    device: torch.device,
    val_transform,
    ck_class_to_idx: dict[str, int],
    num_classes: int,
    k: int,
) -> dict[int, list[Path]]:
    """每类最多收集 k 张验证正确的样本路径（路径排序，先到先得）。"""
    base = ImageFolderDataset(str(data_root), transform=None)
    if ck_class_to_idx and base.class_to_idx != ck_class_to_idx:
        raise ValueError(
            "类别与 checkpoint 不一致。\n"
            f"  ckpt: {ck_class_to_idx}\n  data: {base.class_to_idx}"
        )
    buckets: dict[int, list[Path]] = {c: [] for c in range(num_classes)}
    samples_sorted = sorted(base.samples, key=lambda t: t[0])
    for path_str, y_idx in samples_sorted:
        if len(buckets[y_idx]) >= k:
            continue
        img = Image.open(path_str).convert("RGB")
        x = val_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = net.forward_logits(x)
        pred = int(torch.argmax(logits, dim=1).item())
        if pred == y_idx:
            buckets[y_idx].append(Path(path_str))
    return buckets


def main() -> None:
    ap = argparse.ArgumentParser(description="CasGNet 验证集每类若干预测正确样本 — 经典 CAM 叠加原图")
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/casgnet_supcon_olddata_resplit_new/best_auc_model.pth"),
    )
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=Path("old_data/val"),
        help="ImageFolder 根目录（默认验证集 old_data/val）",
    )
    ap.add_argument(
        "--per-class",
        type=int,
        default=10,
        help="每个类别最多导出几张「验证正确」样本的 CAM（按路径排序先到先得）",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="默认: <checkpoint 父目录>/cam_val_correct",
    )
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument(
        "--overlay-alpha",
        type=float,
        default=0.45,
        help="JET 热力图与原图混合权重（越大热力越显眼）",
    )
    ap.add_argument("--augmentation", type=str, default="standard")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    ck_path = args.checkpoint.resolve()
    data_root = args.data_dir.resolve()
    if not ck_path.is_file():
        print(f"找不到权重: {ck_path}", file=sys.stderr)
        sys.exit(1)
    if not data_root.is_dir():
        print(f"数据目录不存在: {data_root}", file=sys.stderr)
        sys.exit(1)

    k = max(1, int(args.per_class))

    out_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else ck_path.parent / "cam_val_correct"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        ckpt = torch.load(ck_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(ck_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    num_classes = int(ckpt.get("num_classes"))
    ck_class_to_idx: dict[str, int] = ckpt.get("class_to_idx") or {}
    model_name = str(ckpt.get("model", "casgnet_s1"))
    proj_dim, hidden_dim = _infer_proj_dims(state_dict)

    _, val_aug = get_data_augmentation(augmentation_type=args.augmentation, img_size=args.img_size)

    net = SupConClassifierNet(
        num_classes=num_classes,
        model_name=model_name,
        proj_dim=proj_dim,
        hidden_dim=hidden_dim,
        pretrained=False,
    )
    net.load_state_dict(state_dict, strict=True)
    net = net.to(device)
    net.eval()

    enc = net.encoder

    buckets = _pick_k_correct_per_class(
        data_root, net, device, val_aug, ck_class_to_idx, num_classes, k
    )

    idx_to_name = {int(i): n for n, i in sorted(ck_class_to_idx.items(), key=lambda x: x[1])}
    missing_all = [c for c in range(num_classes) if len(buckets[c]) == 0]
    short_of_k = [c for c in range(num_classes) if 0 < len(buckets[c]) < k]
    if missing_all:
        print(
            "错误: 下列类别在数据集中没有「验证正确」样本:",
            [idx_to_name.get(c, str(c)) for c in missing_all],
            file=sys.stderr,
        )
        sys.exit(2)
    if short_of_k:
        print(
            "警告: 下列类别验证正确样本不足 "
            f"{k} 张，仅导出已有张数:",
            [(idx_to_name.get(c, str(c)), len(buckets[c])) for c in short_of_k],
            file=sys.stderr,
        )

    out_hw = (args.img_size, args.img_size)
    for c in range(num_classes):
        name_safe = idx_to_name[c].replace(" ", "_").replace("/", "_")
        for i, img_path in enumerate(buckets[c]):
            save_path = out_dir / f"{name_safe}_class{c}_{i:02d}_cam.png"

            img = Image.open(img_path).convert("RGB")
            x = val_aug(img).unsqueeze(0).to(device)

            cam_map = _compute_gap_cam_map(enc, x, int(c), out_hw=out_hw)
            orig_rs = cv2.resize(np.array(img), (out_hw[1], out_hw[0]), interpolation=cv2.INTER_LINEAR)
            _overlay_jet_save(orig_rs, cam_map, alpha=float(args.overlay_alpha), save_path=save_path)
            print(f"已写入: {save_path}")


if __name__ == "__main__":
    main()
