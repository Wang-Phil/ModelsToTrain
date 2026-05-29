#!/usr/bin/env python3
"""
使用 new_data 训练分类骨干（CASGNet / ResNet / DenseNet 等）：默认仅交叉熵（单视图）；可选 --no-ce-only 启用 SupCon + CE；
按验证集 macro AUC (OVR) 保存最佳模型；训练结束后在验证集概率上做 Bootstrap 得 AUC 的 95% 置信区间。
不执行 K 折交叉验证。
"""

from __future__ import annotations

import argparse
import inspect
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as tvm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from models.classic_models import create_model
from train_multiclass import ImageFolderDataset, get_data_augmentation

try:
    from torch.amp import GradScaler
except ImportError:  # pragma: no cover
    from torch.cuda.amp import GradScaler  # type: ignore


def _grad_scaler_for_cuda() -> "GradScaler":
    """
    torch.cuda.amp.GradScaler 旧版: 首参是 init_scale(float), 传 \"cuda\" 会变成非法标度.
    torch.amp.GradScaler(2.0+): 首参多为 device. 用签名判断, 避免与版本号不一致的安装.
    """
    p = [k for k in inspect.signature(GradScaler.__init__).parameters if k != "self"]
    if p and p[0] == "init_scale":
        return GradScaler()
    if "device" in inspect.signature(GradScaler.__init__).parameters:
        return GradScaler("cuda")
    return GradScaler()


# --------------------------------------------------------------------------- #
# 双视图 (SupCon)
# --------------------------------------------------------------------------- #


class TwoCrops:
    def __init__(self, t: transforms.Compose):
        self.t = t

    def __call__(self, image):
        return self.t(image), self.t(image)


class TransformSubset(Dataset):
    """对 ImageFolder 样本子集使用指定 transform（同 train_cross_validation, 减少导入依赖）。"""

    def __init__(self, base_dataset: ImageFolderDataset, indices, transform):
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        from PIL import Image

        p, y = self.base_dataset.samples[self.indices[idx]]
        image = Image.open(p).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, y


class TwoCropsImageFolder(Dataset):
    def __init__(self, image_folder: ImageFolderDataset, indices: list[int] | np.ndarray, two_crops: TwoCrops):
        self.dataset = image_folder
        self.indices = list(indices)
        self.two_crops = two_crops

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        from PIL import Image

        real_idx = self.indices[idx]
        p, y = self.dataset.samples[real_idx]
        im = Image.open(p).convert("RGB")
        q, k = self.two_crops(im)
        return q, k, y


def collate_two_crops(batch):
    qs, ks, labels = zip(*batch)
    return torch.stack(qs, 0), torch.stack(ks, 0), torch.tensor(labels, dtype=torch.long)


# --------------------------------------------------------------------------- #
# SupCon: 特征 [2B, D] 与标签 y.repeat(2) 一一对应 (前 B 为 view1, 后 B 为 view2)
# --------------------------------------------------------------------------- #


def supcon_loss_flat(
    f: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07, base_temperature: float = 0.07, eps: float = 1e-12
) -> torch.Tensor:
    """
    监督对比: 同类为正样本(不含自身); 对每行 anchor, log_softmax 分母为除自身外全体样本
    见 Khosla et al., Supervised Contrastive Learning
    """
    f = F.normalize(f, dim=1)
    m = f.size(0)
    sim = (f @ f.T) / temperature
    l = labels.view(-1, 1)
    pos_mask = (l == l.T).float()
    self_mask = torch.eye(m, device=f.device, dtype=pos_mask.dtype)
    pos_mask = pos_mask * (1.0 - self_mask)  # 正样本(同类非自身)
    not_self = 1.0 - self_mask
    logits = sim - torch.max(sim, dim=1, keepdim=True)[0].detach()
    log_denom = torch.log((torch.exp(logits) * not_self).sum(1, keepdim=True) + eps)
    log_prob = logits - log_denom
    mean_log = (pos_mask * log_prob).sum(1) / (pos_mask.sum(1) + eps)
    if (pos_mask.sum(1) <= 0).all():
        return f.sum() * 0.0
    return -mean_log[pos_mask.sum(1) > 0].mean() * (base_temperature / temperature)


# --------------------------------------------------------------------------- #
# 骨干需实现: forward_features -> [B,D], head(fe) -> logits, 属性 in_channel = D
# --------------------------------------------------------------------------- #


def _last_linear_in_features(module: nn.Module) -> int:
    if isinstance(module, nn.Linear):
        return int(module.in_features)
    if isinstance(module, nn.Sequential):
        for layer in reversed(list(module.children())):
            if isinstance(layer, nn.Linear):
                return int(layer.in_features)
    raise TypeError(f"无法从分类头推断特征维度: {type(module).__name__}")


class _ResNetLikeEncoder(nn.Module):
    """torchvision ResNet 等: 除最后一层外为 backbone，最后一层为线性分类头。"""

    def __init__(self, full: nn.Module):
        super().__init__()
        kids = list(full.children())
        if len(kids) < 2:
            raise ValueError("ResNet-like 模型子模块过少")
        self.backbone = nn.Sequential(*kids[:-1])
        self.head = kids[-1]
        if not isinstance(self.head, nn.Linear):
            raise TypeError(f"期望最后一层为 nn.Linear, 实为 {type(self.head).__name__}")
        self.in_channel = int(self.head.in_features)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        return torch.flatten(z, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


class _GoogLeNetEncoder(nn.Module):
    """GoogLeNet 主干不可简单 Sequential（Inception 分支顺序固定）；沿 torchvision._forward 提取池化后向量。"""

    def __init__(self, full: nn.Module):
        super().__init__()
        self.g = full
        self.head = full.fc
        if not isinstance(self.head, nn.Linear):
            raise TypeError(f"期望 fc 为 Linear, 实为 {type(self.head).__name__}")
        self.in_channel = int(self.head.in_features)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        g = self.g
        x = g._transform_input(x)
        x = g.conv1(x)
        x = g.maxpool1(x)
        x = g.conv2(x)
        x = g.conv3(x)
        x = g.maxpool2(x)
        x = g.inception3a(x)
        x = g.inception3b(x)
        x = g.maxpool3(x)
        x = g.inception4a(x)
        if g.aux1 is not None and self.training:
            _ = g.aux1(x)
        x = g.inception4b(x)
        x = g.inception4c(x)
        x = g.inception4d(x)
        if g.aux2 is not None and self.training:
            _ = g.aux2(x)
        x = g.inception4e(x)
        x = g.maxpool4(x)
        x = g.inception5a(x)
        x = g.inception5b(x)
        x = g.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.forward_features(x)
        z = self.g.dropout(z)
        return self.head(z)


class _DenseNetEncoder(nn.Module):
    def __init__(self, full: tvm.DenseNet):
        super().__init__()
        self.features = full.features
        self.head = full.classifier
        self.in_channel = int(full.classifier.in_features)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        z = F.relu(z, inplace=True)
        z = F.adaptive_avg_pool2d(z, (1, 1))
        return torch.flatten(z, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


class _MobileNetEncoder(nn.Module):
    def __init__(self, full: nn.Module):
        super().__init__()
        self.features = full.features
        self.head = full.classifier
        self.in_channel = _last_linear_in_features(full.classifier)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        z = F.adaptive_avg_pool2d(z, (1, 1))
        return torch.flatten(z, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


class _EfficientNetB0Encoder(nn.Module):
    def __init__(self, full: tvm.EfficientNet):
        super().__init__()
        self.features = full.features
        self.avgpool = full.avgpool
        self.head = full.classifier
        self.in_channel = _last_linear_in_features(full.classifier)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        z = self.avgpool(z)
        return torch.flatten(z, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


class _StarNetEncoder(nn.Module):
    """适配原始 StarNet: stem + stages + norm + avgpool + head。"""

    def __init__(self, full: nn.Module):
        super().__init__()
        self.stem = full.stem
        self.stages = full.stages
        self.norm = full.norm
        self.avgpool = full.avgpool
        self.head = full.head
        self.in_channel = int(full.in_channel)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = torch.flatten(self.avgpool(self.norm(x)), 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


def _adapt_encoder_for_supcon(full: nn.Module) -> nn.Module:
    """将 create_model 得到的分类网络包装为 forward_features + head + in_channel。"""
    if hasattr(full, "forward_features") and hasattr(full, "head") and callable(getattr(full, "forward_features")):
        if not hasattr(full, "in_channel") or full.in_channel is None:
            h = full.head
            if isinstance(h, nn.Linear):
                full.in_channel = int(h.in_features)
            else:
                raise TypeError("骨干有 forward_features/head 但缺少 in_channel 且 head 非 Linear")
        return full
    if isinstance(full, tvm.ResNet):
        return _ResNetLikeEncoder(full)
    if isinstance(full, tvm.GoogLeNet):
        return _GoogLeNetEncoder(full)
    if isinstance(full, tvm.DenseNet):
        return _DenseNetEncoder(full)
    if isinstance(full, tvm.MobileNetV2) or isinstance(full, tvm.MobileNetV3):
        return _MobileNetEncoder(full)
    if isinstance(full, tvm.EfficientNet):
        return _EfficientNetB0Encoder(full)
    if all(hasattr(full, k) for k in ("stem", "stages", "norm", "avgpool", "head", "in_channel")):
        return _StarNetEncoder(full)
    raise ValueError(
        "当前骨干未适配 SupCon（需 CASG 风格、StarNet 或 torchvision ResNet/DenseNet/MobileNet/EfficientNet）。"
        "可改用 --model casgnet_s1、casgnet_s2、starnet_s1、resnet18、resnet50 等。"
    )


def build_supcon_encoder(model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
    mn = model_name.lower()
    # starnet_s1 为论文 StarNet-S1（StarNet 24×[2,2,8,3]）；超小网络请用 starnet_s050/s100/s150
    mn_alias = {
        "starnet_official_s1": "starnet_s1",
        "starnet_s2": "starnet_s100",
        "starnet_s3": "starnet_s150",
    }
    mn = mn_alias.get(mn, mn)
    full = create_model(mn, num_classes=num_classes, pretrained=pretrained)
    return _adapt_encoder_for_supcon(full)


SUPPORTED_SUPCON_MODELS: tuple[str, ...] = (
    "casgnet_s1",
    "casgnet_s2",
    # CASGNet-S1：SA / GRN / 末 stage SK 的 2^3 全因子消融（见 models/casgnet.py）
    "casgnet_s1_ab000",
    "casgnet_s1_ab100",
    "casgnet_s1_ab010",
    "casgnet_s1_ab001",
    "casgnet_s1_ab110",
    "casgnet_s1_ab101",
    "casgnet_s1_ab011",
    "casgnet_s1_ab111",
    "starnet_s1",
    "starnet_official_s1",
    "starnet_s2",
    "starnet_s050",
    "starnet_s100",
    "starnet_s150",
    # StarNet-SK 核组合（与 models/starnetsk.py、classic_models 一致）
    "starnet_s1_sk13",
    "starnet_s1_sk15",
    "starnet_s1_sk17",
    "starnet_s1_sk19",
    "starnet_s1_sk35",
    "starnet_s1_sk37",
    "starnet_s1_sk39",
    "starnet_s1_sk57",
    "starnet_s1_sk59",
    "starnet_s1_sk79",
    "resnet18",
    "resnet50",
    "resnet101",
    "googlenet",
    "densenet121",
    "mobilenetv2",
    "mobilenetv3_small",
    "mobilenetv3_large",
    "mobilenetv4_m",
    "lsnet_b",
    "efficientnet_b0",
)


class SupConClassifierNet(nn.Module):
    """任意适配后的 encoder + 投影头；CE-only 时仅用 logits；开启 SupCon 时用投影特征。"""

    def __init__(
        self,
        num_classes: int,
        model_name: str,
        proj_dim: int = 128,
        hidden_dim: int = 512,
        pretrained: bool = True,
    ):
        super().__init__()
        self.model_name = model_name.lower()
        self.encoder = build_supcon_encoder(self.model_name, num_classes=num_classes, pretrained=pretrained)
        d = int(self.encoder.in_channel)
        self.proj = nn.Sequential(nn.Linear(d, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, proj_dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fe = self.encoder.forward_features(x)
        logits = self.encoder.head(fe)
        p = F.normalize(self.proj(fe), dim=1)
        return logits, p

    @torch.inference_mode()
    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.head(self.encoder.forward_features(x))


# --------------------------------------------------------------------------- #
# 指标
# --------------------------------------------------------------------------- #


def collect_val_probs(
    model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probs_list: list = []
    y_list: list = []
    y_pred: list = []
    use_ac = use_amp and device.type == "cuda"
    m = model.module if hasattr(model, "module") else model
    for images, y in tqdm(loader, desc="[Val]"):
        images = images.to(device, non_blocking=True)
        y = y.to(device)
        if use_ac:
            with torch.amp.autocast("cuda"):
                logits = m.forward_logits(images)
        else:
            logits = m.forward_logits(images)
        p = F.softmax(logits.float(), dim=1).cpu().numpy()
        pr = torch.argmax(logits, dim=1)
        probs_list.append(p)
        y_list.append(y.cpu().numpy())
        y_pred.append(pr.cpu().numpy())
    return np.vstack(probs_list), np.concatenate(y_list), np.concatenate(y_pred)


def compute_macro_auc_ovr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    稳健的 macro-OVR AUC:
    - 对每个类别做 one-vs-rest 二分类 AUC
    - 若某类别在 y_true 中仅单一取值(全正/全负), 跳过该类别
    - 对可计算类别求均值; 若都不可算则返回 0.0
    """
    n_classes = y_score.shape[1]
    aucs: list[float] = []
    for c in range(n_classes):
        y_bin = (y_true == c).astype(np.int32)
        if np.unique(y_bin).size < 2:
            continue
        try:
            a = float(roc_auc_score(y_bin, y_score[:, c]))
        except (ValueError, TypeError):
            continue
        if np.isfinite(a):
            aucs.append(a)
    if not aucs:
        return 0.0
    return float(np.mean(aucs))


def _safe_divide(num: float, den: float) -> float:
    if den <= 0:
        return float("nan")
    return float(num / den)


def compute_macro_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> tuple[dict, list[dict]]:
    """
    多分类 one-vs-rest 指标:
    - sensitivity (recall), specificity, ppv (precision), npv, acc
    返回:
    - macro 指标字典
    - 各类别指标列表
    """
    per_class: list[dict] = []
    sens_list: list[float] = []
    spec_list: list[float] = []
    ppv_list: list[float] = []
    npv_list: list[float] = []
    acc_list: list[float] = []

    for c in range(n_classes):
        yt = (y_true == c).astype(np.int32)
        yp = (y_pred == c).astype(np.int32)
        tp = int(np.sum((yt == 1) & (yp == 1)))
        tn = int(np.sum((yt == 0) & (yp == 0)))
        fp = int(np.sum((yt == 0) & (yp == 1)))
        fn = int(np.sum((yt == 1) & (yp == 0)))
        total = tp + tn + fp + fn

        sensitivity = _safe_divide(tp, tp + fn)
        specificity = _safe_divide(tn, tn + fp)
        ppv = _safe_divide(tp, tp + fp)
        npv = _safe_divide(tn, tn + fn)
        acc = _safe_divide(tp + tn, total)

        sens_list.append(sensitivity)
        spec_list.append(specificity)
        ppv_list.append(ppv)
        npv_list.append(npv)
        acc_list.append(acc)

        per_class.append(
            {
                "class_idx": int(c),
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "ppv": ppv,
                "npv": npv,
                "acc": acc,
            }
        )

    def _macro(v: list[float]) -> float:
        arr = np.asarray(v, dtype=np.float64)
        if np.isfinite(arr).any():
            return float(np.nanmean(arr))
        return 0.0

    macro = {
        "sensitivity": _macro(sens_list),
        "specificity": _macro(spec_list),
        "ppv": _macro(ppv_list),
        "npv": _macro(npv_list),
        "acc": _macro(acc_list),
    }
    return macro, per_class


def bootstrap_auc_ci(
    y_true: np.ndarray, y_score: np.ndarray, n_boot: int = 1000, random_state: int = 42, confidence: float = 0.95
) -> tuple[float, float, float]:
    n = len(y_true)
    n_classes = y_score.shape[1]
    rng = np.random.RandomState(random_state)
    aucs: list[float] = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        ys = y_score[idx]
        if len(yt) < 3 or len(np.unique(yt)) < 2:
            continue
        a = compute_macro_auc_ovr(yt, ys)
        if np.isfinite(a):
            aucs.append(a)
    if not aucs:
        m = compute_macro_auc_ovr(y_true, y_score)
        return m, m, m
    aucs_arr = np.array(aucs)
    p_lo = (1.0 - confidence) / 2.0
    p_hi = 1.0 - p_lo
    return float(aucs_arr.mean()), float(np.percentile(aucs_arr, p_lo * 100)), float(np.percentile(aucs_arr, p_hi * 100))


def bootstrap_classification_metrics_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_classes: int,
    n_boot: int = 1000,
    random_state: int = 42,
    confidence: float = 0.95,
) -> dict:
    n = len(y_true)
    rng = np.random.RandomState(random_state)
    keys = ("sensitivity", "specificity", "npv", "ppv", "acc")
    values: dict[str, list[float]] = {k: [] for k in keys}

    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        macro, _ = compute_macro_classification_metrics(yt, yp, n_classes=n_classes)
        for k in keys:
            v = float(macro.get(k, float("nan")))
            if np.isfinite(v):
                values[k].append(v)

    p_lo = (1.0 - confidence) / 2.0
    p_hi = 1.0 - p_lo
    out: dict[str, dict[str, float]] = {}
    point_macro, _ = compute_macro_classification_metrics(y_true, y_pred, n_classes=n_classes)
    for k in keys:
        arr = np.asarray(values[k], dtype=np.float64)
        if arr.size == 0:
            m = float(point_macro.get(k, 0.0))
            out[k] = {"mean": m, "ci95_low": m, "ci95_high": m}
        else:
            out[k] = {
                "mean": float(np.mean(arr)),
                "ci95_low": float(np.percentile(arr, p_lo * 100)),
                "ci95_high": float(np.percentile(arr, p_hi * 100)),
            }
    return out


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    w_sup: float,
    w_ce: float,
    sup_temperature: float,
    scaler: GradScaler | None,
    ce_only: bool,
) -> float:
    model.train()
    running = 0.0
    n = 0
    use_sc = scaler is not None
    pbar = tqdm(loader, desc="[Train]")
    for batch in pbar:
        optimizer.zero_grad(set_to_none=True)
        m = model.module if hasattr(model, "module") else model
        if ce_only:
            x, y = batch
            b = y.size(0)
            y = y.to(device, non_blocking=True)
            x = x.to(device, non_blocking=True)
            if use_sc and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    logits, _ = m(x)
                    loss = F.cross_entropy(logits, y)
            else:
                logits, _ = m(x)
                loss = F.cross_entropy(logits, y)
        else:
            q, k, y = batch
            b = y.size(0)
            y = y.to(device, non_blocking=True)
            x = torch.cat([q, k], dim=0).to(device, non_blocking=True)
            y_double = y.repeat(2)
            if use_sc and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    logits, proj = m(x)
                    f1, f2 = proj[:b], proj[b:]
                    fcat = torch.cat([f1, f2], dim=0)
                    l_sup = supcon_loss_flat(fcat, y_double, temperature=sup_temperature)
                    l_ce = F.cross_entropy(logits, y_double)
                    loss = w_sup * l_sup + w_ce * l_ce
            else:
                logits, proj = m(x)
                f1, f2 = proj[:b], proj[b:]
                fcat = torch.cat([f1, f2], dim=0)
                l_sup = supcon_loss_flat(fcat, y_double, temperature=sup_temperature)
                l_ce = F.cross_entropy(logits, y_double)
                loss = w_sup * l_sup + w_ce * l_ce
        if use_sc and device.type == "cuda":
            scaler.scale(loss).backward()  # type: ignore[union-attr]
            scaler.step(optimizer)  # type: ignore[union-attr]
            scaler.update()  # type: ignore[union-attr]
        else:
            loss.backward()
            optimizer.step()
        running += loss.item() * b
        n += b
        pbar.set_postfix({"loss": f"{running / max(n, 1):.4f}"})
    return running / max(n, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="new_data", help="类子目录 = 各类别 (ImageFolder)")
    parser.add_argument("--train-dir", type=str, default=None, help="固定划分: 训练集目录 (train/<class>/*)")
    parser.add_argument("--val-dir", type=str, default=None, help="固定划分: 验证集目录 (val/<class>/*)")
    parser.add_argument(
        "--test-dir",
        type=str,
        default=None,
        help="可选: 测试集目录 (test/<class>/*)，仅固定划分；训练结束后评估（不参与选 checkpoint）",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集比例 (分层随机划分)")
    parser.add_argument(
        "--output-dir", type=str, default="checkpoints/casgnet_supcon_newdata", help="最佳权重与 JSON 结果"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="casgnet_s1",
        choices=list(SUPPORTED_SUPCON_MODELS),
        help="分类骨干（classic_models）；默认 CE-only，可用 --no-ce-only 开 SupCon",
    )
    parser.add_argument(
        "--model-variant",
        type=str,
        default=None,
        choices=["s1", "s2"],
        help="兼容旧版: 若指定则固定 CASGNet 并覆盖 --model 为 casgnet_s1/s2",
    )
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--ce-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="默认 True：单视图 + 仅交叉熵；--no-ce-only 为双视图 SupCon + CE",
    )
    parser.add_argument("--w-supcon", type=float, default=1.0, help="监督对比损失权重（仅 --no-ce-only 时生效）")
    parser.add_argument("--w-ce", type=float, default=1.0, help="交叉熵权重（仅 --no-ce-only 时与 SupCon 组合）")
    parser.add_argument("--supcon-temp", type=float, default=0.07, help="对比温度（仅 --no-ce-only）")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--augmentation", type=str, default="standard", help="见 train_multiclass: standard/strong/...")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Bootstrap 次数 (95%% CI)")
    parser.add_argument("--use-amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", type=str, default=None, help="如 cuda:0, 留空则自动")
    args = parser.parse_args()

    if args.model_variant is not None:
        if args.model not in ("casgnet_s1", "casgnet_s2"):
            print(
                "警告: 已指定 --model-variant，将使用 CASGNet 并忽略原 --model",
                file=sys.stderr,
            )
        args.model = f"casgnet_{args.model_variant}"

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_aug, val_aug = get_data_augmentation(augmentation_type=args.augmentation, img_size=args.img_size)

    # 支持两种模式:
    # 1) 随机划分模式: 仅使用 --data-dir 与 --val-ratio
    # 2) 固定划分模式: 同时传入 --train-dir 与 --val-dir
    if (args.train_dir is None) ^ (args.val_dir is None):
        print("错误: --train-dir 与 --val-dir 必须同时提供", file=sys.stderr)
        sys.exit(1)

    if args.test_dir is not None and (args.train_dir is None or args.val_dir is None):
        print("错误: --test-dir 仅支持与 --train-dir、--val-dir 同时使用（固定划分）", file=sys.stderr)
        sys.exit(1)

    if args.train_dir is not None and args.val_dir is not None:
        train_root = Path(args.train_dir)
        val_root = Path(args.val_dir)
        if not train_root.is_dir():
            print(f"训练目录不存在: {train_root}", file=sys.stderr)
            sys.exit(1)
        if not val_root.is_dir():
            print(f"验证目录不存在: {val_root}", file=sys.stderr)
            sys.exit(1)

        train_base = ImageFolderDataset(str(train_root), transform=None)  # type: ignore
        val_base = ImageFolderDataset(str(val_root), transform=None)  # type: ignore

        train_classes = set(train_base.class_to_idx.keys())
        val_classes = set(val_base.class_to_idx.keys())
        if train_classes != val_classes:
            miss_in_val = sorted(train_classes - val_classes)
            miss_in_train = sorted(val_classes - train_classes)
            print(
                f"类别不一致: 仅 train 有 {miss_in_val}; 仅 val 有 {miss_in_train}",
                file=sys.stderr,
            )
            sys.exit(1)

        # 统一 train/val 的标签索引映射, 避免目录顺序差异导致标签错位
        class_names = sorted(train_classes)
        class_to_idx = {name: i for i, name in enumerate(class_names)}
        idx_to_class = {i: name for name, i in class_to_idx.items()}

        def _remap_labels(ds: ImageFolderDataset) -> None:
            remapped = []
            for p, y in ds.samples:
                cls_name = ds.idx_to_class[y]
                remapped.append((p, class_to_idx[cls_name]))
            ds.samples = remapped
            ds.class_to_idx = class_to_idx
            ds.idx_to_class = idx_to_class

        _remap_labels(train_base)
        _remap_labels(val_base)

        test_loader = None
        n_test = 0
        if args.test_dir:
            test_root = Path(args.test_dir)
            if not test_root.is_dir():
                print(f"测试目录不存在: {test_root}", file=sys.stderr)
                sys.exit(1)
            test_base = ImageFolderDataset(str(test_root), transform=None)  # type: ignore
            test_classes = set(test_base.class_to_idx.keys())
            if test_classes != train_classes:
                miss_in_test = sorted(train_classes - test_classes)
                miss_in_train = sorted(test_classes - train_classes)
                print(
                    f"测试集类别与 train 不一致: 仅 train 有 {miss_in_test}; 仅 test 有 {miss_in_train}",
                    file=sys.stderr,
                )
                sys.exit(1)
            _remap_labels(test_base)
            n_test = len(test_base)
            print(f"测试集: {n_test} ({test_root})（仅最终评估）")

        tr_idx = np.arange(len(train_base))
        va_idx = np.arange(len(val_base))
        num_classes = len(class_to_idx)
        split_mode = "fixed"
        data_root = None
        print(
            f"固定划分 | 训练: {len(tr_idx)} ({train_root}) | 验证: {len(va_idx)} ({val_root}) | "
            f"类别: {num_classes} | 骨干: {args.model} | 设备: {device}"
        )
        if args.ce_only:
            train_ds = TransformSubset(train_base, tr_idx, transform=train_aug)
        else:
            two_crops = TwoCrops(train_aug)
            train_ds = TwoCropsImageFolder(train_base, tr_idx, two_crops)
        val_ds = TransformSubset(val_base, va_idx, transform=val_aug)
    else:
        data_root = Path(args.data_dir)
        if not data_root.is_dir():
            print(f"数据目录不存在: {data_root}", file=sys.stderr)
            sys.exit(1)
        base = ImageFolderDataset(str(data_root), transform=None)  # type: ignore
        n_samples = len(base)
        lab_list = [lab for _, lab in base.samples]
        tr_idx, va_idx = train_test_split(
            np.arange(n_samples), test_size=args.val_ratio, stratify=lab_list, random_state=args.seed, shuffle=True
        )
        num_classes = len(base.class_to_idx)
        split_mode = "random"
        class_to_idx = base.class_to_idx
        print(
            f"随机划分 | 总样本: {n_samples} | 训练: {len(tr_idx)} | 验证: {len(va_idx)} | 类别: {num_classes} | "
            f"骨干: {args.model} | 设备: {device}"
        )
        if args.ce_only:
            train_ds = TransformSubset(base, tr_idx, transform=train_aug)
        else:
            two_crops = TwoCrops(train_aug)
            train_ds = TwoCropsImageFolder(base, tr_idx, two_crops)
        val_ds = TransformSubset(base, va_idx, transform=val_aug)
        test_loader = None
        n_test = 0

    pin = device.type == "cuda"
    pkw: dict = {}
    if args.num_workers > 0:
        pkw["persistent_workers"] = True
    tr_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
        collate_fn=None if args.ce_only else collate_two_crops,
        **pkw,
    )
    va_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin, **pkw
    )

    if split_mode == "fixed" and args.test_dir:
        test_ds_final = TransformSubset(
            test_base, np.arange(len(test_base)), transform=val_aug  # type: ignore[name-defined]
        )
        test_loader = DataLoader(
            test_ds_final,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin,
            **pkw,
        )

    model = SupConClassifierNet(
        num_classes,
        model_name=args.model,
        proj_dim=args.proj_dim,
        hidden_dim=args.hidden_dim,
        pretrained=args.pretrained,
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler: GradScaler | None = None
    if args.use_amp and device.type == "cuda":
        scaler = _grad_scaler_for_cuda()

    best_auc = -1.0
    best_path = out / "best_auc_model.pth"
    history: dict = {"epoch": [], "train_loss": [], "val_auc": []}
    t0 = time.time()
    checkpoint: dict | None = None

    for ep in range(1, args.epochs + 1):
        tr_loss = train_epoch(
            model,
            tr_loader,
            optimizer,
            device,
            args.w_supcon,
            args.w_ce,
            args.supcon_temp,
            scaler,
            ce_only=bool(args.ce_only),
        )
        probs, yt, _ = collect_val_probs(model, va_loader, device, bool(args.use_amp and device.type == "cuda"))
        v_auc = compute_macro_auc_ovr(yt, probs)
        cur_best = max(v_auc, best_auc) if best_auc >= 0 else v_auc
        history["epoch"].append(ep)
        history["train_loss"].append(tr_loss)
        history["val_auc"].append(v_auc)
        print(f"Epoch {ep}/{args.epochs}  train_loss={tr_loss:.4f}  val_auc (macro OVR)={v_auc:.4f}  历史最好={cur_best:.4f}")

        if v_auc > best_auc:
            best_auc = v_auc
            save_obj = {
                "epoch": ep,
                "val_auc": v_auc,
                "state_dict": model.state_dict(),
                "num_classes": num_classes,
                "class_to_idx": class_to_idx,
                "model": args.model,
                "model_variant": args.model.replace("casgnet_", "") if args.model.startswith("casgnet_") else None,
            }
            torch.save(save_obj, best_path)
            print(f"  -> 已保存最佳权重 (AUC 提升) {best_path}")
        else:
            print("  -> 未覆盖 (AUC 未超过历史最高)")

    if best_auc < 0:
        print("警告: 无有效 AUC 记录", file=sys.stderr)

    if best_path.is_file():
        try:
            checkpoint = torch.load(best_path, map_location=device, weights_only=True)  # type: ignore[call-arg]
        except TypeError:  # older torch
            checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)

    use_amp_val = bool(args.use_amp and device.type == "cuda")
    probs, yt, yhat = collect_val_probs(model, va_loader, device, use_amp_val)
    point_auc = compute_macro_auc_ovr(yt, probs)
    macro_cls_metrics, per_class_metrics = compute_macro_classification_metrics(yt, yhat, n_classes=num_classes)
    mean_b, lo, hi = bootstrap_auc_ci(yt, probs, n_boot=args.n_bootstrap, random_state=args.seed)
    cls_boot = bootstrap_classification_metrics_ci(
        yt, yhat, n_classes=num_classes, n_boot=args.n_bootstrap, random_state=args.seed
    )
    best_saved = float(checkpoint["val_auc"]) if checkpoint and "val_auc" in checkpoint else point_auc

    test_eval: dict | None = None
    if args.test_dir and split_mode == "fixed" and test_loader is not None:
        probs_te, yt_te, yhat_te = collect_val_probs(model, test_loader, device, use_amp_val)
        point_auc_te = compute_macro_auc_ovr(yt_te, probs_te)
        macro_te, per_class_te = compute_macro_classification_metrics(yt_te, yhat_te, n_classes=num_classes)
        mean_b_te, lo_te, hi_te = bootstrap_auc_ci(
            yt_te, probs_te, n_boot=args.n_bootstrap, random_state=args.seed
        )
        cls_boot_te = bootstrap_classification_metrics_ci(
            yt_te, yhat_te, n_classes=num_classes, n_boot=args.n_bootstrap, random_state=args.seed
        )
        test_eval = {
            "auc": point_auc_te,
            "sensitivity": macro_te["sensitivity"],
            "specificity": macro_te["specificity"],
            "npv": macro_te["npv"],
            "ppv": macro_te["ppv"],
            "acc": macro_te["acc"],
            "bootstrap_auc": {
                "mean": mean_b_te,
                "ci95_low": lo_te,
                "ci95_high": hi_te,
                "n_bootstrap": args.n_bootstrap,
            },
            "bootstrap_metrics": {
                "sensitivity": cls_boot_te["sensitivity"],
                "specificity": cls_boot_te["specificity"],
                "npv": cls_boot_te["npv"],
                "ppv": cls_boot_te["ppv"],
                "acc": cls_boot_te["acc"],
                "n_bootstrap": args.n_bootstrap,
            },
            "n_test": int(n_test),
            "reloaded_test_auc": point_auc_te,
        }
        print(
            json.dumps(
                {"test_macro_auc_ovr": point_auc_te, "test_bootstrap_auc_mean": mean_b_te},
                indent=2,
                ensure_ascii=False,
            )
        )

    summary = {
        "split_mode": split_mode,
        "data_dir": str(data_root) if data_root is not None else None,
        "train_dir": args.train_dir,
        "val_dir": args.val_dir,
        "test_dir": args.test_dir if split_mode == "fixed" else None,
        "val_ratio": args.val_ratio if split_mode == "random" else None,
        "model": args.model,
        "n_train": int(len(tr_idx)),
        "n_val": int(len(va_idx)),
        "num_classes": num_classes,
        "auc": point_auc,
        "sensitivity": macro_cls_metrics["sensitivity"],
        "specificity": macro_cls_metrics["specificity"],
        "npv": macro_cls_metrics["npv"],
        "ppv": macro_cls_metrics["ppv"],
        "acc": macro_cls_metrics["acc"],
        "best_val_auc_on_save": best_saved,
        "reloaded_val_auc": point_auc,
        "bootstrap_auc": {
            "mean": mean_b,
            "ci95_low": lo,
            "ci95_high": hi,
            "n_bootstrap": args.n_bootstrap,
        },
        "bootstrap_metrics": {
            "sensitivity": cls_boot["sensitivity"],
            "specificity": cls_boot["specificity"],
            "npv": cls_boot["npv"],
            "ppv": cls_boot["ppv"],
            "acc": cls_boot["acc"],
            "n_bootstrap": args.n_bootstrap,
        },
        "loss_mode": "CE only (single-view)" if args.ce_only else "SupCon (2-view) + CE",
        "contrastive": (
            "cross-entropy only; save on val AUC"
            if args.ce_only
            else "SupCon (2-view, same-class positives) + CE; save only on val AUC"
        ),
        "seconds": time.time() - t0,
    }
    if test_eval is not None:
        summary["test_eval"] = test_eval
        summary["n_test"] = test_eval["n_test"]

    with open(out / "result_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(out / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    with open(out / "per_class_metrics_val.json", "w", encoding="utf-8") as f:
        json.dump(per_class_metrics, f, indent=2, ensure_ascii=False)
    if test_eval is not None:
        with open(out / "per_class_metrics_test.json", "w", encoding="utf-8") as f:
            json.dump(per_class_te, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
