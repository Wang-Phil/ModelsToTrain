#!/usr/bin/env python3
"""
CLIP 风格训练，五折交叉验证。
默认可按配置组合：CLIP 对称对比损失、可选 SupCon、可选分类（CE/Focal）。
与 train_biomedcoop.py 共用 CLIPDataset / CLIPSubset / get_data_augmentation 接口。
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from tqdm import tqdm

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
if "HF_ENDPOINT" not in os.environ:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from calculate_metrics import calculate_classification_metrics
from models.clip import CLIPModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_data_augmentation(augmentation: str, img_size: int):
    """与 train_biomedcoop 一致的增强接口。"""
    from torchvision import transforms

    if augmentation == "none":
        train_t = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    elif augmentation == "minimal":
        train_t = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        train_t = transforms.Compose(
            [
                transforms.Resize((img_size + 32, img_size + 32)),
                transforms.RandomCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    val_t = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_t, val_t


class CLIPDataset(Dataset):
    """
    按类别文件夹组织的数据集；samples 为 (path, label, class_name) 三元组，
    与 train_biomedcoop.create_folds_from_dataset 一致。
    """

    def __init__(
        self,
        root_dir,
        transform=None,
        text_template=None,
        class_texts_dict=None,
        class_texts_file=None,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.text_template = text_template or "a photo of a {}"
        self.samples = []
        self.class_texts_map = {}

        if class_texts_file:
            p = Path(class_texts_file)
            if not p.is_file():
                alt = self.root_dir.parent / class_texts_file
                if alt.is_file():
                    p = alt
            if p.is_file():
                with open(p, "r", encoding="utf-8") as f:
                    self.class_texts_map = json.load(f)
        elif class_texts_dict:
            self.class_texts_map = dict(class_texts_dict)

        excluded = {"split_fewshot", "__pycache__", ".ipynb_checkpoints"}
        classes = sorted(
            d.name
            for d in self.root_dir.iterdir()
            if d.is_dir() and d.name not in excluded and not d.name.startswith("split_")
        )
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}

        for cls_name in classes:
            cdir = self.root_dir / cls_name
            label = self.class_to_idx[cls_name]
            for img_file in cdir.glob("*"):
                if img_file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
                    self.samples.append((str(img_file), label, cls_name))

        print(f"CLIPDataset: {len(self.samples)} 张图, {len(classes)} 类")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label, cls_name = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, cls_name

    def class_prompts_ordered(self):
        """按 label 顺序的类别文本，用于预计算文本特征。"""
        out = []
        for i in range(len(self.class_to_idx)):
            name = self.idx_to_class[i]
            if name in self.class_texts_map:
                out.append(self.class_texts_map[name])
            else:
                out.append(self.text_template.format(name))
        return out


class CLIPSubset(Dataset):
    def __init__(self, base: CLIPDataset, indices, transform=None):
        self.base = base
        self.indices = list(indices)
        self.transform = transform
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, i):
        path, label, cls_name = self.base.samples[self.indices[i]]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, cls_name


def create_folds_from_dataset(dataset: CLIPDataset, n_splits=5, shuffle=True, random_state=42):
    labels = [t[1] for t in dataset.samples]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    folds = []
    for tr, va in skf.split(range(len(dataset)), labels):
        folds.append((tr.tolist(), va.tolist()))
    return folds


def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    ce = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce)
    loss = alpha * (1 - pt) ** gamma * ce
    return loss.mean()


def supcon_loss(features, labels, temperature=0.07):
    """有监督对比损失（特征已 L2 归一化亦可）。"""
    z = F.normalize(features, dim=1)
    device = z.device
    b = z.size(0)
    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)
    logits_mask = torch.ones_like(mask) - torch.eye(b, device=device)
    mask = mask * logits_mask
    sim = torch.matmul(z, z.T) / temperature
    logits_max, _ = torch.max(sim, dim=1, keepdim=True)
    logits = sim - logits_max.detach()
    exp = torch.exp(logits) * logits_mask
    denom = exp.sum(1, keepdim=True).clamp(min=1e-12)
    log_prob = logits - torch.log(denom)
    n_pos = mask.sum(1).clamp(min=1.0)
    mean_log_pos = (mask * log_prob).sum(1) / n_pos
    valid = mask.sum(1) > 0
    if valid.any():
        return -mean_log_pos[valid].mean()
    return torch.tensor(0.0, device=device, requires_grad=True)


def clip_symmetric_loss(img_f, txt_f, temperature):
    """对称 CLIP 式对比损失（对角线为正样本）。"""
    logits = (img_f @ txt_f.T) / temperature.clamp(min=1e-4)
    targets = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets))


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    prompts,
    class_text_features_fix,
    scaler,
    use_amp,
    freeze_text,
    use_supcon,
    supcon_temp,
    supcon_w,
    clip_w,
    class_w,
    use_focal,
    focal_alpha,
    focal_gamma,
):
    model.train()
    if freeze_text:
        model.text_encoder.eval()

    tot = 0.0
    n = 0
    for images, labels, _ in tqdm(loader, desc="Train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            img_f = model.image_encoder(images)
            if freeze_text:
                text_all = class_text_features_fix
            else:
                text_all = model.text_encoder(texts=prompts)

            loss_cls = torch.tensor(0.0, device=device)
            if class_w > 0:
                logits = model.compute_similarity(img_f, text_all)
                if use_focal:
                    loss_cls = focal_loss(logits, labels, alpha=focal_alpha, gamma=focal_gamma)
                else:
                    loss_cls = F.cross_entropy(logits, labels)

            loss_s = torch.tensor(0.0, device=device)
            if use_supcon and supcon_w > 0:
                loss_s = supcon_loss(img_f, labels, temperature=supcon_temp)

            loss_c = torch.tensor(0.0, device=device)
            if clip_w > 0:
                txt_b = text_all[labels]
                loss_c = clip_symmetric_loss(img_f, txt_b, model.temperature)

            loss = class_w * loss_cls + supcon_w * loss_s + clip_w * loss_c

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        tot += loss.item() * images.size(0)
        n += images.size(0)

    return tot / max(n, 1)


@torch.no_grad()
def validate(model, loader, device, class_text_features, use_amp):
    model.eval()
    preds, gts = [], []
    vloss = 0.0
    vn = 0
    for images, labels, _ in tqdm(loader, desc="Val", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            img_f = model.image_encoder(images)
            logits = model.compute_similarity(img_f, class_text_features)
            loss = F.cross_entropy(logits, labels)
        vloss += loss.item() * images.size(0)
        vn += images.size(0)
        pred = logits.argmax(dim=1)
        preds.append(pred.cpu())
        gts.append(labels.cpu())
    preds = torch.cat(preds)
    gts = torch.cat(gts)
    num_classes = class_text_features.size(0)
    metrics = calculate_classification_metrics(gts, preds, num_classes)
    return vloss / max(vn, 1), metrics


@torch.no_grad()
def encode_class_texts(model, prompts, device, batch_size=32):
    feats = []
    model.eval()
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        f = model.text_encoder(texts=chunk)
        feats.append(f)
    return torch.cat(feats, dim=0)


def train_cross_validation(args, cfg: dict):
    set_seed(cfg.get("random_state", 42))
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gpu_id = int(cfg.get("gpu_id", args.gpu_id))
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    img_enc = cfg.get("image_encoder_name", cfg.get("image_encoder", "resnet18"))
    txt_enc = cfg.get("text_encoder_name", cfg.get("text_encoder", "clip:ViT-B/32"))
    embed_dim = int(cfg.get("embed_dim", 512))
    temperature = float(cfg.get("temperature", 0.07))

    batch_size = int(cfg.get("batch_size", 32))
    epochs = int(cfg.get("epochs", 100))
    lr = float(cfg.get("learning_rate", 1e-4))
    wd = float(cfg.get("weight_decay", 0.01))
    img_size = int(cfg.get("img_size", 224))
    aug = cfg.get("augmentation", "standard")
    num_workers = int(cfg.get("num_workers", 4))
    use_amp = bool(cfg.get("use_amp", True)) and not args.no_amp

    n_splits = int(cfg.get("n_splits", 5))
    random_state = int(cfg.get("random_state", 42))

    class_texts_file = cfg.get("class_texts_file")
    use_weighted_sampling = bool(cfg.get("use_weighted_sampling", False))
    weight_method = cfg.get("weight_method", "inverse_freq")
    weight_smooth = float(cfg.get("weight_smooth_factor", 1.0))

    freeze_img = bool(cfg.get("freeze_image_encoder", False))
    freeze_txt = bool(cfg.get("freeze_text_encoder", False))

    use_supcon = bool(cfg.get("use_supcon_loss", False))
    supcon_temp = float(cfg.get("supcon_temperature", 0.07))
    supcon_w = float(cfg.get("supcon_loss_weight", 0.0))
    clip_w = float(cfg.get("clip_loss_weight", 1.0))
    class_w = float(cfg.get("class_loss_weight", 1.0))

    use_focal = bool(cfg.get("use_focal_loss", False))
    focal_alpha = float(cfg.get("focal_alpha", 0.25))
    focal_gamma = float(cfg.get("focal_gamma", 2.0))

    print(
        f"训练损失权重: CLIP={clip_w}, 分类={class_w}, SupCon={supcon_w} (use_supcon={use_supcon}); "
        f"加权采样={use_weighted_sampling}"
    )

    es_patience = cfg.get("early_stopping_patience")
    es_delta = float(cfg.get("early_stopping_min_delta", 0.0))
    es_monitor = cfg.get("early_stopping_monitor", "val_loss")

    train_t, val_t = get_data_augmentation(aug, img_size)
    full = CLIPDataset(
        data_dir,
        transform=None,
        text_template=None,
        class_texts_dict=None,
        class_texts_file=class_texts_file,
    )
    prompts = full.class_prompts_ordered()

    folds = create_folds_from_dataset(full, n_splits=n_splits, shuffle=True, random_state=random_state)

    all_fold = {
        "fold_best_val_acc": [],
        "fold_best_val_mAP": [],
        "fold_best_val_precision": [],
        "fold_best_val_recall": [],
        "fold_best_val_f1": [],
        "fold_best_epoch": [],
        "fold_val_loss": [],
    }

    pretrained_path = cfg.get("pretrained_model_path")

    for fold_idx, (tr_idx, va_idx) in enumerate(folds, start=1):
        print(f"\n{'='*60}\nFold {fold_idx}/{n_splits}\n{'='*60}")
        fold_dir = output_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        train_ds = CLIPSubset(full, tr_idx, transform=train_t)
        val_ds = CLIPSubset(full, va_idx, transform=val_t)
        
        sampler = None
        if use_weighted_sampling:
            sub_labels = [full.samples[i][1] for i in tr_idx]
            cc = Counter(sub_labels)
            total = len(sub_labels)
            n_cls = len(cc)
            cw = {}
            for cidx, cnt in cc.items():
                if weight_method in ("inverse_freq", "balanced"):
                    cw[cidx] = total / (n_cls * (cnt + weight_smooth))
                elif weight_method == "inverse_sqrt":
                    cw[cidx] = np.sqrt(total / (cnt + weight_smooth))
                else:
                    raise ValueError(weight_method)
            w = [cw[y] for y in sub_labels]
            sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            num_workers=num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )

        model = CLIPModel(
            image_encoder_name=img_enc,
            text_encoder_name=txt_enc,
            embed_dim=embed_dim,
            temperature=temperature,
        ).to(device)

        if pretrained_path:
            try:
                ck = torch.load(pretrained_path, map_location=device, weights_only=True)
            except TypeError:
                ck = torch.load(pretrained_path, map_location=device)
            sd = ck.get("model_state_dict", ck.get("state_dict", ck))
            model.load_state_dict(sd, strict=False)
            print(f"已加载部分/全部权重: {pretrained_path}")

        if freeze_img:
            for p in model.image_encoder.parameters():
                p.requires_grad = False
        if freeze_txt:
            for p in model.text_encoder.parameters():
                p.requires_grad = False

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
        scaler = GradScaler(enabled=use_amp)

        with torch.no_grad():
            class_text_fix = encode_class_texts(model, prompts, device).to(device)

        best_acc = -1.0
        best_mAP = -1.0
        best_loss = float("inf")
        best_epoch = 0
        best_metrics = None
        bad = 0
        use_es = es_patience is not None

        for epoch in range(epochs):
            tr_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                prompts,
                class_text_fix,
                scaler,
                use_amp,
                freeze_txt,
                use_supcon,
                supcon_temp,
                supcon_w,
                clip_w,
                class_w,
                use_focal,
                focal_alpha,
                focal_gamma,
            )
            with torch.no_grad():
                val_tf = encode_class_texts(model, prompts, device).to(device)
            val_loss, metrics = validate(model, val_loader, device, val_tf, use_amp)
            acc = metrics["accuracy"]
            mAP = metrics["mAP"]
            print(
                f"Epoch {epoch+1}/{epochs}  train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  "
                f"acc={acc:.2f}%  mAP={mAP:.2f}%"
            )

            improved = False
            if es_monitor == "val_acc":
                if acc > best_acc + es_delta:
                    improved = True
            elif es_monitor == "val_mAP":
                if mAP > best_mAP + es_delta:
                    improved = True
            else:
                if val_loss < best_loss - es_delta:
                    improved = True

            if improved:
                best_acc = acc
                best_mAP = mAP
                best_loss = val_loss
                best_epoch = epoch + 1
                best_metrics = dict(metrics)
                bad = 0
                torch.save(
                    {"model_state_dict": model.state_dict(), "epoch": best_epoch, "metrics": best_metrics},
                    fold_dir / "best_model.pth",
                )
            elif use_es:
                bad += 1
                if bad >= es_patience:
                    print(f"早停: {es_patience} epoch 无改善 (monitor={es_monitor})")
                    break
            
        all_fold["fold_best_val_acc"].append(best_acc)
        all_fold["fold_best_val_mAP"].append(best_mAP)
        all_fold["fold_best_val_precision"].append(
            best_metrics["precision_macro"] if best_metrics else 0.0
        )
        all_fold["fold_best_val_recall"].append(best_metrics["recall_macro"] if best_metrics else 0.0)
        all_fold["fold_best_val_f1"].append(best_metrics["f1_macro"] if best_metrics else 0.0)
        all_fold["fold_best_epoch"].append(best_epoch)
        all_fold["fold_val_loss"].append(best_loss)

    def _mean_std(key):
        arr = np.array(all_fold[key], dtype=np.float64)
        return float(arr.mean()), float(arr.std())

    m_acc, s_acc = _mean_std("fold_best_val_acc")
    m_map, s_map = _mean_std("fold_best_val_mAP")
    m_prec, s_prec = _mean_std("fold_best_val_precision")
    m_rec, s_rec = _mean_std("fold_best_val_recall")
    m_f1, s_f1 = _mean_std("fold_best_val_f1")

    average_results = {
        "avg_best_val_acc": m_acc,
        "std_best_val_acc": s_acc,
        "avg_best_val_mAP": m_map,
        "std_best_val_mAP": s_map,
        "avg_best_precision": m_prec,
        "std_best_precision": s_prec,
        "avg_best_recall": m_rec,
        "std_best_recall": s_rec,
        "avg_best_f1": m_f1,
        "std_best_f1": s_f1,
    }

    cv_summary = {
        "mode": "cv",
        "n_splits": n_splits,
        "random_state": random_state,
        "image_encoder": img_enc,
        "text_encoder": txt_enc,
        "average_best_val_acc": m_acc,
        "std_best_val_acc": s_acc,
        "average_best_val_mAP": m_map,
        "std_best_val_mAP": s_map,
        "average_best_val_precision": m_prec,
        "std_best_val_precision": s_prec,
        "average_best_val_recall": m_rec,
        "std_best_val_recall": s_rec,
        "average_best_val_f1": m_f1,
        "std_best_val_f1": s_f1,
        "average_results": average_results,
        "fold_results": all_fold,
    }

    with open(output_dir / "cv_summary.json", "w", encoding="utf-8") as f:
        json.dump(cv_summary, f, indent=2, ensure_ascii=False)

    cfg_save = dict(cfg)
    cfg_save["image_encoder_name"] = img_enc
    cfg_save["text_encoder_name"] = txt_enc
    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg_save, f, indent=2, ensure_ascii=False)

    print(f"\n交叉验证完成，结果: {output_dir / 'cv_summary.json'}")


def build_arg_parser():
    p = argparse.ArgumentParser(description="CLIP 训练 (SupCon + CLIP + 分类/Focal)")
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--config-file", type=str, default=None)
    p.add_argument("--multi-config", action="store_true", help="兼容 run_ablation_study.py，无额外作用")

    p.add_argument("--image-encoder", type=str, default=None)
    p.add_argument("--text-encoder", type=str, default=None)
    p.add_argument("--embed-dim", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--img-size", type=int, default=None)
    p.add_argument("--augmentation", type=str, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--use-cv", action="store_true")
    p.add_argument("--n-splits", type=int, default=None)
    p.add_argument("--random-state", type=int, default=None)
    p.add_argument("--class-texts-file", type=str, default=None)
    p.add_argument("--use-weighted-sampling", action="store_true")
    p.add_argument("--weight-method", type=str, default=None)
    p.add_argument("--weight-smooth-factor", type=float, default=None)
    p.add_argument("--freeze-image-encoder", action="store_true")
    p.add_argument("--freeze-text-encoder", action="store_true")
    p.add_argument("--use-supcon-loss", action="store_true")
    p.add_argument("--supcon-temperature", type=float, default=None)
    p.add_argument("--supcon-loss-weight", type=float, default=None)
    p.add_argument("--clip-loss-weight", type=float, default=None)
    p.add_argument("--class-loss-weight", type=float, default=None)
    p.add_argument("--use-focal-loss", action="store_true")
    p.add_argument("--focal-alpha", type=float, default=None)
    p.add_argument("--focal-gamma", type=float, default=None)
    p.add_argument("--early-stopping-patience", type=int, default=None)
    p.add_argument("--early-stopping-min-delta", type=float, default=None)
    p.add_argument("--early-stopping-monitor", type=str, default=None)
    return p


def merge_config(args, file_cfg: dict | None) -> dict:
    cfg = dict(file_cfg) if file_cfg else {}

    def ov(key, argval, cfg_key=None):
        ck = cfg_key or key
        if argval is not None:
            cfg[ck] = argval
        return cfg.get(ck)

    ov("image_encoder_name", args.image_encoder)
    ov("text_encoder_name", args.text_encoder)
    ov("embed_dim", args.embed_dim)
    ov("batch_size", args.batch_size)
    ov("epochs", args.epochs)
    ov("learning_rate", args.learning_rate)
    ov("weight_decay", args.weight_decay)
    ov("temperature", args.temperature)
    ov("img_size", args.img_size)
    ov("augmentation", args.augmentation)
    ov("num_workers", args.num_workers)
    ov("n_splits", args.n_splits)
    ov("random_state", args.random_state)
    ov("class_texts_file", args.class_texts_file)
    ov("weight_method", args.weight_method)
    ov("weight_smooth_factor", args.weight_smooth_factor)
    ov("supcon_temperature", args.supcon_temperature)
    ov("supcon_loss_weight", args.supcon_loss_weight)
    ov("clip_loss_weight", args.clip_loss_weight)
    ov("class_loss_weight", args.class_loss_weight)
    ov("focal_alpha", args.focal_alpha)
    ov("focal_gamma", args.focal_gamma)
    ov("early_stopping_patience", args.early_stopping_patience)
    ov("early_stopping_min_delta", args.early_stopping_min_delta)
    ov("early_stopping_monitor", args.early_stopping_monitor)

    if args.gpu_id is not None:
        cfg["gpu_id"] = args.gpu_id
    if args.use_weighted_sampling:
        cfg["use_weighted_sampling"] = True
    if args.freeze_image_encoder:
        cfg["freeze_image_encoder"] = True
    if args.freeze_text_encoder:
        cfg["freeze_text_encoder"] = True
    if args.use_supcon_loss:
        cfg["use_supcon_loss"] = True
    if args.use_focal_loss:
        cfg["use_focal_loss"] = True

    return cfg


def main():
    args = build_arg_parser().parse_args()
    file_cfg = None
    if args.config_file:
        with open(args.config_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        file_cfg = data[0] if isinstance(data, list) else data

    cfg = merge_config(args, file_cfg)

    cfg["use_cv"] = True
    train_cross_validation(args, cfg)


if __name__ == "__main__":
    main()
