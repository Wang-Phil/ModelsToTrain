#!/usr/bin/env python3
"""Evaluate each per-epoch checkpoint (CasGNet / StarNet) on the subset217 manifest
and write per_epoch_subset217_auc.csv with columns: model, epoch, subset217_auc.

Reuses SupConClassifierNet + compute_macro_auc_ovr from the training script so the
AUC definition matches Table1's val_auc (macro OVR).

Usage:
    cd /home/ln/wangweicheng/ModelsTotrain
    CUDA_VISIBLE_DEVICES=0 python evaluation_results/excel_aligned/training_curves/casgnet_vs_starnet/subset217/evaluate_per_epoch_subset217.py
"""
from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT))

from train_casgnet_contrastive_newdata import SupConClassifierNet, compute_macro_auc_ovr  # noqa: E402
from train_multiclass import get_data_augmentation  # noqa: E402

MANIFEST = (
    ROOT
    / "evaluation_results/excel_aligned/table1_per_model/manifests/casgnet_table1_manifest.json"
)
CKPT_ROOT = ROOT / "checkpoints/per_epoch_retrain"
MODELS = {
    "casgnet": {"model_name": "casgnet_s1", "ckpt_dir": CKPT_ROOT / "casgnet_s1_ce_only"},
    "starnet": {"model_name": "starnet_s1", "ckpt_dir": CKPT_ROOT / "starnet_s1_ce_only"},
}
EPOCHS = 200
BATCH_SIZE = 64
NUM_WORKERS = 8
IMG_SIZE = 224
OUT_CSV = HERE / "per_epoch_subset217_auc.csv"


class PathListDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples  # list of (path, label_idx)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        p, y = self.samples[i]
        img = Image.open(p).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, y


def main():
    import json

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    paths = manifest["paths_relative_to_cwd"]
    class_names = sorted({os.path.basename(os.path.dirname(p)) for p in paths})
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    num_classes = len(class_names)
    print(f"[manifest] n={len(paths)} num_classes={num_classes} classes={class_names}")

    samples = [(p, class_to_idx[os.path.basename(os.path.dirname(p))]) for p in paths]

    _, val_aug = get_data_augmentation(augmentation_type="standard", img_size=IMG_SIZE)
    ds = PathListDataset(samples, val_aug)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    rows = []
    for label, cfg in MODELS.items():
        ckpt_dir = cfg["ckpt_dir"]
        model_name = cfg["model_name"]
        if not ckpt_dir.is_dir():
            print(f"[warn] missing ckpt dir: {ckpt_dir} -- skipping {label}")
            continue

        # Build model once; reuse across epochs via load_state_dict.
        model = SupConClassifierNet(
            num_classes=num_classes,
            model_name=model_name,
            proj_dim=128,
            hidden_dim=512,
            pretrained=False,
        ).to(device)
        model.eval()

        t_model = time.time()
        for ep in range(1, EPOCHS + 1):
            ck = ckpt_dir / f"checkpoint_epoch_{ep}.pth"
            if not ck.is_file():
                continue
            try:
                state = torch.load(ck, map_location=device, weights_only=True)
            except TypeError:
                state = torch.load(ck, map_location=device)
            try:
                model.load_state_dict(state["state_dict"], strict=True)
            except Exception as e:
                print(f"[err] {label} ep{ep} load_state_dict failed: {e}")
                continue

            probs_list = []
            y_list = []
            with torch.inference_mode():
                for images, y in loader:
                    images = images.to(device, non_blocking=True)
                    logits = model.forward_logits(images)
                    p = F.softmax(logits.float(), dim=1).cpu().numpy()
                    probs_list.append(p)
                    y_list.append(y.numpy())
            probs = np.vstack(probs_list)
            yt = np.concatenate(y_list)
            auc = compute_macro_auc_ovr(yt, probs)
            rows.append({"model": label, "epoch": ep, "subset217_auc": float(auc)})
            if ep % 20 == 0 or ep == 1:
                print(f"  [{label}] ep {ep:>3}/{EPOCHS}  subset217_auc={auc:.4f}", flush=True)
        print(f"[done] {label} in {time.time()-t_model:.1f}s")

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "epoch", "subset217_auc"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[wrote] {OUT_CSV}  rows={len(rows)}")


if __name__ == "__main__":
    main()
