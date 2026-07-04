#!/usr/bin/env python3
"""
从 best_auc_model.pth + 固定 train/val 目录重新计算验证集指标与 bootstrap CI，
写回 result_summary.json（与 train_casgnet_contrastive_newdata.py 训练结束时的字段对齐）。

用法:
  python refresh_supcon_checkpoint_metrics.py checkpoints/casgnet_supcon_olddata_resplit_new
  python refresh_supcon_checkpoint_metrics.py checkpoints/casgnet_supcon_olddata_resplit_new \\
      --merge-comparison checkpoints/old_data_supcon_compare
  python refresh_supcon_checkpoint_metrics.py --comparison-refresh-all checkpoints/old_data_supcon_compare \\
      --project-root . --n-bootstrap 1000

还会在输出目录生成（验证集按类 one-vs-rest，bootstrap 次数与主流程 n_bootstrap 一致，默认 1000）:
  - per_class_metrics_val.csv / .xlsx / .json
  - 列与 comparison_summary.csv 相同：model, auc, sensitivity, specificity, npv, ppv, acc（三位小数 + 95% CI）
  result_summary.json 内 fields per_class_val、per_class_bootstrap_n。

合并对比目录下的 comparison_summary.xlsx 时为双工作表：
  - comparison_summary：宏观指标（与 CSV 一致）
  - per_class_val：列 experiment + 上述各指标；各类一行，stack 所有实验。
  实验目录解析顺序：comparison_experiment_dirs.json → <comparison_root>/<model> →
  与 comparison_root 同级的 <model>（用于 casgnet_* 等在 checkpoints/ 下的目录）。
  若无 checkpoint，仅有旧版 per_class_metrics_val.json（无概率）时，各类 auc 无法复原，该列为空。

对比目录下所有模型统一按 bootstrap（默认 1000）算各类指标：
  python refresh_comparison_per_class_metrics.py checkpoints/old_data_supcon_compare
  # 或等价：
  python refresh_supcon_checkpoint_metrics.py --comparison-refresh-all checkpoints/old_data_supcon_compare \\
      --project-root .

依赖各实验目录下 best_auc_model.pth；train_dir/val_dir 相对路径相对于 --project-root（默认推断为仓库根目录）。
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# 训练脚本中的评估逻辑（同一实现，避免重复）
from train_casgnet_contrastive_newdata import (
    SupConClassifierNet,
    TransformSubset,
    bootstrap_auc_ci,
    bootstrap_classification_metrics_ci,
    collect_val_probs,
    compute_macro_auc_ovr,
    compute_macro_classification_metrics,
)
from train_multiclass import ImageFolderDataset, get_data_augmentation

PER_CLASS_COMPARISON_FIELDS = ["model", "auc", "sensitivity", "specificity", "npv", "ppv", "acc"]


def fmt_ci(mean_v, low_v, high_v) -> str:
    """与 comparison_summary.csv 一致：三位小数，形如 0.xxx(lo-hi)。"""
    if mean_v is None:
        return ""
    try:
        m = float(mean_v)
    except Exception:
        return ""
    if not np.isfinite(m):
        return ""
    if low_v is None or high_v is None:
        return f"{m:.3f}"
    try:
        lo = float(low_v)
        hi = float(high_v)
    except Exception:
        return f"{m:.3f}"
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return f"{m:.3f}"
    return f"{m:.3f}({lo:.3f}-{hi:.3f})"


def _per_class_auc_ovr(y_true: np.ndarray, y_score: np.ndarray, n_classes: int) -> list[float | None]:
    """各类别 one-vs-rest AUC（与其它类别合并为负类）。"""
    from sklearn.metrics import roc_auc_score

    out: list[float | None] = []
    for c in range(n_classes):
        y_bin = (y_true == c).astype(np.int32)
        if np.unique(y_bin).size < 2:
            out.append(None)
            continue
        try:
            out.append(float(roc_auc_score(y_bin, y_score[:, c])))
        except (ValueError, TypeError):
            out.append(None)
    return out


def _finalize_boot(
    samples: list[float], point: float | None, p_lo: float, p_hi: float
) -> tuple[float, float, float]:
    """Bootstrap 均值与分位数 CI；无有效样本时退化为点估计 (m,m,m)。"""
    arr = np.asarray(samples, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        if point is not None and np.isfinite(point):
            p = float(point)
            return p, p, p
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.mean(arr)),
        float(np.percentile(arr, p_lo * 100)),
        float(np.percentile(arr, p_hi * 100)),
    )


def bootstrap_per_class_comparison_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    per_class_point: list[dict],
    aucs_point: list[float | None],
    class_names: list[str],
    n_boot: int,
    random_state: int,
    confidence: float = 0.95,
) -> list[dict]:
    """
    各类别 one-vs-rest：bootstrap n_boot 次后输出与 comparison_summary.csv 相同列、三位小数 CI 字符串。
    """
    from sklearn.metrics import roc_auc_score

    rng = np.random.RandomState(random_state)
    n = len(y_true)
    p_lo = (1.0 - confidence) / 2.0
    p_hi = 1.0 - p_lo
    n_classes = len(class_names)

    stores = {c: {k: [] for k in ("auc", "sensitivity", "specificity", "npv", "ppv", "acc")} for c in range(n_classes)}

    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        ys = y_score[idx]
        for c in range(n_classes):
            yt_bin = (yt == c).astype(np.int32)
            yp_bin = (yp == c).astype(np.int32)
            tp = int(np.sum((yt_bin == 1) & (yp_bin == 1)))
            tn = int(np.sum((yt_bin == 0) & (yp_bin == 0)))
            fp = int(np.sum((yt_bin == 0) & (yp_bin == 1)))
            fn = int(np.sum((yt_bin == 1) & (yp_bin == 0)))

            sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
            spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
            ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
            npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
            tot = tp + tn + fp + fn
            acc = (tp + tn) / tot if tot > 0 else float("nan")

            if np.unique(yt_bin).size >= 2:
                try:
                    auc_c = float(roc_auc_score(yt_bin, ys[:, c]))
                except (ValueError, TypeError):
                    auc_c = float("nan")
            else:
                auc_c = float("nan")

            stores[c]["auc"].append(auc_c)
            stores[c]["sensitivity"].append(sens)
            stores[c]["specificity"].append(spec)
            stores[c]["npv"].append(npv)
            stores[c]["ppv"].append(ppv)
            stores[c]["acc"].append(acc)

    rows: list[dict] = []
    for c in range(n_classes):
        pc = per_class_point[c]
        if int(pc["class_idx"]) != c:
            raise RuntimeError("per_class_metrics 与类别索引不一致")

        def _pt(x) -> float | None:
            try:
                v = float(x)
                return v if np.isfinite(v) else None
            except Exception:
                return None

        pt_sens = _pt(pc["sensitivity"])
        pt_spec = _pt(pc["specificity"])
        pt_npv = _pt(pc["npv"])
        pt_ppv = _pt(pc["ppv"])
        pt_acc = _pt(pc["acc"])
        pt_auc_f = _pt(aucs_point[c]) if c < len(aucs_point) else None

        m_auc, lo_auc, hi_auc = _finalize_boot(stores[c]["auc"], pt_auc_f, p_lo, p_hi)
        m_s, lo_s, hi_s = _finalize_boot(stores[c]["sensitivity"], pt_sens, p_lo, p_hi)
        m_sp, lo_sp, hi_sp = _finalize_boot(stores[c]["specificity"], pt_spec, p_lo, p_hi)
        m_npv, lo_npv, hi_npv = _finalize_boot(stores[c]["npv"], pt_npv, p_lo, p_hi)
        m_ppv, lo_ppv, hi_ppv = _finalize_boot(stores[c]["ppv"], pt_ppv, p_lo, p_hi)
        m_acc, lo_acc, hi_acc = _finalize_boot(stores[c]["acc"], pt_acc, p_lo, p_hi)

        sort_key = m_auc if np.isfinite(m_auc) else -1.0
        rows.append(
            {
                "model": class_names[c],
                "auc": fmt_ci(m_auc, lo_auc, hi_auc),
                "sensitivity": fmt_ci(m_s, lo_s, hi_s),
                "specificity": fmt_ci(m_sp, lo_sp, hi_sp),
                "npv": fmt_ci(m_npv, lo_npv, hi_npv),
                "ppv": fmt_ci(m_ppv, lo_ppv, hi_ppv),
                "acc": fmt_ci(m_acc, lo_acc, hi_acc),
                "_auc_sort": sort_key,
            }
        )

    rows.sort(key=lambda r: -float(r["_auc_sort"]))
    for r in rows:
        del r["_auc_sort"]
    return rows


def _export_per_class_test_tables(rows: list[dict], ckpt_dir: Path) -> None:
    """测试集各类指标（列同 comparison_summary.csv）。"""
    json_path = ckpt_dir / "per_class_metrics_test.json"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = ckpt_dir / "per_class_metrics_test.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PER_CLASS_COMPARISON_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in PER_CLASS_COMPARISON_FIELDS})

    df = pd.DataFrame(rows)[PER_CLASS_COMPARISON_FIELDS]
    xlsx_path = ckpt_dir / "per_class_metrics_test.xlsx"
    try:
        df.to_excel(xlsx_path, index=False)
        print(f"已写入: {json_path}")
        print(f"已写入: {csv_path}")
        print(f"已写入: {xlsx_path}")
    except ModuleNotFoundError as e:
        if "openpyxl" in str(e):
            print(f"已写入: {json_path}")
            print(f"已写入: {csv_path}")
            print("未安装 openpyxl，跳过 per_class_metrics_test.xlsx")
        else:
            raise


def _export_per_class_comparison_tables(rows: list[dict], ckpt_dir: Path) -> None:
    """列与 comparison_summary.csv 一致：model, auc, sensitivity, specificity, npv, ppv, acc。"""
    json_path = ckpt_dir / "per_class_metrics_val.json"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = ckpt_dir / "per_class_metrics_val.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PER_CLASS_COMPARISON_FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in PER_CLASS_COMPARISON_FIELDS})

    df = pd.DataFrame(rows)[PER_CLASS_COMPARISON_FIELDS]
    xlsx_path = ckpt_dir / "per_class_metrics_val.xlsx"
    try:
        df.to_excel(xlsx_path, index=False)
        print(f"已写入: {json_path}")
        print(f"已写入: {csv_path}")
        print(f"已写入: {xlsx_path}")
    except ModuleNotFoundError as e:
        if "openpyxl" in str(e):
            print(f"已写入: {json_path}")
            print(f"已写入: {csv_path}")
            print("未安装 openpyxl，跳过 per_class_metrics_val.xlsx")
        else:
            raise


def _infer_proj_dims(state_dict: dict) -> tuple[int, int]:
    w0 = state_dict.get("proj.0.weight")
    w2 = state_dict.get("proj.2.weight")
    if w0 is None or w2 is None:
        return 128, 512
    hidden_dim = int(w0.shape[0])
    proj_dim = int(w2.shape[0])
    return proj_dim, hidden_dim


def _remap_to_sorted_classes(train_base: ImageFolderDataset, val_base: ImageFolderDataset) -> dict:
    train_classes = set(train_base.class_to_idx.keys())
    val_classes = set(val_base.class_to_idx.keys())
    if train_classes != val_classes:
        raise ValueError(
            f"train/val 类别不一致: 仅 train {sorted(train_classes - val_classes)}; "
            f"仅 val {sorted(val_classes - train_classes)}"
        )
    class_names = sorted(train_classes)
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    idx_to_class = {i: name for name, i in class_to_idx.items()}

    def _remap(ds: ImageFolderDataset) -> None:
        remapped = []
        for p, y in ds.samples:
            cls_name = ds.idx_to_class[y]
            remapped.append((p, class_to_idx[cls_name]))
        ds.samples = remapped
        ds.class_to_idx = class_to_idx
        ds.idx_to_class = idx_to_class

    _remap(train_base)
    _remap(val_base)
    return class_to_idx


def _resolve_data_path(path_str: str, project_root: Path | None) -> Path:
    """将 result_summary 中的 train_dir/val_dir 转为绝对路径（相对路径相对 project_root，默认 cwd）。"""
    p = Path(path_str)
    if p.is_absolute():
        return p.resolve()
    base = (project_root if project_root is not None else Path.cwd()).resolve()
    return (base / p).resolve()


def refresh_checkpoint_dir(
    ckpt_dir: Path,
    *,
    project_root: Path | None = None,
    augmentation: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    n_bootstrap: int,
    seed: int,
    device: torch.device,
) -> dict:
    summary_path = ckpt_dir / "result_summary.json"
    best_path = ckpt_dir / "best_auc_model.pth"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    if not best_path.is_file():
        raise FileNotFoundError(best_path)

    prev = json.loads(summary_path.read_text(encoding="utf-8"))
    train_dir = prev.get("train_dir")
    val_dir = prev.get("val_dir")
    if not train_dir or not val_dir:
        raise ValueError("result_summary.json 缺少 train_dir / val_dir")

    train_root = _resolve_data_path(str(train_dir), project_root)
    val_root = _resolve_data_path(str(val_dir), project_root)
    if not train_root.is_dir():
        raise FileNotFoundError(f"训练目录不存在: {train_root}")
    if not val_root.is_dir():
        raise FileNotFoundError(f"验证目录不存在: {val_root}")

    try:
        ckpt = torch.load(best_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(best_path, map_location=device)

    state_dict = ckpt.get("state_dict", ckpt)
    num_classes = int(ckpt.get("num_classes", prev.get("num_classes")))
    model_name = str(ckpt.get("model", prev.get("model", "casgnet_s1")))
    proj_dim, hidden_dim = _infer_proj_dims(state_dict)

    train_aug, val_aug = get_data_augmentation(augmentation_type=augmentation, img_size=img_size)
    train_base = ImageFolderDataset(str(train_root), transform=None)
    val_base = ImageFolderDataset(str(val_root), transform=None)
    _remap_to_sorted_classes(train_base, val_base)

    va_idx = np.arange(len(val_base))
    val_ds = TransformSubset(val_base, va_idx, transform=val_aug)
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    # 尝试使用原始模型架构加载（不使用 SupConClassifierNet 包装）
    from models import casgnet
    
    # 直接创建 CASGNet 模型
    try:
        raw_model = casgnet.casgnet_s1(num_classes=num_classes, pretrained=False)
        raw_model.load_state_dict(state_dict, strict=True)
        print("Successfully loaded raw CASGNet model with strict=True")
        
        # 创建一个简单的包装器来提供 forward_logits 方法
        class SimpleModelWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            @torch.inference_mode()
            def forward_logits(self, x):
                return self.model(x)
        
        net = SimpleModelWrapper(raw_model)
    except RuntimeError as e:
        print(f"Failed to load raw model: {e}")
        print("Falling back to SupConClassifierNet...")
        
        # 回退到原来的 SupConClassifierNet 方式
        net = SupConClassifierNet(
            num_classes=num_classes,
            model_name=model_name,
            proj_dim=proj_dim,
            hidden_dim=hidden_dim,
            pretrained=False,
        )
        
        # 尝试处理state_dict中的键名不匹配问题
        try:
            net.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            if "Unexpected key(s)" in str(e):
                print(f"Warning: Key mismatch detected, trying to fix state_dict keys...")
                # 尝试移除 'encoder.' 前缀
                new_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith("encoder."):
                        new_key = key[8:]  # Remove 'encoder.' prefix
                        new_state_dict[new_key] = value
                    else:
                        new_state_dict[key] = value
                
                try:
                    net.load_state_dict(new_state_dict, strict=True)
                    print("Successfully loaded with fixed keys (removed 'encoder.' prefix)")
                except RuntimeError as e2:
                    print(f"Still failed after fixing keys: {e2}")
                    print("Trying with strict=False...")
                    net.load_state_dict(new_state_dict, strict=False)
            else:
                raise
    
    net = net.to(device)

    use_amp = device.type == "cuda"
    probs, yt, yhat = collect_val_probs(net, val_loader, device, use_amp)
    point_auc = compute_macro_auc_ovr(yt, probs)
    macro_cls_metrics, per_class_metrics = compute_macro_classification_metrics(yt, yhat, n_classes=num_classes)
    aucs_ovr_point = _per_class_auc_ovr(yt, probs, num_classes)
    idx_to_class: dict[int, str] = dict(val_base.idx_to_class)
    class_names_ordered = [idx_to_class[i] for i in range(num_classes)]

    pc_comparison_rows = bootstrap_per_class_comparison_rows(
        yt,
        yhat,
        probs,
        per_class_metrics,
        aucs_ovr_point,
        class_names_ordered,
        n_boot=n_bootstrap,
        random_state=seed,
    )

    mean_b, lo, hi = bootstrap_auc_ci(yt, probs, n_boot=n_bootstrap, random_state=seed)
    cls_boot = bootstrap_classification_metrics_ci(
        yt, yhat, n_classes=num_classes, n_boot=n_bootstrap, random_state=seed
    )

    best_saved = float(ckpt.get("val_auc", point_auc))

    test_eval_block: dict | None = None
    pc_test_rows: list[dict] | None = None
    n_test_samples = 0
    test_dir = prev.get("test_dir")
    if test_dir:
        test_root = _resolve_data_path(str(test_dir), project_root)
        if not test_root.is_dir():
            raise FileNotFoundError(f"测试目录不存在: {test_root}")
        test_base = ImageFolderDataset(str(test_root), transform=None)
        trn_cls = set(train_base.class_to_idx.keys())
        tst_cls = set(test_base.class_to_idx.keys())
        if trn_cls != tst_cls:
            raise ValueError(
                f"测试集类别与 train 不一致: 仅 train 有 {sorted(trn_cls - tst_cls)}; "
                f"仅 test 有 {sorted(tst_cls - trn_cls)}"
            )
        class_to_idx = train_base.class_to_idx
        idx_to_class_map = train_base.idx_to_class
        remapped_te = []
        for p, y in test_base.samples:
            cls_name = test_base.idx_to_class[y]
            remapped_te.append((p, class_to_idx[cls_name]))
        test_base.samples = remapped_te
        test_base.class_to_idx = class_to_idx
        test_base.idx_to_class = idx_to_class_map
        n_test_samples = len(test_base)
        te_idx = np.arange(n_test_samples)
        test_ds = TransformSubset(test_base, te_idx, transform=val_aug)
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )
        probs_te, yt_te, yhat_te = collect_val_probs(net, test_loader, device, use_amp)
        point_auc_te = compute_macro_auc_ovr(yt_te, probs_te)
        macro_te, per_class_te = compute_macro_classification_metrics(yt_te, yhat_te, n_classes=num_classes)
        aucs_ovr_te = _per_class_auc_ovr(yt_te, probs_te, num_classes)
        pc_test_rows = bootstrap_per_class_comparison_rows(
            yt_te,
            yhat_te,
            probs_te,
            per_class_te,
            aucs_ovr_te,
            class_names_ordered,
            n_boot=n_bootstrap,
            random_state=seed,
        )
        mean_b_te, lo_te, hi_te = bootstrap_auc_ci(yt_te, probs_te, n_boot=n_bootstrap, random_state=seed)
        cls_boot_te = bootstrap_classification_metrics_ci(
            yt_te, yhat_te, n_classes=num_classes, n_boot=n_bootstrap, random_state=seed
        )
        test_eval_block = {
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
                "n_bootstrap": n_bootstrap,
            },
            "bootstrap_metrics": {
                "sensitivity": cls_boot_te["sensitivity"],
                "specificity": cls_boot_te["specificity"],
                "npv": cls_boot_te["npv"],
                "ppv": cls_boot_te["ppv"],
                "acc": cls_boot_te["acc"],
                "n_bootstrap": n_bootstrap,
            },
            "n_test": int(n_test_samples),
            "reloaded_test_auc": point_auc_te,
        }

    summary = {
        **prev,
        "model": model_name,
        "n_train": int(prev.get("n_train", len(train_base))),
        "n_val": int(prev.get("n_val", len(val_base))),
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
            "n_bootstrap": n_bootstrap,
        },
        "bootstrap_metrics": {
            "sensitivity": cls_boot["sensitivity"],
            "specificity": cls_boot["specificity"],
            "npv": cls_boot["npv"],
            "ppv": cls_boot["ppv"],
            "acc": cls_boot["acc"],
            "n_bootstrap": n_bootstrap,
        },
        "per_class_val": pc_comparison_rows,
        "per_class_bootstrap_n": n_bootstrap,
    }
    if test_eval_block is not None and pc_test_rows is not None:
        summary["test_dir"] = test_dir
        summary["n_test"] = int(n_test_samples)
        summary["test_eval"] = test_eval_block
        summary["per_class_test"] = pc_test_rows

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _export_per_class_comparison_tables(pc_comparison_rows, ckpt_dir)
    if pc_test_rows is not None:
        _export_per_class_test_tables(pc_test_rows, ckpt_dir)
    print(f"已更新: {summary_path}")
    return summary


def row_from_test_eval(run_name: str, data: dict) -> dict | None:
    """从 result_summary.json 的 test_eval 块生成宏观一行（无测试集时返回 None）。"""
    te = data.get("test_eval")
    if not isinstance(te, dict):
        return None
    b = te.get("bootstrap_auc", {}) or {}
    bm = te.get("bootstrap_metrics", {}) or {}

    sens_b = bm.get("sensitivity", {}) if isinstance(bm.get("sensitivity", {}), dict) else {}
    spec_b = bm.get("specificity", {}) if isinstance(bm.get("specificity", {}), dict) else {}
    npv_b = bm.get("npv", {}) if isinstance(bm.get("npv", {}), dict) else {}
    ppv_b = bm.get("ppv", {}) if isinstance(bm.get("ppv", {}), dict) else {}
    acc_b = bm.get("acc", {}) if isinstance(bm.get("acc", {}), dict) else {}

    auc_point = te.get("auc", te.get("reloaded_test_auc"))
    auc_mean = b.get("mean", auc_point)

    return {
        "model": run_name,
        "auc": fmt_ci(auc_mean, b.get("ci95_low"), b.get("ci95_high")),
        "sensitivity": fmt_ci(
            sens_b.get("mean", te.get("sensitivity")),
            sens_b.get("ci95_low"),
            sens_b.get("ci95_high"),
        ),
        "specificity": fmt_ci(
            spec_b.get("mean", te.get("specificity")),
            spec_b.get("ci95_low"),
            spec_b.get("ci95_high"),
        ),
        "npv": fmt_ci(npv_b.get("mean", te.get("npv")), npv_b.get("ci95_low"), npv_b.get("ci95_high")),
        "ppv": fmt_ci(ppv_b.get("mean", te.get("ppv")), ppv_b.get("ci95_low"), ppv_b.get("ci95_high")),
        "acc": fmt_ci(acc_b.get("mean", te.get("acc")), acc_b.get("ci95_low"), acc_b.get("ci95_high")),
        "_auc_sort": auc_point if auc_point is not None else -1.0,
    }


def row_from_summary(run_name: str, data: dict) -> dict:
    b = data.get("bootstrap_auc", {}) or {}
    bm = data.get("bootstrap_metrics", {}) or {}

    sens_b = bm.get("sensitivity", {}) if isinstance(bm.get("sensitivity", {}), dict) else {}
    spec_b = bm.get("specificity", {}) if isinstance(bm.get("specificity", {}), dict) else {}
    npv_b = bm.get("npv", {}) if isinstance(bm.get("npv", {}), dict) else {}
    ppv_b = bm.get("ppv", {}) if isinstance(bm.get("ppv", {}), dict) else {}
    acc_b = bm.get("acc", {}) if isinstance(bm.get("acc", {}), dict) else {}

    auc_point = data.get("auc", data.get("reloaded_val_auc"))
    auc_mean = b.get("mean", auc_point)

    return {
        "model": run_name,
        "auc": fmt_ci(auc_mean, b.get("ci95_low"), b.get("ci95_high")),
        "sensitivity": fmt_ci(
            sens_b.get("mean", data.get("sensitivity")),
            sens_b.get("ci95_low"),
            sens_b.get("ci95_high"),
        ),
        "specificity": fmt_ci(
            spec_b.get("mean", data.get("specificity")),
            spec_b.get("ci95_low"),
            spec_b.get("ci95_high"),
        ),
        "npv": fmt_ci(npv_b.get("mean", data.get("npv")), npv_b.get("ci95_low"), npv_b.get("ci95_high")),
        "ppv": fmt_ci(ppv_b.get("mean", data.get("ppv")), ppv_b.get("ci95_low"), ppv_b.get("ci95_high")),
        "acc": fmt_ci(acc_b.get("mean", data.get("acc")), acc_b.get("ci95_low"), acc_b.get("ci95_high")),
        "_auc_sort": auc_point if auc_point is not None else -1.0,
    }


def infer_project_root(comparison_root: Path) -> Path:
    """假定 comparison_root 为 <repo>/checkpoints/<compare_tag>，则项目根含 train_multiclass.py。"""
    cand = comparison_root.resolve().parent.parent
    if (cand / "train_multiclass.py").is_file():
        return cand
    return comparison_root.resolve()


def load_comparison_dir_map(comparison_root: Path) -> dict[str, str]:
    p = comparison_root / "comparison_experiment_dirs.json"
    if not p.is_file():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items()}


def resolve_experiment_dir(comparison_root: Path, experiment_name: str, dir_map: dict[str, str]) -> Path | None:
    if experiment_name in dir_map:
        q = Path(dir_map[experiment_name])
        path = q.resolve() if q.is_absolute() else (comparison_root / q).resolve()
        if path.is_dir():
            return path
    direct = (comparison_root / experiment_name).resolve()
    if direct.is_dir():
        return direct
    sibling = (comparison_root.parent / experiment_name).resolve()
    if sibling.is_dir():
        return sibling
    return None


def _rows_from_per_class_val_list(pc: list) -> list[dict]:
    out: list[dict] = []
    for r in pc:
        if not isinstance(r, dict):
            continue
        row = {k: r.get(k, "") for k in PER_CLASS_COMPARISON_FIELDS}
        out.append(row)
    return out


def legacy_per_class_to_comparison_rows(
    legacy: list, summary_path: Path, project_root: Path
) -> list[dict]:
    """旧版 JSON（tp/fp、无 AUC）：输出三位小数点估计，auc 列为空。"""
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    val_dir = data.get("val_dir")
    if not val_dir:
        return []
    val_root = _resolve_data_path(str(val_dir), project_root)
    if not val_root.is_dir():
        return []
    val_base = ImageFolderDataset(str(val_root), transform=None)
    class_names = sorted(val_base.class_to_idx.keys())
    rows: list[dict] = []
    for item in legacy:
        if not isinstance(item, dict):
            continue
        idx = int(item.get("class_idx", -1))
        name = class_names[idx] if 0 <= idx < len(class_names) else str(idx)
        rows.append(
            {
                "model": name,
                "auc": "",
                "sensitivity": fmt_ci(item.get("sensitivity"), None, None),
                "specificity": fmt_ci(item.get("specificity"), None, None),
                "npv": fmt_ci(item.get("npv"), None, None),
                "ppv": fmt_ci(item.get("ppv"), None, None),
                "acc": fmt_ci(item.get("acc"), None, None),
            }
        )
    return rows


def load_per_class_rows_for_experiment_test(
    exp_dir: Path, project_root: Path, *, allow_legacy_fallback: bool = True
) -> tuple[list[dict], str]:
    summary_path = exp_dir / "result_summary.json"
    if summary_path.is_file():
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
        pc = data.get("per_class_test")
        if isinstance(pc, list) and len(pc) > 0:
            rows = _rows_from_per_class_val_list(pc)
            if rows:
                return rows, "result_summary.per_class_test"

    csv_path = exp_dir / "per_class_metrics_test.csv"
    if csv_path.is_file():
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            df = None
        if df is not None and all(c in df.columns for c in PER_CLASS_COMPARISON_FIELDS):
            return df[PER_CLASS_COMPARISON_FIELDS].to_dict("records"), "per_class_metrics_test.csv"

    json_path = exp_dir / "per_class_metrics_test.json"
    if json_path.is_file():
        try:
            raw = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            raw = None
        if isinstance(raw, list) and raw:
            fst = raw[0]
            if isinstance(fst, dict) and isinstance(fst.get("auc"), str):
                return _rows_from_per_class_val_list(raw), "per_class_metrics_test.json"
            if isinstance(fst, dict) and "class_idx" in fst and summary_path.is_file():
                if not allow_legacy_fallback:
                    return [], "legacy_skipped"
                leg = legacy_per_class_to_comparison_rows(raw, summary_path, project_root)
                if leg:
                    return leg, "legacy_json_test"

    return [], "missing"


def load_per_class_rows_for_experiment(
    exp_dir: Path, project_root: Path, *, allow_legacy_fallback: bool = True
) -> tuple[list[dict], str]:
    summary_path = exp_dir / "result_summary.json"
    if summary_path.is_file():
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
        pc = data.get("per_class_val")
        if isinstance(pc, list) and len(pc) > 0:
            rows = _rows_from_per_class_val_list(pc)
            if rows:
                return rows, "result_summary.per_class_val"

    csv_path = exp_dir / "per_class_metrics_val.csv"
    if csv_path.is_file():
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            df = None
        if df is not None and all(c in df.columns for c in PER_CLASS_COMPARISON_FIELDS):
            return df[PER_CLASS_COMPARISON_FIELDS].to_dict("records"), "per_class_metrics_val.csv"

    json_path = exp_dir / "per_class_metrics_val.json"
    if json_path.is_file():
        try:
            raw = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            raw = None
        if isinstance(raw, list) and raw:
            fst = raw[0]
            if isinstance(fst, dict) and isinstance(fst.get("auc"), str):
                return _rows_from_per_class_val_list(raw), "per_class_metrics_val.json"
            if isinstance(fst, dict) and "class_idx" in fst and summary_path.is_file():
                if not allow_legacy_fallback:
                    return [], "legacy_skipped"
                leg = legacy_per_class_to_comparison_rows(raw, summary_path, project_root)
                if leg:
                    return leg, "legacy_json"

    return [], "missing"


def read_comparison_csv_models(comparison_root: Path) -> list[str]:
    csv_path = comparison_root / "comparison_summary.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    names: list[str] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            m = (row.get("model") or "").strip()
            if m:
                names.append(m)
    return names


def refresh_comparison_all_per_class(
    comparison_root: Path,
    *,
    project_root: Path | None = None,
    augmentation: str = "standard",
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    n_bootstrap: int = 1000,
    seed: int = 42,
    device: torch.device | None = None,
    write_xlsx: bool = True,
    allow_legacy_fallback: bool = False,
) -> None:
    """
    按 comparison_summary.csv 中的 model 列，对每个实验目录调用 refresh_checkpoint_dir（默认 n_bootstrap=1000）。
    全部成功后写入双表 comparison_summary.xlsx（默认不允许 legacy 占位）。
    """
    comparison_root = comparison_root.resolve()
    proj = project_root.resolve() if project_root is not None else infer_project_root(comparison_root)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dir_map = load_comparison_dir_map(comparison_root)
    failures: list[str] = []
    models = read_comparison_csv_models(comparison_root)

    for name in models:
        exp_dir = resolve_experiment_dir(comparison_root, name, dir_map)
        if exp_dir is None:
            failures.append(f"{name}: 未找到实验目录（可配置 comparison_experiment_dirs.json）")
            continue
        if not (exp_dir / "best_auc_model.pth").is_file():
            failures.append(f"{name}: 缺少 {exp_dir / 'best_auc_model.pth'}")
            continue
        print(f"\n>>> [{name}] {exp_dir}")
        refresh_checkpoint_dir(
            exp_dir,
            project_root=proj,
            augmentation=augmentation,
            img_size=img_size,
            batch_size=batch_size,
            num_workers=num_workers,
            n_bootstrap=n_bootstrap,
            seed=seed,
            device=device,
        )

    if failures:
        raise RuntimeError(
            "下列实验未能完成 bootstrap 刷新（需要目录与 best_auc_model.pth）：\n"
            + "\n".join(failures)
        )

    rebuild_comparison_macro_csv(comparison_root)
    rebuild_comparison_macro_csv_test(comparison_root)

    if write_xlsx:
        write_comparison_workbook(
            comparison_root,
            project_root=proj,
            allow_legacy_fallback=allow_legacy_fallback,
        )


def rebuild_comparison_macro_csv(comparison_root: Path) -> None:
    """按 comparison_summary.csv 中的 model 列，用各实验目录最新 result_summary 重写宏观 CSV。"""
    comparison_root = comparison_root.resolve()
    csv_path = comparison_root / "comparison_summary.csv"
    fieldnames = ["model", "auc", "sensitivity", "specificity", "npv", "ppv", "acc"]
    dir_map = load_comparison_dir_map(comparison_root)
    names = read_comparison_csv_models(comparison_root)

    rows_out: list[dict] = []
    missing: list[str] = []
    for name in names:
        exp_dir = resolve_experiment_dir(comparison_root, name, dir_map)
        if exp_dir is None:
            missing.append(name)
            continue
        summary_path = exp_dir / "result_summary.json"
        if not summary_path.is_file():
            missing.append(name)
            continue
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        rows_out.append(row_from_summary(name, data))

    if missing:
        raise RuntimeError(
            "无法重写 comparison_summary.csv，下列实验无目录或 result_summary.json：\n" + "\n".join(missing)
        )

    rows_out.sort(key=lambda x: -(x.get("_auc_sort", -1.0) or -1.0))
    for x in rows_out:
        x.pop("_auc_sort", None)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)
    print(f"已按刷新后的指标重写: {csv_path}")


def rebuild_comparison_macro_csv_val_export(comparison_root: Path, *, output_filename: str = "comparison_summary_val.csv") -> None:
    """验证集（old_data/val）宏观 + 按类汇总写入同一 CSV：先 macro，空一行后为 experiment × class 堆叠表。"""
    comparison_root = comparison_root.resolve()
    csv_path = comparison_root / output_filename
    fieldnames_macro = ["model", "auc", "sensitivity", "specificity", "npv", "ppv", "acc"]
    fieldnames_pc = ["experiment"] + PER_CLASS_COMPARISON_FIELDS
    dir_map = load_comparison_dir_map(comparison_root)
    names = read_comparison_csv_models(comparison_root)

    rows_out: list[dict] = []
    missing: list[str] = []
    summary_by_exp: dict[str, dict] = {}
    for name in names:
        exp_dir = resolve_experiment_dir(comparison_root, name, dir_map)
        if exp_dir is None:
            missing.append(name)
            continue
        summary_path = exp_dir / "result_summary.json"
        if not summary_path.is_file():
            missing.append(name)
            continue
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        summary_by_exp[name] = data
        rows_out.append(row_from_summary(name, data))

    if missing:
        raise RuntimeError(
            f"无法写入 {output_filename}，下列实验无目录或 result_summary.json：\n" + "\n".join(missing)
        )

    rows_out.sort(key=lambda x: -(x.get("_auc_sort", -1.0) or -1.0))
    for x in rows_out:
        x.pop("_auc_sort", None)

    stacked_pc: list[dict] = []
    missing_pc: list[str] = []
    for r in rows_out:
        exp_name = r["model"]
        data = summary_by_exp.get(exp_name, {})
        pcl = data.get("per_class_val")
        if not isinstance(pcl, list) or len(pcl) == 0:
            missing_pc.append(exp_name)
            continue
        for row in pcl:
            if not isinstance(row, dict):
                continue
            stacked_pc.append(
                {
                    "experiment": exp_name,
                    **{k: row.get(k, "") for k in PER_CLASS_COMPARISON_FIELDS},
                }
            )

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames_macro)
        w.writeheader()
        w.writerows(rows_out)
        f.write("\n\n")
        w2 = csv.DictWriter(f, fieldnames=fieldnames_pc)
        w2.writeheader()
        w2.writerows(stacked_pc)

    print(f"已写入验证集 CSV（macro + per-class）: {csv_path}")
    if missing_pc:
        print(f"[{output_filename}] 无 per_class_val，跳过按类块: {', '.join(missing_pc)}")


def rebuild_comparison_macro_csv_from_test_eval(comparison_root: Path, *, output_filename: str) -> None:
    """与各实验目录 result_summary.json 中的 test_eval 对齐（对应训练时 result_summary['test_dir']，一般为 old_data/test）。"""
    comparison_root = comparison_root.resolve()
    csv_path = comparison_root / output_filename
    fieldnames = ["model", "auc", "sensitivity", "specificity", "npv", "ppv", "acc"]
    dir_map = load_comparison_dir_map(comparison_root)
    names = read_comparison_csv_models(comparison_root)

    rows_out: list[dict] = []
    skipped: list[str] = []
    for name in names:
        exp_dir = resolve_experiment_dir(comparison_root, name, dir_map)
        if exp_dir is None:
            skipped.append(name)
            continue
        summary_path = exp_dir / "result_summary.json"
        if not summary_path.is_file():
            skipped.append(name)
            continue
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        row = row_from_test_eval(name, data)
        if row is None:
            skipped.append(f"{name} (无 test_eval)")
            continue
        rows_out.append(row)

    label = output_filename.replace(".csv", "")
    if skipped:
        print(f"[{label}] 跳过或无 test_eval: {', '.join(skipped)}")
    rows_out.sort(key=lambda x: -(x.get("_auc_sort", -1.0) or -1.0))
    for x in rows_out:
        x.pop("_auc_sort", None)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)
    print(f"已写入（来自 result_summary.test_eval）: {csv_path}")


def rebuild_comparison_macro_csv_test(comparison_root: Path) -> None:
    """与各实验目录 result_summary.json 中的 test_eval 对齐，写 comparison_summary_test.csv。"""
    rebuild_comparison_macro_csv_from_test_eval(comparison_root, output_filename="comparison_summary_test.csv")


def apply_comparison_layout_test_primary_eltra(
    comparison_root: Path,
    *,
    project_root: Path | None = None,
    eltra_rel: str = "eltra_test",
    skip_eltra_infer: bool = False,
) -> None:
    """
    生成用户期望的宏观表含义：
    - comparison_summary.csv：各模型在「训练时 test_dir」（一般为 old_data/test）上的 test_eval，由 refresh 写入 result_summary。
    - comparison_summary_test.csv：在 eltra_test 上重新推理得到的宏观指标（调用 compare_models_on_eltra_test.py）。
    - comparison_summary_val.csv：验证集宏观表 + 空行 + 各模型 per_class_val 堆叠（experiment, model=类别名, 指标列）。

    说明：comparison_summary.xlsx 中「per_class_test」工作表仍来自各实验目录 refresh 时的 old_data/test（非 eltra），
    与第二张宏观表数据来源不一致；若需 eltra 各类指标可使用 compare_models_on_eltra_test 生成的 *_per_class.csv。
    """
    comparison_root = comparison_root.resolve()
    proj = project_root.resolve() if project_root else infer_project_root(comparison_root)

    rebuild_comparison_macro_csv_val_export(comparison_root, output_filename="comparison_summary_val.csv")
    rebuild_comparison_macro_csv_from_test_eval(comparison_root, output_filename="comparison_summary.csv")

    if not skip_eltra_infer:
        script = proj / "compare_models_on_eltra_test.py"
        if not script.is_file():
            raise FileNotFoundError(f"未找到 {script}")
        out_te = comparison_root / "comparison_summary_test.csv"
        cmd = [
            sys.executable,
            str(script),
            "--comparison-root",
            str(comparison_root),
            "--test-dir",
            eltra_rel,
            "--output-csv",
            str(out_te),
            "--project-root",
            str(proj),
        ]
        print("运行:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    write_comparison_workbook(comparison_root, project_root=proj, allow_legacy_fallback=True)
    print("已完成布局: comparison_summary.csv=old_data/test(test_eval), comparison_summary_test.csv=eltra_test")


def write_comparison_workbook(
    comparison_root: Path,
    *,
    project_root: Path | None = None,
    macro_sheet: str = "comparison_summary",
    macro_sheet_test: str = "comparison_summary_test",
    per_class_sheet: str = "per_class_val",
    per_class_sheet_test: str = "per_class_test",
    allow_legacy_fallback: bool = True,
) -> None:
    """读取 comparison_summary*.csv，写入四表 xlsx（验证宏观 / 测试宏观 / 各类验证 / 各类测试）。"""
    comparison_root = comparison_root.resolve()
    csv_path = comparison_root / "comparison_summary.csv"
    xlsx_path = comparison_root / "comparison_summary.xlsx"
    if not csv_path.is_file():
        print(f"跳过写入 xlsx：缺少 {csv_path}")
        return

    proj = project_root.resolve() if project_root is not None else infer_project_root(comparison_root)

    csv_test_path = comparison_root / "comparison_summary_test.csv"
    macro_rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if not r.get("model"):
                continue
            macro_rows.append({k: r.get(k, "") for k in PER_CLASS_COMPARISON_FIELDS})

    macro_rows_test: list[dict] = []
    if csv_test_path.is_file():
        with csv_test_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if not r.get("model"):
                    continue
                macro_rows_test.append({k: r.get(k, "") for k in PER_CLASS_COMPARISON_FIELDS})

    dir_map = load_comparison_dir_map(comparison_root)
    stacked: list[dict] = []
    stacked_test: list[dict] = []
    notes: list[str] = []
    for mr in macro_rows:
        exp = mr["model"]
        exp_dir = resolve_experiment_dir(comparison_root, exp, dir_map)
        if exp_dir is None:
            notes.append(f"[per-class] 未找到实验目录: {exp}")
            continue
        subrows, src = load_per_class_rows_for_experiment(
            exp_dir, proj, allow_legacy_fallback=allow_legacy_fallback
        )
        if not subrows:
            if src == "legacy_skipped":
                notes.append(
                    f"[per-class] {exp}: 仅有旧版 JSON，已跳过（请运行 "
                    f"python refresh_comparison_per_class_metrics.py {comparison_root}）"
                )
            else:
                notes.append(f"[per-class] 无各类指标: {exp} ({exp_dir})")
            continue
        if src == "legacy_json":
            notes.append(
                f"[per-class] {exp}: 使用旧版 JSON（无 bootstrap CI，auc 为空）；"
                f"若需统一 bootstrap 请运行 refresh_comparison_per_class_metrics.py"
            )
        for sr in subrows:
            stacked.append({"experiment": exp, **sr})

    for mr in macro_rows_test or []:
        exp = mr["model"]
        exp_dir = resolve_experiment_dir(comparison_root, exp, dir_map)
        if exp_dir is None:
            notes.append(f"[per-class-test] 未找到实验目录: {exp}")
            continue
        subrows_te, src_te = load_per_class_rows_for_experiment_test(
            exp_dir, proj, allow_legacy_fallback=allow_legacy_fallback
        )
        if not subrows_te:
            notes.append(f"[per-class-test] 无各类指标: {exp} ({exp_dir}) [{src_te}]")
            continue
        for sr in subrows_te:
            stacked_test.append({"experiment": exp, **sr})

    macro_df = pd.DataFrame(macro_rows)
    macro_df_test = pd.DataFrame(macro_rows_test)
    pc_cols = ["experiment"] + PER_CLASS_COMPARISON_FIELDS
    per_class_df = pd.DataFrame(stacked, columns=pc_cols) if stacked else pd.DataFrame(columns=pc_cols)
    per_class_df_test = (
        pd.DataFrame(stacked_test, columns=pc_cols) if stacked_test else pd.DataFrame(columns=pc_cols)
    )

    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            macro_df.to_excel(writer, sheet_name=macro_sheet, index=False)
            macro_df_test.to_excel(writer, sheet_name=macro_sheet_test, index=False)
            per_class_df.to_excel(writer, sheet_name=per_class_sheet, index=False)
            per_class_df_test.to_excel(writer, sheet_name=per_class_sheet_test, index=False)
        print(f"已写入四表工作簿: {xlsx_path}")
        for n in notes:
            print(n)
    except ModuleNotFoundError as e:
        if "openpyxl" in str(e):
            print(f"未安装 openpyxl，跳过 comparison_summary.xlsx（CSV 已就绪）: {csv_path}")
        else:
            raise


def merge_comparison(comparison_root: Path, run_name: str, summary: dict) -> None:
    csv_path = comparison_root / "comparison_summary.csv"
    fieldnames = ["model", "auc", "sensitivity", "specificity", "npv", "ppv", "acc"]

    rows: list[dict] = []
    if csv_path.is_file():
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if not r.get("model"):
                    continue
                if r["model"] == run_name:
                    continue
                auc_cell = r.get("auc", "")
                sort_key = -1.0
                if auc_cell:
                    try:
                        sort_key = float(str(auc_cell).split("(", 1)[0].strip())
                    except ValueError:
                        pass
                rows.append({**{k: r.get(k, "") for k in fieldnames}, "_auc_sort": sort_key})

    new_row = row_from_summary(run_name, summary)
    rows.append(new_row)
    rows.sort(key=lambda x: -(x.get("_auc_sort", -1.0) or -1.0))
    for x in rows:
        x.pop("_auc_sort", None)

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"已更新: {csv_path}")
    rebuild_comparison_macro_csv_test(comparison_root.resolve())
    write_comparison_workbook(comparison_root.resolve(), allow_legacy_fallback=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="补全 SupCon 实验的 result_summary 并可选合并对比表")
    ap.add_argument("checkpoint_dir", type=Path, nargs="?", help="含 result_summary.json 与 best_auc_model.pth 的目录")
    ap.add_argument(
        "--comparison-refresh-all",
        type=Path,
        default=None,
        help="对比目录：按 comparison_summary.csv 对所有实验做 bootstrap 各类指标并写 comparison_summary.xlsx",
    )
    ap.add_argument("--merge-comparison", type=Path, default=None, help="例如 checkpoints/old_data_supcon_compare")
    ap.add_argument("--run-name", type=str, default=None, help="写入对比表时的模型名（默认用 checkpoint 目录名）")
    ap.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="解析 train_dir/val_dir 相对路径时的项目根（默认：对比目录的上两级或 cwd）",
    )
    ap.add_argument("--augmentation", type=str, default="standard")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument(
        "--comparison-layout-test-primary-eltra",
        type=Path,
        default=None,
        metavar="COMPARISON_ROOT",
        help="调整宏观 CSV：comparison_summary.csv=各模型 result_summary.test_eval（通常为 old_data/test），"
        "comparison_summary_test.csv=在 eltra_test 上推理；comparison_summary_val.csv=验证集；并重写 comparison_summary.xlsx",
    )
    ap.add_argument(
        "--comparison-layout-skip-eltra",
        action="store_true",
        help="与上一项连用：跳过 eltra_test 推理，仅根据已有 JSON 重写 CSV（不写 comparison_summary_test.csv）",
    )
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.comparison_layout_test_primary_eltra:
        lay_root = args.comparison_layout_test_primary_eltra.resolve()
        proj = args.project_root.resolve() if args.project_root else None
        apply_comparison_layout_test_primary_eltra(
            lay_root,
            project_root=proj,
            skip_eltra_infer=args.comparison_layout_skip_eltra,
        )
        return

    if args.comparison_refresh_all:
        root = args.comparison_refresh_all.resolve()
        proj = args.project_root.resolve() if args.project_root else None
        refresh_comparison_all_per_class(
            root,
            project_root=proj,
            augmentation=args.augmentation,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            device=device,
        )
        return

    if args.checkpoint_dir is None:
        ap.error("请提供 checkpoint_dir，或使用 --comparison-refresh-all")

    ckpt_dir = args.checkpoint_dir.resolve()
    proj = args.project_root.resolve() if args.project_root else None

    summary = refresh_checkpoint_dir(
        ckpt_dir,
        project_root=proj,
        augmentation=args.augmentation,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        device=device,
    )

    if args.merge_comparison:
        run_name = args.run_name or ckpt_dir.name
        merge_comparison(args.merge_comparison.resolve(), run_name, summary)


if __name__ == "__main__":
    main()
