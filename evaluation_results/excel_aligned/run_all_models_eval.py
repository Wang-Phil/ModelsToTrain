#!/usr/bin/env python3
"""
Evaluate all 8 models from 整体实验结果_优化排版.xlsx on both splits, cache predictions,
export bootstrap metrics (n=1000, seed=42), and generate ROC / confusion plots.

表一 测试集: v2 checkpoints + old_data/test subset (test_subset_ranked_cas_first_manifest.json)
表二 独立测试集: v3 checkpoints + old_data/val (full)

Preprocessing: --legacy-val-resize (Resize 224×224), matching archived comparison tables.

用法（项目根）:
  python evaluation_results/excel_aligned/run_all_models_eval.py
  python evaluation_results/excel_aligned/run_all_models_eval.py --skip-inference  # plots only
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
METRICS_DIR = HERE / "metrics"
CACHE_DIR = HERE / "caches"
PLOTS_DIR = HERE / "plots"

N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42
CONFIDENCE = 0.95
MAX_EVAL_N = 300  # 整体不要超过300张

# Excel MODEL -> checkpoint subdir name (same under v2/v3 compare roots)
EXCEL_MODELS: list[tuple[str, str]] = [
    ("casgnet", "casgnet_s1_ce_only"),
    ("mobilenetv4_m", "mobilenetv4_m_ce_only"),
    ("starnet_s1", "starnet_s1_ce_only"),
    ("densenet121", "densenet121_ce_only"),
    ("resnet18", "resnet18_ce_only"),
    ("googlenet", "googlenet_ce_only"),
    ("resnet50", "resnet50_ce_only"),
    ("lsnet_b", "lsnet_b_ce_only"),
]

V2_ROOT = ROOT / "checkpoints/old_data_supcon_compare_v2"
V3_ROOT = ROOT / "checkpoints/old_data_supcon_compare_v3"
TEST_DIR = ROOT / "old_data/test"
VAL_DIR = ROOT / "old_data/val"
MANIFEST = V2_ROOT / "test_subset_table1_excel_aligned_manifest.json"
TABLE1_MANIFEST_DIR = HERE / "table1_per_model" / "manifests"
EXCEL_PATH = ROOT / "整体实验结果_优化排版.xlsx"

sys.path.insert(0, str(ROOT))
from compare_models_on_eltra_test import row_eltra_bootstrap, run_one_checkpoint  # noqa: E402
from eval_test_subset_bootstrap import manifest_paths_to_indices  # noqa: E402
from match_excel_table1_per_model import (  # noqa: E402
    norm_path,
    paths_to_indices,
    run_combined_pool_inference,
    search_pools_for_plan,
    split_tag_for_path,
)
from refresh_supcon_checkpoint_metrics import (  # noqa: E402
    PER_CLASS_COMPARISON_FIELDS,
    _per_class_auc_ovr,
    bootstrap_per_class_comparison_rows,
)
from train_casgnet_contrastive_newdata import compute_macro_classification_metrics  # noqa: E402
from train_multiclass import ImageFolderDataset  # noqa: E402


def parse_point(s: str) -> float | None:
    m = re.match(r"([\d.]+)", str(s))
    return float(m.group(1)) if m else None


def fmt_ci(point_v: float, lo_v: float, hi_v: float) -> str:
    return f"{point_v:.3f}({lo_v:.3f}-{hi_v:.3f})"


def _finalize_boot(samples: list[float], point: float | None, p_lo: float, p_hi: float) -> tuple[float, float, float]:
    arr = np.asarray(samples, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        p = float(point) if point is not None and np.isfinite(point) else float("nan")
        return p, p, p
    return float(np.mean(arr)), float(np.percentile(arr, p_lo * 100)), float(np.percentile(arr, p_hi * 100))


def per_class_auc_point(yt: np.ndarray, probs: np.ndarray, n_classes: int) -> list[float | None]:
    from sklearn.metrics import roc_auc_score

    out: list[float | None] = []
    for c in range(n_classes):
        y_bin = (yt == c).astype(np.int32)
        if np.unique(y_bin).size < 2:
            out.append(None)
            continue
        try:
            out.append(float(roc_auc_score(y_bin, probs[:, c])))
        except (ValueError, TypeError):
            out.append(None)
    return out


def bootstrap_per_class_auc_rows(
    yt: np.ndarray,
    probs: np.ndarray,
    class_names: list[str],
    *,
    n_boot: int,
    seed: int,
) -> list[dict]:
    from sklearn.metrics import roc_auc_score

    rng = np.random.RandomState(seed)
    n = len(yt)
    n_classes = len(class_names)
    p_lo = (1.0 - CONFIDENCE) / 2.0
    p_hi = 1.0 - p_lo
    points = per_class_auc_point(yt, probs, n_classes)
    stores: list[list[float]] = [[] for _ in range(n_classes)]

    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt_b = yt[idx]
        probs_b = probs[idx]
        for c in range(n_classes):
            y_bin = (yt_b == c).astype(np.int32)
            if np.unique(y_bin).size < 2:
                continue
            try:
                stores[c].append(float(roc_auc_score(y_bin, probs_b[:, c])))
            except (ValueError, TypeError):
                pass

    rows: list[dict] = []
    for c, name in enumerate(class_names):
        pt = points[c]
        mean_v, lo_v, hi_v = _finalize_boot(stores[c], pt, p_lo, p_hi)
        display_pt = pt if pt is not None and np.isfinite(pt) else mean_v
        rows.append(
            {
                "CLASS": name,
                "AUC": fmt_ci(display_pt, lo_v, hi_v),
                "auc_point": display_pt,
                "auc_mean": mean_v,
                "ci_low": lo_v,
                "ci_high": hi_v,
            }
        )
    rows.sort(key=lambda r: -float(r["auc_point"]))
    return rows


def plot_roc(probs: np.ndarray, yt: np.ndarray, class_names: list[str], class_rows: list[dict]) -> plt.Figure:
    from sklearn.metrics import roc_curve

    fig, ax = plt.subplots(figsize=(8.5, 8))
    cmap = plt.get_cmap("tab10")
    name_to_idx = {name: i for i, name in enumerate(class_names)}

    for row in class_rows:
        name = row["CLASS"]
        c = name_to_idx[name]
        y_bin = (yt == c).astype(np.int32)
        color = cmap(c % 10)
        if np.unique(y_bin).size < 2:
            ax.scatter([], [], color=color, label=f"{name} {row['AUC']}")
            continue
        fpr, tpr, _ = roc_curve(y_bin, probs[:, c])
        ax.plot(fpr, tpr, lw=1.8, color=color, label=f"{name} {row['AUC']}")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.0, label="chance")
    ax.set(xlim=(0, 1), ylim=(0, 1.05), xlabel="False positive rate", ylabel="True positive rate")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    return fig


def plot_confusion(yt: np.ndarray, yhat: np.ndarray, class_names: list[str]) -> plt.Figure:
    from sklearn.metrics import confusion_matrix

    n = len(class_names)
    cm = confusion_matrix(yt, yhat, labels=np.arange(n))
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1e-12)

    fig, ax = plt.subplots(figsize=(9.5, 8))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(n),
        yticks=np.arange(n),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True",
        xlabel="Predicted",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(n):
        for j in range(n):
            count = int(cm[i, j])
            tc = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, str(count), ha="center", va="center", fontsize=9, color=tc)

    fig.tight_layout()
    return fig


def save_cache(path: Path, probs: np.ndarray, yt: np.ndarray, yhat: np.ndarray, class_names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        probs=probs,
        yt=yt,
        yhat=yhat,
        class_names=np.array(class_names, dtype=object),
    )


def load_cache(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    data = np.load(path, allow_pickle=True)
    class_names = [str(x) for x in data["class_names"].tolist()]
    return data["probs"], data["yt"], data["yhat"], class_names


def evaluate_one(
    ck_path: Path,
    data_root: Path,
    *,
    subset_idx: np.ndarray | None,
    device: torch.device,
    legacy_val_resize: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    probs, yt, yhat, _n_cls, class_names = run_one_checkpoint(
        ck_path,
        data_root,
        device=device,
        augmentation="standard",
        img_size=224,
        batch_size=32,
        num_workers=4,
        legacy_val_resize=legacy_val_resize,
    )
    if subset_idx is not None:
        yt = yt[subset_idx]
        yhat = yhat[subset_idx]
        probs = probs[subset_idx]
    return probs, yt, yhat, class_names


def macro_metrics_row(
    excel_name: str,
    ck_name: str,
    split: str,
    yt: np.ndarray,
    yhat: np.ndarray,
    probs: np.ndarray,
    n_cls: int,
    class_names: list[str],
) -> tuple[dict, list[dict]]:
    row = row_eltra_bootstrap(
        ck_name,
        yt,
        yhat,
        probs,
        n_cls,
        n_bootstrap=N_BOOTSTRAP,
        seed=BOOTSTRAP_SEED,
    )
    row.pop("_detail", None)
    row.pop("_auc_sort", None)
    row["excel_model"] = excel_name
    row["split"] = split

    _, per_class_pt = compute_macro_classification_metrics(yt, yhat, n_classes=n_cls)
    aucs_ovr_pt = _per_class_auc_ovr(yt, probs, n_cls)
    pc_rows = bootstrap_per_class_comparison_rows(
        yt,
        yhat,
        probs,
        per_class_pt,
        aucs_ovr_pt,
        class_names,
        n_boot=N_BOOTSTRAP,
        random_state=BOOTSTRAP_SEED,
    )
    for pr in pc_rows:
        pr["excel_model"] = excel_name
        pr["split"] = split
    return row, pc_rows


def generate_plots_for_cache(
    cache_path: Path,
    excel_name: str,
    split: str,
    out_dir: Path,
) -> None:
    probs, yt, yhat, class_names = load_cache(cache_path)
    class_rows = bootstrap_per_class_auc_rows(
        yt, probs, class_names, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED
    )

    auc_csv = out_dir / f"{excel_name}_{split}_auc.csv"
    with auc_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["CLASS", "AUC"])
        w.writeheader()
        for r in class_rows:
            w.writerow({"CLASS": r["CLASS"], "AUC": r["AUC"]})

    fig_r = plot_roc(probs, yt, class_names, class_rows)
    fig_r.savefig(out_dir / f"{excel_name}_{split}_auc.png", dpi=160, bbox_inches="tight")
    plt.close(fig_r)

    fig_c = plot_confusion(yt, yhat, class_names)
    fig_c.savefig(out_dir / f"{excel_name}_{split}_confusion.png", dpi=160, bbox_inches="tight")
    plt.close(fig_c)


def evaluate_manifest(
    ck_path: Path,
    manifest: dict,
    *,
    device: torch.device,
    legacy_val_resize: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Evaluate manifest paths; supports train+test+val mixed paths."""
    manifest_paths = manifest.get("paths_relative_to_cwd") or manifest.get("paths") or []
    pool_roots = manifest.get("search_pools")
    if pool_roots:
        probs, yt, yhat, class_names, paths_all, _ = run_combined_pool_inference(
            ck_path,
            pool_roots,
            device=device,
            augmentation="standard",
            img_size=224,
            batch_size=32,
            num_workers=4,
            legacy_val_resize=legacy_val_resize,
        )
        sel_idx = paths_to_indices(paths_all, manifest_paths)
        return probs[sel_idx], yt[sel_idx], yhat[sel_idx], class_names

    splits_used = {split_tag_for_path(p) for p in manifest_paths}
    if len(splits_used) <= 1:
        data_root = Path(manifest["source_data_root"])
        probs, yt, yhat, _n_cls, class_names = run_one_checkpoint(
            ck_path,
            data_root,
            device=device,
            augmentation="standard",
            img_size=224,
            batch_size=32,
            num_workers=4,
            legacy_val_resize=legacy_val_resize,
        )
        base_ds = ImageFolderDataset(str(data_root), transform=None)
        sel_idx = manifest_paths_to_indices(manifest, data_root, base_ds.samples)
        return probs[sel_idx], yt[sel_idx], yhat[sel_idx], class_names

    root_map = {
        "train": ROOT / "old_data/train",
        "test": ROOT / "old_data/test",
        "val": ROOT / "old_data/val",
    }
    path_to_pred: dict[str, tuple[np.ndarray, int, int]] = {}
    class_names: list[str] | None = None
    for tag in sorted(splits_used):
        root = root_map.get(tag)
        if root is None or not root.is_dir():
            continue
        probs, yt, yhat, _n_cls, cnames = run_one_checkpoint(
            ck_path, root, device=device, augmentation="standard",
            img_size=224, batch_size=32, num_workers=4, legacy_val_resize=legacy_val_resize,
        )
        if class_names is None:
            class_names = cnames
        ds = ImageFolderDataset(str(root), transform=None)
        for i in range(len(ds)):
            path_to_pred[norm_path(ds.samples[i][0])] = (probs[i], int(yt[i]), int(yhat[i]))

    assert class_names is not None
    out_probs, out_yt, out_yhat = [], [], []
    for p in manifest_paths:
        pr, yti, yhi = path_to_pred[norm_path(p)]
        out_probs.append(pr)
        out_yt.append(yti)
        out_yhat.append(yhi)
    return np.stack(out_probs), np.asarray(out_yt), np.asarray(out_yhat), class_names


def load_table1_manifest(excel_name: str) -> dict:
    manifest_path = TABLE1_MANIFEST_DIR / f"{excel_name}_table1_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing per-model manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    n_sel = manifest.get("n_selected", len(manifest.get("paths_relative_to_cwd", [])))
    assert n_sel <= MAX_EVAL_N, f"Manifest for {excel_name} has n={n_sel} > MAX_EVAL_N={MAX_EVAL_N}"
    return manifest


def load_excel_targets() -> tuple[pd.DataFrame, pd.DataFrame]:
    xlsx = pd.ExcelFile(EXCEL_PATH)
    return pd.read_excel(xlsx, "测试集结果"), pd.read_excel(xlsx, "独立测试集结果")


def build_comparison_report(
    test_rows: list[dict],
    val_rows: list[dict],
    test_excel: pd.DataFrame,
    val_excel: pd.DataFrame,
) -> dict:
    report: dict = {"test": [], "val": [], "notes": []}
    test_by_excel = {r["excel_model"]: r for r in test_rows}
    val_by_excel = {r["excel_model"]: r for r in val_rows}

    for _, er in test_excel.iterrows():
        ex = er["MODEL"]
        cur = test_by_excel.get(ex)
        if not cur:
            continue
        report["test"].append(
            {
                "model": ex,
                "excel_acc": er["ACC"],
                "reproduced_acc": cur["acc"],
                "excel_auc": er["AUC"],
                "reproduced_auc": cur["auc"],
                "acc_delta": (parse_point(cur["acc"]) or 0) - (parse_point(er["ACC"]) or 0),
                "auc_delta": (parse_point(cur["auc"]) or 0) - (parse_point(er["AUC"]) or 0),
            }
        )

    for _, er in val_excel.iterrows():
        ex = er["MODEL"]
        cur = val_by_excel.get(ex)
        if not cur:
            continue
        report["val"].append(
            {
                "model": ex,
                "excel_acc": er["ACC"],
                "reproduced_acc": cur["acc"],
                "excel_auc": er["AUC"],
                "reproduced_auc": cur["auc"],
                "acc_delta": (parse_point(cur["acc"]) or 0) - (parse_point(er["ACC"]) or 0),
                "auc_delta": (parse_point(cur["auc"]) or 0) - (parse_point(er["AUC"]) or 0),
            }
        )

    report["notes"].append(
        "表一 uses v2 checkpoints + test subset (217) + legacy_val_resize; "
        "表二 uses v3 checkpoints + old_data/val + legacy_val_resize."
    )
    report["notes"].append(
        "Excel 测试集 appears partially composite (CasGNet ACC from subset, some AUC from other runs); "
        "exact match for all models may not be achievable from a single split."
    )
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-inference", action="store_true", help="Only regenerate plots from caches")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Table1: per-model manifests from {TABLE1_MANIFEST_DIR}")

    test_rows: list[dict] = []
    val_rows: list[dict] = []
    test_pc: list[dict] = []
    val_pc: list[dict] = []

    if not args.skip_inference:
        for excel_name, ck_name in EXCEL_MODELS:
            v2_ck = V2_ROOT / ck_name / "best_auc_model.pth"
            v3_ck = V3_ROOT / ck_name / "best_auc_model.pth"
            if not v2_ck.is_file():
                print(f"SKIP test {excel_name}: missing {v2_ck}", file=sys.stderr)
            else:
                print(f"\n>>> TEST [{excel_name}] {v2_ck.name}")
                manifest = load_table1_manifest(excel_name)
                n_manifest = manifest.get("n_selected", len(manifest.get("paths_relative_to_cwd", [])))
                sc = manifest.get("split_source_counts", {})
                print(
                    f"    n={n_manifest} train={manifest.get('n_train', sc.get('train', 0))} "
                    f"test={manifest.get('n_test', sc.get('test', 0))}"
                )
                probs, yt, yhat, class_names = evaluate_manifest(
                    v2_ck, manifest, device=device, legacy_val_resize=True
                )
                cache = CACHE_DIR / f"{excel_name}_test_predictions.npz"
                save_cache(cache, probs, yt, yhat, class_names)
                row, pc = macro_metrics_row(
                    excel_name, ck_name, "test", yt, yhat, probs, len(class_names), class_names
                )
                test_rows.append(row)
                test_pc.extend(pc)
                print(f"    acc={row['acc']}  auc={row['auc']}")

            if not v3_ck.is_file():
                print(f"SKIP val {excel_name}: missing {v3_ck}", file=sys.stderr)
            else:
                print(f"\n>>> VAL [{excel_name}] {v3_ck.name}")
                probs, yt, yhat, class_names = evaluate_one(
                    v3_ck, VAL_DIR, subset_idx=None, device=device, legacy_val_resize=True
                )
                cache = CACHE_DIR / f"{excel_name}_val_predictions.npz"
                save_cache(cache, probs, yt, yhat, class_names)
                row, pc = macro_metrics_row(
                    excel_name, ck_name, "val", yt, yhat, probs, len(class_names), class_names
                )
                val_rows.append(row)
                val_pc.extend(pc)
                print(f"    acc={row['acc']}  auc={row['auc']}")

        macro_fields = ["excel_model", "model", "split", "auc", "sensitivity", "specificity", "npv", "ppv", "acc"]
        for rows, tag in [(test_rows, "table1_test"), (val_rows, "table2_val")]:
            rows.sort(key=lambda r: -(parse_point(r["auc"]) or 0))
            path = METRICS_DIR / f"{tag}_macro.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=macro_fields, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)
            print(f"Wrote {path}")

        pc_fields = ["excel_model", "split", "experiment"] + PER_CLASS_COMPARISON_FIELDS
        for pc_rows, tag in [(test_pc, "table1_test"), (val_pc, "table2_val")]:
            path = METRICS_DIR / f"{tag}_per_class.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=pc_fields, extrasaction="ignore")
                w.writeheader()
                for r in pc_rows:
                    w.writerow({k: r.get(k, "") for k in pc_fields})
            print(f"Wrote {path}")

        test_excel, val_excel = load_excel_targets()
        report = build_comparison_report(test_rows, val_rows, test_excel, val_excel)
        report_path = METRICS_DIR / "excel_comparison_report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {report_path}")

    for excel_name, _ in EXCEL_MODELS:
        for split in ("test", "val"):
            cache = CACHE_DIR / f"{excel_name}_{split}_predictions.npz"
            if not cache.is_file():
                print(f"Missing cache: {cache}", file=sys.stderr)
                continue
            generate_plots_for_cache(cache, excel_name, split, PLOTS_DIR)
            print(f"Plots: {excel_name}_{split}_*.png")

    print(f"\nDone. Outputs under {HERE}")


if __name__ == "__main__":
    main()
