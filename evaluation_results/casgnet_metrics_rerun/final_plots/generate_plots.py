#!/usr/bin/env python3
"""
从本目录 predictions.npz 生成 4 张 ROC / 混淆矩阵图，并导出各类 AUC（含 95% CI）。

依赖: numpy, scikit-learn, matplotlib
用法:
  python generate_plots.py

文件:
  表一 test:     test_predictions.npz (old_data/test subset, n=217)
                  -> test_auc.png, test_confusion.png, test_auc.csv
  表二 eltra_test: eltra_test_predictions.npz -> eltra_test_auc.png, eltra_test_confusion.png, eltra_test_auc.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc as curve_auc, confusion_matrix, roc_auc_score, roc_curve

HERE = Path(__file__).resolve().parent
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42
CONFIDENCE = 0.95

PLOTS = [
    {
        "label": "表一 test",
        "cache": "test_predictions.npz",
        "auc_png": "test_auc.png",
        "confusion_png": "test_confusion.png",
        "auc_csv": "test_auc.csv",
    },
    {
        "label": "表二 eltra_test",
        "cache": "eltra_test_predictions.npz",
        "auc_png": "eltra_test_auc.png",
        "confusion_png": "eltra_test_confusion.png",
        "auc_csv": "eltra_test_auc.csv",
    },
]


def fmt_ci(point_v: float, lo_v: float, hi_v: float) -> str:
    """点估计（三位小数四舍五入）+ bootstrap 95% CI。"""
    return f"{point_v:.3f}({lo_v:.3f}-{hi_v:.3f})"


def _finalize_boot(samples: list[float], point: float | None, p_lo: float, p_hi: float) -> tuple[float, float, float]:
    arr = np.asarray(samples, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        p = float(point) if point is not None and np.isfinite(point) else float("nan")
        return p, p, p
    return float(np.mean(arr)), float(np.percentile(arr, p_lo * 100)), float(np.percentile(arr, p_hi * 100))


def per_class_auc_point(yt: np.ndarray, probs: np.ndarray, n_classes: int) -> list[float | None]:
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


def write_auc_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["CLASS", "AUC"])
        w.writeheader()
        for r in rows:
            w.writerow({"CLASS": r["CLASS"], "AUC": r["AUC"]})


def plot_roc(
    probs: np.ndarray,
    yt: np.ndarray,
    class_names: list[str],
    class_rows: list[dict],
) -> plt.Figure:
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


def load_cache(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"缺少缓存文件: {path.name}")
    data = np.load(path, allow_pickle=True)
    class_names = [str(x) for x in data["class_names"].tolist()]
    return data["probs"], data["yt"], data["yhat"], class_names


def main() -> None:
    for spec in PLOTS:
        probs, yt, yhat, class_names = load_cache(HERE / spec["cache"])
        class_rows = bootstrap_per_class_auc_rows(
            yt, probs, class_names, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED
        )

        csv_path = HERE / spec["auc_csv"]
        write_auc_csv(csv_path, class_rows)

        auc_path = HERE / spec["auc_png"]
        fig_r = plot_roc(probs, yt, class_names, class_rows)
        fig_r.savefig(auc_path, dpi=160, bbox_inches="tight")
        plt.close(fig_r)

        cm_path = HERE / spec["confusion_png"]
        fig_c = plot_confusion(yt, yhat, class_names)
        fig_c.savefig(cm_path, dpi=160, bbox_inches="tight")
        plt.close(fig_c)


if __name__ == "__main__":
    main()
