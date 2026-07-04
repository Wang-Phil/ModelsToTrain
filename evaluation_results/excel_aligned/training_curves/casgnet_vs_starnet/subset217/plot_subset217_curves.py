#!/usr/bin/env python3
"""Plot subset217 per-epoch AUC curves for CasGNet vs StarNet.

Generates:
  - subset217_auc_curves.png        (2 lines, y in [0.90, 0.97], final-epoch annotations)
  - subset217_auc_difference.png    (CasGNet - StarNet)
  - subset217_auc_smoothed.png      (10-epoch moving average)
  - subset217_curves_enhanced.pdf    (multi-panel: raw, smoothed, diff)

Reads per_epoch_subset217_auc.csv (columns: model, epoch, subset217_auc).
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "per_epoch_subset217_auc.csv"
Y_MIN, Y_MAX = 0.90, 0.97
SMOOTH_WINDOW = 10


def load_csv():
    by_model = defaultdict(lambda: {"epoch": [], "auc": []})
    with open(CSV_PATH, newline="") as f:
        for row in csv.DictReader(f):
            by_model[row["model"]]["epoch"].append(int(row["epoch"]))
            by_model[row["model"]]["auc"].append(float(row["subset217_auc"]))
    out = {}
    for m, d in by_model.items():
        idx = np.argsort(d["epoch"])
        out[m] = (np.array(d["epoch"])[idx], np.array(d["auc"])[idx])
    return out


def moving_avg(x, w=SMOOTH_WINDOW):
    if len(x) < w:
        return x
    ker = np.ones(w) / w
    pad = w // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xp, ker, mode="valid")[: len(x)]


def align_epochs(cas, sta):
    ec, ac = cas
    es, as_ = sta
    common = np.intersect1d(ec, es)
    acf = np.array([ac[np.where(ec == e)[0][0]] for e in common])
    asf = np.array([as_[np.where(es == e)[0][0]] for e in common])
    return common, acf, asf


def main():
    data = load_csv()
    if "casgnet" not in data or "starnet" not in data:
        print(f"[err] expected casgnet & starnet in CSV, got {list(data)}")
        return
    cas = data["casgnet"]
    sta = data["starnet"]

    # ---- raw curves ----
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(cas[0], cas[1], label="CasGNet", color="#1f77b4", linewidth=1.6)
    ax.plot(sta[0], sta[1], label="StarNet", color="#d62728", linewidth=1.6)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("subset217 macro AUC (OVR)")
    ax.set_title("Per-epoch AUC on subset217 (n=230): CasGNet vs StarNet")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    # final-epoch annotations
    for lbl, (e, a) in [("CasGNet", cas), ("StarNet", sta)]:
        if len(a):
            af = a[-1]
            ax.annotate(f"{lbl} ep{e[-1]}: {af:.3f}",
                        xy=(e[-1], af), xytext=(8, 8 if lbl == "CasGNet" else -14),
                        textcoords="offset points", fontsize=9,
                        color="#1f77b4" if lbl == "CasGNet" else "#d62728")
    fig.tight_layout()
    p1 = HERE / "subset217_auc_curves.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"[wrote] {p1}")

    # ---- difference curve ----
    common, acf, asf = align_epochs(cas, sta)
    diff = acf - asf
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(common, diff, color="#2ca02c", linewidth=1.6)
    ax.axhline(0, color="k", linewidth=0.8, alpha=0.5)
    ax.axhline(0.02, color="r", linestyle="--", linewidth=0.8, alpha=0.6, label="~0.02 gap")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CasGNet - StarNet AUC")
    ax.set_title("Per-epoch AUC difference on subset217 (CasGNet - StarNet)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    p2 = HERE / "subset217_auc_difference.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"[wrote] {p2}")

    # ---- smoothed ----
    fig, ax = plt.subplots(figsize=(10, 6))
    cas_s = moving_avg(cas[1])
    sta_s = moving_avg(sta[1])
    ax.plot(cas[0], cas_s, label=f"CasGNet (MA{SMOOTH_WINDOW})", color="#1f77b4", linewidth=2)
    ax.plot(sta[0], sta_s, label=f"StarNet (MA{SMOOTH_WINDOW})", color="#d62728", linewidth=2)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"subset217 AUC ({SMOOTH_WINDOW}-epoch moving avg)")
    ax.set_title("Smoothed per-epoch AUC on subset217: CasGNet vs StarNet")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    p3 = HERE / "subset217_auc_smoothed.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    print(f"[wrote] {p3}")

    # ---- enhanced multi-panel PDF ----
    fig, axes = plt.subplots(3, 1, figsize=(11, 13))
    ax = axes[0]
    ax.plot(cas[0], cas[1], label="CasGNet", color="#1f77b4", linewidth=1.4)
    ax.plot(sta[0], sta[1], label="StarNet", color="#d62728", linewidth=1.4)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_ylabel("AUC (raw)")
    ax.set_title("Raw per-epoch AUC on subset217")
    ax.grid(True, alpha=0.3); ax.legend()

    ax = axes[1]
    ax.plot(cas[0], cas_s, label=f"CasGNet MA{SMOOTH_WINDOW}", color="#1f77b4", linewidth=2)
    ax.plot(sta[0], sta_s, label=f"StarNet MA{SMOOTH_WINDOW}", color="#d62728", linewidth=2)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_ylabel("AUC (smoothed)")
    ax.set_title(f"{SMOOTH_WINDOW}-epoch moving average")
    ax.grid(True, alpha=0.3); ax.legend()

    ax = axes[2]
    diff_s = moving_avg(diff)
    ax.plot(common, diff, color="#2ca02c", linewidth=1.0, alpha=0.5, label="raw diff")
    ax.plot(common, diff_s, color="#2ca02c", linewidth=2, label=f"diff MA{SMOOTH_WINDOW}")
    ax.axhline(0, color="k", linewidth=0.8, alpha=0.5)
    ax.axhline(0.02, color="r", linestyle="--", linewidth=0.8, alpha=0.6, label="~0.02 gap")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CasGNet - StarNet")
    ax.set_title("AUC difference on subset217")
    ax.grid(True, alpha=0.3); ax.legend()

    fig.suptitle("subset217 per-epoch AUC: CasGNet vs StarNet (retrain + per-epoch eval)", y=0.995)
    fig.tight_layout()
    p4 = HERE / "subset217_curves_enhanced.pdf"
    fig.savefig(p4)
    plt.close(fig)
    print(f"[wrote] {p4}")

    # ---- summary stats ----
    print("\n=== summary ===")
    for lbl, (e, a) in [("casgnet", cas), ("starnet", sta)]:
        print(f"{lbl:8s} n_epochs={len(a)} final={a[-1]:.4f} max={a.max():.4f} "
              f"mean(last20)={a[-20:].mean():.4f}")
    common, acf, asf = align_epochs(cas, sta)
    diff = acf - asf
    print(f"diff       mean={diff.mean():.4f} mean(last20)={diff[-20:].mean():.4f} "
          f"final={diff[-1]:.4f}")


if __name__ == "__main__":
    main()
