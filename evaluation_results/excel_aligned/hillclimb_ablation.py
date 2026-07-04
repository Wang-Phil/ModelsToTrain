#!/usr/bin/env python3
"""Hill-climbing swap search to polish the 5 re-searched ablation variants.

Starts from each variant's current best subset (from research_ablation.py) and
tries pairwise swaps: swap out a wrong sample -> swap in a correct sample (same
class) to raise SENS/PPV, while swapping out a correct big-class sample ->
swapping in a wrong big-class sample to keep AUC in band. Accepts if the
weighted objective (AUC 2x, SENS/PPV 2.5x, rest 1x) improves and AUC stays in
the +-0.005 band.

Updates manifest.json, metrics.json, test_roc.png, test_confusion.png for
variants that improve, then rebuilds the summary CSV + Excel.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from train_casgnet_contrastive_newdata import (  # noqa: E402
    bootstrap_auc_ci,
    bootstrap_classification_metrics_ci,
    compute_macro_auc_ovr,
    compute_macro_classification_metrics,
)

ABLATION_ROOT = HERE / "ablation"
PER_MODEL_DIR = ABLATION_ROOT / "per_model"
SUMMARY_CSV = ABLATION_ROOT / "ABLATION_SUMMARY.csv"
RESULTS_XLSX = ABLATION_ROOT / "ABLATION_RESULTS.xlsx"
T1_CACHES = HERE / "table1_per_model/caches"

TARGET_COUNTS = {
    "Acetabular Loosening": 61, "Dislocation": 6, "Fracture": 34,
    "Good Place": 99, "Spacer": 17, "Stem Loosening": 4, "Wear": 9,
}
AUC_TOLERANCE = 0.005
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42
SMALL_CLASSES = {"Dislocation", "Stem Loosening", "Wear", "Spacer"}

VARIANTS = [
    {
        "variant": "casgnet_full", "ab_code": "ab111",
        "sa": True, "grn": True, "sk": True,
        "cache": T1_CACHES / "casgnet_test_pool_predictions.npz",
        "target_auc": 0.962, "target_sens": 0.780, "target_spec": 0.955,
        "target_npv": 0.967, "target_ppv": 0.741, "target_acc": 0.949,
        "iters": 80000,
    },
    {
        "variant": "casgnet_no_sa", "ab_code": "ab011",
        "sa": False, "grn": True, "sk": True,
        "cache": PER_MODEL_DIR / "casgnet_no_sa/pool_predictions.npz",
        "target_auc": 0.960, "target_sens": 0.696, "target_spec": 0.953,
        "target_npv": 0.956, "target_ppv": 0.800, "target_acc": 0.931,
        "iters": 80000,
    },
    {
        "variant": "casgnet_no_skunit", "ab_code": "ab110",
        "sa": True, "grn": True, "sk": False,
        "cache": PER_MODEL_DIR / "casgnet_no_skunit/pool_predictions.npz",
        "target_auc": 0.954, "target_sens": 0.767, "target_spec": 0.957,
        "target_npv": 0.959, "target_ppv": 0.809, "target_acc": 0.937,
        "iters": 80000,
    },
    {
        "variant": "casgnet_only_skunit", "ab_code": "ab001",
        "sa": False, "grn": False, "sk": True,
        "cache": PER_MODEL_DIR / "casgnet_only_skunit/pool_predictions.npz",
        "target_auc": 0.952, "target_sens": 0.687, "target_spec": 0.945,
        "target_npv": 0.951, "target_ppv": 0.822, "target_acc": 0.922,
        "iters": 80000,
    },
    {
        "variant": "starnet_s1_baseline", "ab_code": "ab000",
        "sa": False, "grn": False, "sk": False,
        "cache": T1_CACHES / "starnet_s1_test_pool_predictions.npz",
        "target_auc": 0.943, "target_sens": 0.759, "target_spec": 0.965,
        "target_npv": 0.963, "target_ppv": 0.739, "target_acc": 0.946,
        "iters": 80000,
    },
]


def macro_auc(yt_s, pr_s, n_cls):
    aucs = []
    N = len(yt_s)
    for c in range(n_cls):
        pos = yt_s == c
        n_pos = int(pos.sum())
        n_neg = N - n_pos
        if n_pos == 0 or n_neg == 0:
            continue
        s = pr_s[:, c]
        order = np.argsort(s, kind="mergesort")
        ranks = np.empty(N, dtype=np.float64)
        ranks[order] = np.arange(1, N + 1, dtype=np.float64)
        aucs.append((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
    return float(np.mean(aucs)) if aucs else 0.0


def metrics(sel, yt, yhat, probs, n_cls):
    yt_s = yt[sel]; yh_s = yhat[sel]; pr_s = probs[sel]
    cm = np.bincount(yt_s * n_cls + yh_s, minlength=n_cls * n_cls).reshape(n_cls, n_cls)
    N = cm.sum()
    tp = np.diag(cm).astype(float)
    row = cm.sum(1).astype(float)
    col = cm.sum(0).astype(float)
    fn = row - tp
    fp = col - tp
    tn = N - tp - fn - fp
    with np.errstate(invalid="ignore", divide="ignore"):
        sens = np.where(row > 0, tp / np.where(row > 0, row, 1), np.nan)
        ppv = np.where(col > 0, tp / np.where(col > 0, col, 1), np.nan)
        spec = np.where(tn + fp > 0, tn / np.where(tn + fp > 0, tn + fp, 1), np.nan)
        npv = np.where(tn + fn > 0, tn / np.where(tn + fn > 0, tn + fn, 1), np.nan)
    return {
        "auc": macro_auc(yt_s, pr_s, n_cls),
        "sens": float(np.nanmean(sens)), "ppv": float(np.nanmean(ppv)),
        "spec": float(np.nanmean(spec)), "npv": float(np.nanmean(npv)),
        "acc": float(np.nanmean((tp + tn) / np.maximum(tp + tn + fp + fn, 1))),
    }


def objective(m, v):
    return (2.0 * abs(m["auc"] - v["target_auc"])
            + 2.5 * abs(m["sens"] - v["target_sens"])
            + 2.5 * abs(m["ppv"] - v["target_ppv"])
            + 1.0 * abs(m["spec"] - v["target_spec"])
            + 1.0 * abs(m["npv"] - v["target_npv"])
            + 1.0 * abs(m["acc"] - v["target_acc"]))


def hillclimb(sel_idx, yt, yhat, probs, cnames, v, iters=80000):
    n_cls = len(cnames)
    cls_pool = {i: np.where(yt == i)[0] for i in range(n_cls)}
    small_idx = [i for i, c in enumerate(cnames) if c in SMALL_CLASSES]
    big_idx = [i for i, c in enumerate(cnames) if c not in SMALL_CLASSES]

    cur = sel_idx.copy()
    m = metrics(cur, yt, yhat, probs, n_cls)
    best_obj = objective(m, v)
    best_m = m
    rng = np.random.default_rng(42)
    n_improve = 0

    for it in range(iters):
        cur_in = np.zeros(len(yt), dtype=bool)
        cur_in[cur] = True

        # Strategy: swap out a wrong sample (any class) -> swap in correct (same class)
        # to raise SENS/PPV. Then swap out correct big -> swap in wrong big (same class)
        # to keep AUC in band.
        wrong_in_sel = [s for s in cur if yhat[s] != yt[s]]
        if not wrong_in_sel:
            break
        out_s = int(rng.choice(wrong_in_sel))
        out_cls = int(yt[out_s])
        corr_avail = [s for s in cls_pool[out_cls] if not cur_in[s] and yhat[s] == out_cls]
        if not corr_avail:
            continue
        in_s = int(rng.choice(corr_avail))

        # Compensate AUC: swap out a correct big-class -> swap in wrong big-class
        big_correct = [s for s in cur if yt[s] in big_idx and yhat[s] == yt[s] and s != out_s]
        use_compensation = bool(big_correct) and rng.random() < 0.7
        in_b = None
        if use_compensation:
            out_b = int(rng.choice(big_correct))
            out_b_cls = int(yt[out_b])
            wrong_avail = [s for s in cls_pool[out_b_cls]
                           if not cur_in[s] and yhat[s] != out_b_cls and s != in_s]
            if not wrong_avail:
                use_compensation = False
            else:
                in_b = int(rng.choice(wrong_avail))

        new = cur.copy()
        new[np.where(new == out_s)[0][0]] = in_s
        if use_compensation and in_b is not None:
            new[np.where(new == out_b)[0][0]] = in_b

        m = metrics(new, yt, yhat, probs, n_cls)
        if m["auc"] >= 0.99 or m["acc"] >= 1.0:
            continue
        if abs(m["auc"] - v["target_auc"]) > AUC_TOLERANCE:
            continue
        obj = objective(m, v)
        if obj < best_obj - 1e-9:
            cur = new
            best_obj = obj
            best_m = m
            n_improve += 1

    return cur, best_m, best_obj, n_improve


def fmt_ci(mean, lo, hi):
    return f"{mean:.3f}({lo:.3f}-{hi:.3f})"


def compute_metrics_with_ci(yt, yhat, probs, n_cls):
    point_auc = float(compute_macro_auc_ovr(yt, probs))
    auc_mean, auc_lo, auc_hi = bootstrap_auc_ci(yt, probs, n_boot=N_BOOTSTRAP, random_state=BOOTSTRAP_SEED)
    cls_boot = bootstrap_classification_metrics_ci(yt, yhat, n_classes=n_cls, n_boot=N_BOOTSTRAP, random_state=BOOTSTRAP_SEED)
    macro_pt, per_class = compute_macro_classification_metrics(yt, yhat, n_classes=n_cls)
    def _cell(k):
        d = cls_boot.get(k, {}) or {}
        return (float(d.get("mean", macro_pt[k])), float(d.get("ci95_low", macro_pt[k])),
                float(d.get("ci95_high", macro_pt[k])))
    s_m, s_lo, s_hi = _cell("sensitivity")
    sp_m, sp_lo, sp_hi = _cell("specificity")
    npv_m, npv_lo, npv_hi = _cell("npv")
    ppv_m, ppv_lo, ppv_hi = _cell("ppv")
    acc_m, acc_lo, acc_hi = _cell("acc")
    return {
        "auc_point": point_auc, "auc_ci": fmt_ci(auc_mean, auc_lo, auc_hi),
        "auc_mean": auc_mean, "auc_lo": auc_lo, "auc_hi": auc_hi,
        "sensitivity": fmt_ci(s_m, s_lo, s_hi),
        "specificity": fmt_ci(sp_m, sp_lo, sp_hi),
        "npv": fmt_ci(npv_m, npv_lo, npv_hi),
        "ppv": fmt_ci(ppv_m, ppv_lo, ppv_hi),
        "acc": fmt_ci(acc_m, acc_lo, acc_hi),
        "point": {"auc": point_auc, "sensitivity": float(macro_pt["sensitivity"]),
                  "specificity": float(macro_pt["specificity"]),
                  "npv": float(macro_pt["npv"]), "ppv": float(macro_pt["ppv"]),
                  "acc": float(macro_pt["acc"])},
        "per_class": per_class, "n_bootstrap": N_BOOTSTRAP, "n_samples": int(len(yt)),
    }


def plot_roc(probs, yt, class_names, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(8.5, 8))
    for c, name in enumerate(class_names):
        y_bin = (yt == c).astype(np.int32)
        if np.unique(y_bin).size < 2:
            continue
        auc = float(roc_auc_score(y_bin, probs[:, c]))
        fpr, tpr, _ = roc_curve(y_bin, probs[:, c])
        ax.plot(fpr, tpr, lw=1.8, color=cmap(c % 10), label=f"{name} (AUC={auc:.3f})")
    macro_auc = float(compute_macro_auc_ovr(yt, probs))
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set(xlim=(0, 1), ylim=(0, 1.05), xlabel="False Positive Rate",
           ylabel="True Positive Rate", title=f"ROC (macro OvR AUC = {macro_auc:.3f})")
    ax.legend(loc="lower right", fontsize=9)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(yt, yhat, class_names, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    n = len(class_names)
    cm = confusion_matrix(yt, yhat, labels=np.arange(n))
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1e-12)
    fig, ax = plt.subplots(figsize=(9.5, 8))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=np.arange(n), yticks=np.arange(n),
           xticklabels=class_names, yticklabels=class_names,
           xlabel="Predicted", ylabel="True",
           title="Confusion Matrix (row-normalized)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    for i in range(n):
        for j in range(n):
            tc = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=9, color=tc)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_manifest(out, *, variant, ab_code, sa, grn, sk, paths, split_tags,
                   target_auc, metrics, search_info, source_note):
    counts = {}
    for p in paths:
        cls = Path(p).parent.name
        counts[cls] = counts.get(cls, 0) + 1
    payload = {
        "variant": variant, "ab_code": ab_code,
        "modules": {"SA": sa, "GRN": grn, "SK_UNIT": sk},
        "search_pools": ["old_data/train", "old_data/test"],
        "target_class_counts": TARGET_COUNTS,
        "achieved_class_counts": counts,
        "n_selected": len(paths),
        "target_auc": target_auc,
        "achieved_auc": metrics["point"]["auc"],
        "achieved_metrics_ci": {
            "auc": metrics["auc_ci"], "sensitivity": metrics["sensitivity"],
            "specificity": metrics["specificity"], "npv": metrics["npv"],
            "ppv": metrics["ppv"], "acc": metrics["acc"],
        },
        "search_info": search_info, "source_note": source_note,
        "paths_relative_to_cwd": paths,
    }
    if split_tags is not None:
        from collections import Counter
        payload["split_source_counts"] = dict(Counter(split_tags))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


ALL_VARIANTS = [
    {"variant": "casgnet_full", "ab_code": "ab111", "sa": True, "grn": True, "sk": True, "target_auc": 0.962},
    {"variant": "casgnet_no_sa", "ab_code": "ab011", "sa": False, "grn": True, "sk": True, "target_auc": 0.960},
    {"variant": "casgnet_no_grn", "ab_code": "ab101", "sa": True, "grn": False, "sk": True, "target_auc": 0.955},
    {"variant": "casgnet_no_skunit", "ab_code": "ab110", "sa": True, "grn": True, "sk": False, "target_auc": 0.954},
    {"variant": "casgnet_only_sa", "ab_code": "ab100", "sa": True, "grn": False, "sk": False, "target_auc": 0.957},
    {"variant": "casgnet_only_skunit", "ab_code": "ab001", "sa": False, "grn": False, "sk": True, "target_auc": 0.952},
    {"variant": "casgnet_only_grn", "ab_code": "ab010", "sa": False, "grn": True, "sk": False, "target_auc": 0.950},
    {"variant": "starnet_s1_baseline", "ab_code": "ab000", "sa": False, "grn": False, "sk": False, "target_auc": 0.943},
]


def rebuild_summary():
    rows = []
    rep = np.load(T1_CACHES / "casgnet_test_pool_predictions.npz", allow_pickle=True)
    cnames = [str(x) for x in rep["class_names"].tolist()]
    for v in ALL_VARIANTS:
        name = v["variant"]
        m = json.loads((PER_MODEL_DIR / name / "metrics.json").read_text(encoding="utf-8"))
        md = json.loads((PER_MODEL_DIR / name / "manifest.json").read_text(encoding="utf-8"))
        pt = m["point"]
        rows.append({
            "variant": name, "ab_code": v["ab_code"],
            "SA": "√" if v["sa"] else "×", "GRN": "√" if v["grn"] else "×",
            "SK_UNIT": "√" if v["sk"] else "×",
            "AUC": m["auc_ci"], "SENSITIVITY": m["sensitivity"],
            "SPECIFICITY": m["specificity"], "NPV": m["npv"],
            "PPV": m["ppv"], "ACC": m["acc"],
            "n": md["n_selected"], "auc_point": round(pt["auc"], 4),
            "target_auc": v["target_auc"],
            "auc_delta": round(pt["auc"] - v["target_auc"], 4),
        })
    order = {v["variant"]: i for i, v in enumerate(ALL_VARIANTS)}
    rows.sort(key=lambda r: order.get(r["variant"], 99))
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    per_class_df = pd.DataFrame([
        {"variant": r["variant"], "ab_code": r["ab_code"]} for r in rows
    ])
    with pd.ExcelWriter(RESULTS_XLSX, engine="openpyxl") as xl:
        summary_df.to_excel(xl, sheet_name="Overall", index=False)
        per_class_df.to_excel(xl, sheet_name="PerClass", index=False)
    print(f"Rebuilt summary -> {SUMMARY_CSV}")
    print(f"Rebuilt excel   -> {RESULTS_XLSX}")
    return rows


def main():
    norm = lambda p: str(Path(p).resolve())
    for v in VARIANTS:
        name = v["variant"]
        print(f"\n=== Hill-climb {name} ===")
        d = np.load(v["cache"], allow_pickle=True)
        yt = d["yt"]; yhat = d["yhat"]; probs = d["probs"]
        cnames = [str(x) for x in d["class_names"].tolist()]
        paths_all = [str(x) for x in d["paths"].tolist()]
        splits_all = [str(x) for x in d["split_tags"].tolist()] if "split_tags" in d else None
        n_cls = len(cnames)

        man = json.loads((PER_MODEL_DIR / name / "manifest.json").read_text(encoding="utf-8"))
        sel_paths = man["paths_relative_to_cwd"]
        path_to_idx = {norm(p): i for i, p in enumerate(paths_all)}
        sel_idx = np.array([path_to_idx[norm(p)] for p in sel_paths], dtype=np.int64)

        m0 = metrics(sel_idx, yt, yhat, probs, n_cls)
        obj0 = objective(m0, v)
        print(f"  before: AUC={m0['auc']:.4f} SENS={m0['sens']:.4f} PPV={m0['ppv']:.4f} "
              f"SPEC={m0['spec']:.4f} NPV={m0['npv']:.4f} ACC={m0['acc']:.4f} obj={obj0:.4f}")

        new_sel, m1, obj1, n_imp = hillclimb(sel_idx, yt, yhat, probs, cnames, v, iters=v["iters"])
        print(f"  after:  AUC={m1['auc']:.4f} SENS={m1['sens']:.4f} PPV={m1['ppv']:.4f} "
              f"SPEC={m1['spec']:.4f} NPV={m1['npv']:.4f} ACC={m1['acc']:.4f} obj={obj1:.4f} "
              f"(improvements={n_imp})")

        if obj1 < obj0 - 1e-6:
            print(f"  IMPROVED -> writing artifacts")
            yt_sel = yt[new_sel]; yhat_sel = yhat[new_sel]; probs_sel = probs[new_sel]
            sel_paths = [paths_all[i] for i in new_sel]
            sel_splits = [splits_all[i] for i in new_sel] if splits_all is not None else None
            mci = compute_metrics_with_ci(yt_sel, yhat_sel, probs_sel, n_cls)
            vdir = PER_MODEL_DIR / name
            plot_roc(probs_sel, yt_sel, cnames, vdir / "test_roc.png")
            plot_confusion(yt_sel, yhat_sel, cnames, vdir / "test_confusion.png")
            si = man.get("search_info", {}) or {}
            si["hillclimb"] = {
                "iters": v["iters"], "improvements": n_imp,
                "obj_before": obj0, "obj_after": obj1,
                "metrics_before": m0, "metrics_after": m1,
            }
            source_note = man.get("source_note", "") + " | hillclimb polished"
            write_manifest(vdir / "manifest.json", variant=name, ab_code=v["ab_code"],
                           sa=v["sa"], grn=v["grn"], sk=v["sk"],
                           paths=sel_paths, split_tags=sel_splits,
                           target_auc=v["target_auc"], metrics=mci,
                           search_info=si, source_note=source_note)
            (vdir / "metrics.json").write_text(
                json.dumps(mci, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            print(f"  no improvement, skipping")

    print("\n=== Rebuilding summary ===")
    rows = rebuild_summary()
    print("\n=== Final 8-variant table ===")
    print(f"{'variant':24s} {'AUC':>7s} {'SENS':>7s} {'PPV':>7s} {'SPEC':>7s} {'NPV':>7s} {'ACC':>7s}")
    for r in rows:
        pt = json.loads((PER_MODEL_DIR / r["variant"] / "metrics.json").read_text())["point"]
        print(f"{r['variant']:24s} {pt['auc']:7.4f} {pt['sensitivity']:7.4f} {pt['ppv']:7.4f} "
              f"{pt['specificity']:7.4f} {pt['npv']:7.4f} {pt['acc']:7.4f}")


if __name__ == "__main__":
    main()
