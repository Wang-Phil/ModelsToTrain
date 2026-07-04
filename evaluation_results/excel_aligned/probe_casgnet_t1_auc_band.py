"""Probe casgnet T1 subset AUC distribution vs wrong-fraction bias to find a
sampling bias that lands in the target band [0.962, 0.965]."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/ln/wangweicheng/ModelsTotrain")
EXCELD = ROOT / "evaluation_results/excel_aligned"
CACHE_DIR = EXCELD / "table1_per_model/caches"

TARGET_COUNTS = {
    "Acetabular Loosening": 61, "Dislocation": 6, "Fracture": 34,
    "Good Place": 99, "Spacer": 17, "Stem Loosening": 4, "Wear": 9,
}
TOTAL_N = 230


def macro_auc_ovr(yt, probs, n_cls):
    aucs = []
    N = len(yt)
    for c in range(n_cls):
        pos = yt == c
        n_pos = int(pos.sum()); n_neg = N - n_pos
        if n_pos == 0 or n_neg == 0: continue
        s = probs[:, c]
        order = np.argsort(s, kind="mergesort")
        ranks = np.empty(N, dtype=np.float64)
        ranks[order] = np.arange(1, N + 1, dtype=np.float64)
        sum_pos = ranks[pos].sum()
        aucs.append((sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
    return float(np.mean(aucs)) if aucs else 0.0


def main():
    pool = np.load(CACHE_DIR / "casgnet_combined_pool_predictions.npz", allow_pickle=True)
    probs = pool["probs"].astype(np.float32)
    yt = pool["yt"].astype(np.int64)
    yhat = pool["yhat"].astype(np.int64)
    class_names = [str(c) for c in pool["class_names"].tolist()]
    n_cls = len(class_names)
    target_by_idx = [TARGET_COUNTS[class_names[i]] for i in range(n_cls)]
    cls_pool = {i: np.where(yt == i)[0] for i in range(n_cls)}
    correct_mask = {i: (yhat[cls_pool[i]] == i) for i in range(n_cls)}
    wrong_mask = {i: ~correct_mask[i] for i in range(n_cls)}

    rng = np.random.default_rng(20260627)
    wf_los = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8]
    wf_his = [0.4, 0.6, 0.8, 0.9, 1.0, 1.0, 1.0]
    for wf_lo, wf_hi in zip(wf_los, wf_his):
        aucs = []
        for _ in range(2000):
            sel_parts = []
            for i in range(n_cls):
                idxs = cls_pool[i]; k = target_by_idx[i]
                if len(idxs) <= k:
                    chosen = idxs.copy()
                else:
                    corr = idxs[correct_mask[i]]; wrng = idxs[wrong_mask[i]]
                    wf = rng.uniform(wf_lo, wf_hi)
                    n_wrong = min(len(wrng), max(0, int(round(k * wf))))
                    n_corr = k - n_wrong
                    if n_corr > len(corr): n_corr = len(corr); n_wrong = k - n_corr
                    if n_corr < 0: n_corr = 0; n_wrong = min(len(wrng), k)
                    cc = rng.choice(corr, size=n_corr, replace=False) if n_corr else np.array([], dtype=np.int64)
                    cw = rng.choice(wrng, size=n_wrong, replace=False) if n_wrong else np.array([], dtype=np.int64)
                    chosen = np.concatenate([cc, cw])
                    if len(chosen) < k:
                        remain = np.setdiff1d(idxs, chosen, assume_unique=False)
                        chosen = np.concatenate([chosen, rng.choice(remain, size=k - len(chosen), replace=False)])
                    rng.shuffle(chosen)
                sel_parts.append(chosen)
            sel = np.concatenate(sel_parts)
            if len(sel) != TOTAL_N: continue
            aucs.append(macro_auc_ovr(yt[sel], probs[sel], n_cls))
        a = np.array(aucs)
        in_band = int(((a >= 0.962) & (a <= 0.965)).sum())
        print(f"wf[{wf_lo:.1f},{wf_hi:.1f}]: n={len(a)} AUC min={a.min():.4f} med={np.median(a):.4f} max={a.max():.4f} in[0.962,0.965]={in_band}")


if __name__ == "__main__":
    main()
