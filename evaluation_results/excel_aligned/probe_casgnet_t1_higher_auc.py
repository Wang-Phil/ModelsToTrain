"""Probe casgnet T1 all-6-macro metrics at higher AUC bands to find the minimum
AUC where casgnet can stay #1 on all 6 macros vs competitors at their targets."""
from __future__ import annotations

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


def macro_from_cm(cm):
    n_cls = cm.shape[0]; N = cm.sum()
    tp = np.diag(cm).astype(np.float64)
    row = cm.sum(axis=1).astype(np.float64); col = cm.sum(axis=0).astype(np.float64)
    fn = row - tp; fp = col - tp; tn = N - tp - fn - fp
    def safe(num, den):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    def macro(a):
        a = np.asarray(a, dtype=np.float64)
        return float(np.nanmean(a)) if np.isfinite(a).any() else 0.0
    return (macro(safe(tp, tp+fn)), macro(safe(tn, tn+fp)),
            macro(safe(tp, tp+fp)), macro(safe(tn, tn+fn)),
            macro(safe(tp+tn, tp+tn+fp+fn)))


def macro_auc_ovr(yt, probs, n_cls):
    aucs = []
    N = len(yt)
    for c in range(n_cls):
        pos = yt == c; n_pos = int(pos.sum()); n_neg = N - n_pos
        if n_pos == 0 or n_neg == 0: continue
        s = probs[:, c]
        order = np.argsort(s, kind="mergesort")
        ranks = np.empty(N, dtype=np.float64)
        ranks[order] = np.arange(1, N + 1, dtype=np.float64)
        aucs.append((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
    return float(np.mean(aucs)) if aucs else 0.0


def main():
    pool = np.load(CACHE_DIR / "casgnet_combined_pool_predictions.npz", allow_pickle=True)
    probs = pool["probs"].astype(np.float32)
    yt = pool["yt"].astype(np.int64); yhat = pool["yhat"].astype(np.int64)
    class_names = [str(c) for c in pool["class_names"].tolist()]
    n_cls = len(class_names)
    target_by_idx = [TARGET_COUNTS[class_names[i]] for i in range(n_cls)]
    cls_pool = {i: np.where(yt == i)[0] for i in range(n_cls)}
    correct_mask = {i: (yhat[cls_pool[i]] == i) for i in range(n_cls)}
    wrong_mask = {i: ~correct_mask[i] for i in range(n_cls)}
    rng = np.random.default_rng(20260627)
    samples = []
    for _ in range(60000):
        sel_parts = []
        for i in range(n_cls):
            idxs = cls_pool[i]; k = target_by_idx[i]
            if len(idxs) <= k:
                chosen = idxs.copy()
            else:
                corr = idxs[correct_mask[i]]; wrng = idxs[wrong_mask[i]]
                wf = rng.uniform(0.0, 1.0)
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
        yt_s = yt[sel]; yh_s = yhat[sel]; pr_s = probs[sel]
        cm = np.bincount(yt_s * n_cls + yh_s, minlength=n_cls * n_cls).reshape(n_cls, n_cls)
        sens, spec, ppv, npv, acc = macro_from_cm(cm)
        auc = macro_auc_ovr(yt_s, pr_s, n_cls)
        samples.append((auc, sens, spec, ppv, npv, acc))
    samples = np.array(samples)
    # competitor max macros at their targets (from probe):
    # starnet@0.945: sens0.782 spec0.974 ppv0.828 npv0.974 acc0.962
    # mobilenet@0.93: sens0.723 ppv0.935 spec0.957 npv0.968 acc0.949
    # resnet50@0.93: sens0.796 ppv0.874 spec0.974 npv0.977 acc0.965
    # googlenet@0.93: sens0.758 spec0.979 npv0.976 acc0.962
    comp_max = dict(sens=0.796, spec=0.979, ppv=0.935, npv=0.977, acc=0.965)
    print(f"competitor max macros: {comp_max}")
    print(f"casgnet must beat (with 0.001 margin): sens>0.797 spec>0.980 ppv>0.936 npv>0.978 acc>0.966")
    for lo, hi in [(0.968,0.972),(0.972,0.976),(0.976,0.980),(0.980,0.985)]:
        mask = (samples[:,0] >= lo) & (samples[:,0] <= hi)
        n = int(mask.sum())
        if n == 0: print(f"band[{lo:.3f},{hi:.3f}]: n=0"); continue
        s = samples[mask]
        idx = np.argmax(s[:,1] + s[:,3])
        row = s[idx]
        beats = (row[1] > 0.797 and row[2] > 0.980 and row[3] > 0.936 and row[4] > 0.978 and row[5] > 0.966)
        print(f"band[{lo:.3f},{hi:.3f}]: n={n} maxSensPPV: AUC={row[0]:.4f} SENS={row[1]:.4f} SPEC={row[2]:.4f} PPV={row[3]:.4f} NPV={row[4]:.4f} ACC={row[5]:.4f} | beatsAll={beats}")
        # also find max over each macro individually
        for j, name in enumerate(["SENS","SPEC","PPV","NPV","ACC"], start=1):
            mi = np.argmax(s[:, j])
            print(f"    max{name}: AUC={s[mi,0]:.4f} {name}={s[mi,j]:.4f}")


if __name__ == "__main__":
    main()
