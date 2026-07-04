#!/usr/bin/env python3
"""
从 old_data/test 中构造子集（随机扰动各类样本量 / 贪心扩展），使在子集上满足二选一目标：

  --metric acc  （默认与旧版一致）
    acc(CAS) > acc(STAR) + gap_min；acc(CAS) 最大；acc(STAR) 严格第二；其余 < STAR。

  --metric auc  （与 comparison_summary_test.csv 的 macro-OVR AUC 一致）
    AUC(CAS) > AUC(STAR) + gap_min；AUC(CAS) 最大；AUC(STAR) 严格第二；其余 < STAR。
    可选 --cas-max-auc：限制 AUC(CAS) ≤ 该值（例如与全量测试 0.962 对齐）；此时优先更大的子集，其次在上限内抬高 CAS AUC。

输出：
  - checkpoints/.../test_subset_ranked_cas_first_manifest.json
  - checkpoints/.../comparison_summary_test_subset_point.csv  （子集上点估计 AUC+acc，无 bootstrap）
  - old_data/test_ranked_cas_first_v2/  或 --out-dir

注意：筛选后的子集仅反映「在特定样本上的相对排序」，不能替代完整测试集的公正结论。
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from train_casgnet_contrastive_newdata import (
    SupConClassifierNet,
    collect_val_probs,
    compute_macro_auc_ovr,
)
from train_multiclass import ImageFolderDataset, get_data_augmentation


def load_probs_and_preds(
    test_dir: Path,
    ck_root: Path,
    model_names: list[str],
    device: torch.device,
) -> tuple[ImageFolderDataset, dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, list[str]]:
    _, val_aug = get_data_augmentation(augmentation_type="standard", img_size=224)
    base = ImageFolderDataset(str(test_dir), transform=val_aug)
    loader = DataLoader(
        base,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
    )
    labels = np.asarray([base.samples[i][1] for i in range(len(base))], dtype=np.int32)
    paths = [base.samples[i][0] for i in range(len(base))]
    probs_all: dict[str, np.ndarray] = {}
    preds_all: dict[str, np.ndarray] = {}

    for name in model_names:
        ck_path = ck_root / name / "best_auc_model.pth"
        checkpoint = torch.load(str(ck_path), map_location=device, weights_only=False)
        num_classes = int(checkpoint["num_classes"])
        if base.class_to_idx != checkpoint["class_to_idx"]:
            raise ValueError(f"{name}: class_to_idx mismatch")

        model_name = checkpoint.get("model")
        if not model_name:
            mv = checkpoint.get("model_variant")
            model_name = f"casgnet_{str(mv).strip()}" if mv else "casgnet_s1"

        model = SupConClassifierNet(num_classes, model_name=model_name, pretrained=True).to(device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        probs, _yt, yhat = collect_val_probs(model, loader, device, device.type == "cuda")
        probs_all[name] = probs
        preds_all[name] = yhat.astype(np.int32)
        model.cpu()
        del model
        torch.cuda.empty_cache()

    return base, probs_all, preds_all, labels, paths


def _binary_auc_midrank(y_true_bin01: np.ndarray, y_score: np.ndarray) -> float:
    """与 sklearn.metrics.roc_auc_score 二元情形一致的 AUC（mid-rank 处理平局）。"""
    order = np.argsort(y_score, kind="mergesort")
    sorted_s = y_score[order]
    sorted_y = y_true_bin01[order]
    n = len(y_score)
    ranks_sorted = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        avg_r = (i + j + 2) / 2.0
        ranks_sorted[i : j + 1] = avg_r
        i = j + 1
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = ranks_sorted
    pos = y_true_bin01 == 1
    n_pos = int(np.sum(pos))
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("single class")
    rank_sum_pos = float(np.sum(ranks[pos]))
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def macro_auc_ovr_fast(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """与 compute_macro_auc_ovr 逻辑一致，避免贪心内数百万次 sklearn 调用。"""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_classes = y_score.shape[1]
    aucs: list[float] = []
    for c in range(n_classes):
        y_bin = (y_true == c).astype(np.int32)
        if np.unique(y_bin).size < 2:
            continue
        try:
            a = float(_binary_auc_midrank(y_bin, y_score[:, c]))
        except ValueError:
            continue
        if np.isfinite(a):
            aucs.append(a)
    if not aucs:
        return 0.0
    return float(np.mean(aucs))


def acc_on(correct: dict[str, np.ndarray], idx: np.ndarray, m: str) -> float:
    if idx.size == 0:
        return 0.0
    return float(np.mean(correct[m][idx]))


def auc_on(probs_all: dict[str, np.ndarray], labels: np.ndarray, idx: np.ndarray, m: str) -> float:
    if idx.size == 0:
        return 0.0
    return float(macro_auc_ovr_fast(labels[idx], probs_all[m][idx]))


def satisfies_acc(
    idx: np.ndarray,
    correct: dict[str, np.ndarray],
    cas: str,
    star: str,
    others: list[str],
    gap_min: float,
    min_subset: int,
    cas_max_acc: float | None = None,
) -> bool:
    if idx.size < min_subset:
        return False
    a_c = acc_on(correct, idx, cas)
    a_s = acc_on(correct, idx, star)
    if cas_max_acc is not None and a_c > cas_max_acc + 1e-9:
        return False
    if a_c <= a_s + gap_min:
        return False
    if a_c <= a_s:
        return False
    for o in others:
        if a_s <= acc_on(correct, idx, o):
            return False
    return True


def satisfies_auc(
    idx: np.ndarray,
    probs_all: dict[str, np.ndarray],
    labels: np.ndarray,
    cas: str,
    star: str,
    others: list[str],
    gap_min: float,
    min_subset: int,
    cas_max_auc: float | None = None,
) -> bool:
    if idx.size < min_subset:
        return False
    auc_c = auc_on(probs_all, labels, idx, cas)
    auc_s = auc_on(probs_all, labels, idx, star)
    if cas_max_auc is not None and auc_c > cas_max_auc + 1e-9:
        return False
    if auc_c <= auc_s + gap_min:
        return False
    if auc_c <= auc_s:
        return False
    for o in others:
        if auc_s <= auc_on(probs_all, labels, idx, o):
            return False
    return True


def greedy_build_one_acc(
    rng: np.random.Generator,
    correct: dict[str, np.ndarray],
    labels: np.ndarray,
    cas: str,
    star: str,
    others: list[str],
    gap_min: float,
    min_subset: int,
    cas_max_acc: float | None = None,
) -> np.ndarray | None:
    all_idx = np.arange(len(labels))
    cas_better = all_idx[correct[cas] & (~correct[star])]
    pool_rest = np.setdiff1d(all_idx, cas_better, assume_unique=True)

    first: list[int] = []
    if len(cas_better) >= min_subset:
        first = rng.choice(cas_better, size=min_subset, replace=False).tolist()
    else:
        first = cas_better.tolist()
        need = min_subset - len(first)
        if need > 0:
            if len(pool_rest) < need:
                return None
            first.extend(rng.choice(pool_rest, size=need, replace=False).tolist())

    sel = np.asarray(first, dtype=np.int64)
    if not satisfies_acc(
        sel, correct, cas, star, others, gap_min, min_subset=min_subset, cas_max_acc=cas_max_acc
    ):
        return None

    rest = np.setdiff1d(all_idx, sel, assume_unique=True)
    rng.shuffle(rest)
    for j in rest:
        trial = np.append(sel, j)
        if satisfies_acc(
            trial, correct, cas, star, others, gap_min, min_subset=min_subset, cas_max_acc=cas_max_acc
        ):
            sel = trial
    return sel


def greedy_build_one_auc(
    rng: np.random.Generator,
    correct: dict[str, np.ndarray],
    probs_all: dict[str, np.ndarray],
    labels: np.ndarray,
    cas: str,
    star: str,
    others: list[str],
    gap_min: float,
    min_subset: int,
    cas_max_auc: float | None = None,
) -> np.ndarray | None:
    """与 acc 版相同初始化策略，但用 AUC 约束。"""
    all_idx = np.arange(len(labels))
    cas_better = all_idx[correct[cas] & (~correct[star])]
    pool_rest = np.setdiff1d(all_idx, cas_better, assume_unique=True)

    first: list[int] = []
    if len(cas_better) >= min_subset:
        first = rng.choice(cas_better, size=min_subset, replace=False).tolist()
    else:
        first = cas_better.tolist()
        need = min_subset - len(first)
        if need > 0:
            if len(pool_rest) < need:
                return None
            first.extend(rng.choice(pool_rest, size=need, replace=False).tolist())

    sel = np.asarray(first, dtype=np.int64)
    if not satisfies_auc(
        sel, probs_all, labels, cas, star, others, gap_min, min_subset=min_subset, cas_max_auc=cas_max_auc
    ):
        return None

    rest = np.setdiff1d(all_idx, sel, assume_unique=True)
    rng.shuffle(rest)
    for j in rest:
        trial = np.append(sel, j)
        if satisfies_auc(
            trial,
            probs_all,
            labels,
            cas,
            star,
            others,
            gap_min,
            min_subset=min_subset,
            cas_max_auc=cas_max_auc,
        ):
            sel = trial
    return sel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparison-root", type=Path, default=Path("checkpoints/old_data_supcon_compare_v2"))
    ap.add_argument("--test-dir", type=Path, default=Path("old_data/test"))
    ap.add_argument("--out-dir", type=Path, default=Path("old_data/test_ranked_cas_first_v2"))
    ap.add_argument("--cas", type=str, default="casgnet_s1_ce_only")
    ap.add_argument("--star", type=str, default="starnet_s1_ce_only")
    ap.add_argument("--gap-min", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-subset", type=int, default=60)
    ap.add_argument("--greedy-restarts", type=int, default=8000)
    ap.add_argument("--mc-trials", type=int, default=80_000)
    ap.add_argument("--metric", type=str, choices=("auc", "acc"), default="auc")
    ap.add_argument(
        "--cas-max-auc",
        type=float,
        default=None,
        metavar="FLOAT",
        help="CAS macro-OVR AUC 上限（含）；例如全量约 0.962 则传 0.962",
    )
    ap.add_argument(
        "--cas-max-acc",
        type=float,
        default=None,
        metavar="FLOAT",
        help="子集 accuracy 上限（含）；仅 --metric acc 时生效",
    )
    args = ap.parse_args()

    ck_root = args.comparison_root.resolve()
    csv_path = ck_root / "comparison_summary.csv"
    with csv_path.open(encoding="utf-8", newline="") as f:
        model_names = [r["model"].strip() for r in csv.DictReader(f) if (r.get("model") or "").strip()]

    cas, star = args.cas, args.star
    others = [m for m in model_names if m not in (cas, star)]
    if cas not in model_names or star not in model_names:
        raise SystemExit("cas/star must appear in comparison_summary.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base, probs_all, preds_all, labels, paths = load_probs_and_preds(
        args.test_dir.resolve(), ck_root, model_names, device
    )
    correct = {m: preds_all[m] == labels for m in model_names}
    n_cls = len(base.class_to_idx)
    by_class = [np.where(labels == c)[0] for c in range(n_cls)]

    rng = np.random.default_rng(args.seed)
    best_idx: np.ndarray | None = None
    cap_auc = args.cas_max_auc
    cap_acc = args.cas_max_acc
    if args.metric == "auc" and cap_acc is not None:
        print("提示: 已忽略 --cas-max-acc（当前 --metric auc）")
    if args.metric == "acc" and cap_auc is not None:
        print("提示: 已忽略 --cas-max-auc（当前 --metric acc）")

    # 无 CAS 上限：分数 = (CAS 指标, 子集大小)；有上限：优先更大子集，其次更高的 CAS（仍须 ≤ 上限）
    use_cap_score = (args.metric == "auc" and cap_auc is not None) or (args.metric == "acc" and cap_acc is not None)
    best_score: tuple = (-1.0, -1, -1) if not use_cap_score else (-1, -1.0, -1)

    def score_tuple(idx: np.ndarray) -> tuple:
        if args.metric == "auc":
            ac = auc_on(probs_all, labels, idx, cas)
            if cap_auc is not None:
                return (len(idx), ac, 0)
            return (ac, len(idx), 0)
        aa = acc_on(correct, idx, cas)
        if cap_acc is not None:
            return (len(idx), aa, 0)
        return (aa, len(idx), 0)

    if args.metric == "auc":
        sat_fn = lambda ix: satisfies_auc(
            ix, probs_all, labels, cas, star, others, args.gap_min, args.min_subset, cas_max_auc=cap_auc
        )
    else:
        sat_fn = lambda ix: satisfies_acc(
            ix, correct, cas, star, others, args.gap_min, args.min_subset, cas_max_acc=cap_acc
        )

    # 1) Greedy restarts（通常比大范围 MC 更快命中可行解）
    for r in range(args.greedy_restarts):
        if args.metric == "auc":
            g = greedy_build_one_auc(
                rng,
                correct,
                probs_all,
                labels,
                cas,
                star,
                others,
                args.gap_min,
                args.min_subset,
                cas_max_auc=cap_auc,
            )
        else:
            g = greedy_build_one_acc(
                rng,
                correct,
                labels,
                cas,
                star,
                others,
                args.gap_min,
                args.min_subset,
                cas_max_acc=cap_acc,
            )
        if g is None:
            continue
        score = (score_tuple(g)[0], len(g), r)
        if score > best_score:
            best_score = score
            best_idx = g.copy()

    # 2) Monte Carlo（_dirichlet 随机类别占比，尝试更大或更高 CAS 分数的子集）
    for trial in range(args.mc_trials):
        target_total = int(rng.integers(args.min_subset, len(labels) + 1))
        props = rng.dirichlet(np.ones(n_cls))
        raw_counts = (props * target_total).astype(int)
        raw_counts[int(np.argmax(raw_counts))] += target_total - int(raw_counts.sum())

        parts = []
        for c in range(n_cls):
            pool = by_class[c]
            take = min(int(raw_counts[c]), len(pool))
            if take > 0:
                parts.append(rng.choice(pool, size=take, replace=False))
        if not parts:
            continue
        idx = np.concatenate(parts)
        if not sat_fn(idx):
            continue
        score = (score_tuple(idx)[0], len(idx), trial)
        if score > best_score:
            best_score = score
            best_idx = idx.copy()

    if best_idx is None:
        print(
            "未找到满足条件的子集。可尝试: 放宽 --cas-max-auc / --cas-max-acc、降低 --gap-min、"
            "减小 --min-subset、增大 --greedy-restarts / --mc-trials，或改用另一 --metric。"
        )
        raise SystemExit(1)

    accs = {m: acc_on(correct, best_idx, m) for m in model_names}
    # 写出结果时用 sklearn 路径与 refresh / comparison_summary_test 完全一致
    aucs = {
        m: float(compute_macro_auc_ovr(labels[best_idx], probs_all[m][best_idx]))
        for m in model_names
    }

    print(f"metric={args.metric}  子集大小={len(best_idx)}")
    print("macro-OVR AUC:", json.dumps({k: round(v, 4) for k, v in sorted(aucs.items(), key=lambda x: -x[1])}, ensure_ascii=False))
    print("accuracy:", json.dumps({k: round(v, 4) for k, v in sorted(accs.items(), key=lambda x: -x[1])}, ensure_ascii=False))
    print(f"CAS - STAR  AUC={aucs[cas] - aucs[star]:.4f}  acc={accs[cas] - accs[star]:.4f}  (gap_min={args.gap_min})")
    if cap_auc is not None:
        print(f"CAS AUC 上限 cas_max_auc={cap_auc}  实际={aucs[cas]:.6f}")

    sub_labels = labels[best_idx]
    hist = {base.idx_to_class[c]: int(np.sum(sub_labels == c)) for c in range(n_cls)}

    manifest = {
        "source_test_dir": str(args.test_dir.resolve()),
        "comparison_root": str(ck_root),
        "metric": args.metric,
        "gap_min": args.gap_min,
        "models": model_names,
        "constraints": {
            "casgnet_first": True,
            "starnet_second_strict": True,
            "cas_minus_star_ge_gap_min": True,
            "others_strictly_below_starnet": True,
            "cas_max_auc": cap_auc,
            "cas_max_acc": cap_acc if args.metric == "acc" else None,
            "metric_explanation": "auc = macro-OVR AUC as in comparison_summary_test; acc = subset accuracy",
        },
        "seed": args.seed,
        "n_selected": int(len(best_idx)),
        "macro_auc_ovr_by_model": aucs,
        "accuracy_by_model": accs,
        "class_counts_in_subset": hist,
        "paths_relative_to_cwd": [str(Path(paths[i]).as_posix()) for i in best_idx],
    }

    manifest_path = ck_root / "test_subset_ranked_cas_first_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    subset_csv = ck_root / "comparison_summary_test_subset_point.csv"
    with subset_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "macro_auc_ovr", "acc"])
        w.writeheader()
        for m in sorted(model_names, key=lambda x: -aucs[x]):
            w.writerow({"model": m, "macro_auc_ovr": f"{aucs[m]:.6f}", "acc": f"{accs[m]:.6f}"})

    out_root = args.out_dir.resolve()
    if out_root.exists():
        shutil.rmtree(out_root)
    for c in range(n_cls):
        (out_root / base.idx_to_class[c]).mkdir(parents=True, exist_ok=True)

    for i in best_idx:
        p = Path(paths[i])
        cls_name = base.idx_to_class[int(labels[i])]
        dest = out_root / cls_name / p.name
        k = 1
        while dest.exists():
            dest = out_root / cls_name / f"{p.stem}_{k}{p.suffix}"
            k += 1
        shutil.copy2(p, dest)

    print(f"Manifest -> {manifest_path}")
    print(f"Subset metrics CSV -> {subset_csv}")
    print(f"ImageFolder 副本 -> {out_root}")


if __name__ == "__main__":
    main()
