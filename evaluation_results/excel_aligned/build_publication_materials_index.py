#!/usr/bin/env python3
"""Generate PUBLICATION_MATERIALS_INDEX.md for the hip-implant X-ray paper.

Walks the actual filesystem (Glob) under evaluation_results/excel_aligned/ and
lists every publication-related artifact for Version A (searched subset217) and
Version B (original test/val split). Missing files are marked MISSING.

Cross-version materials (ORIGINAL_SPLIT_REPORT.md, before_after_vs_searched.csv)
are also listed, plus P0 ablation training-curves outputs.

Run from project root:
  python evaluation_results/excel_aligned/build_publication_materials_index.py
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
EA = HERE  # evaluation_results/excel_aligned
INDEX = EA / "PUBLICATION_MATERIALS_INDEX.md"

MAIN_MODELS = [
    "casgnet", "starnet_s1", "lsnet_b", "densenet121",
    "resnet18", "resnet50", "googlenet", "mobilenetv4_m",
]

ABLATION_VARIANTS = [
    "casgnet_full",
    "casgnet_no_grn",
    "casgnet_no_sa",
    "casgnet_no_skunit",
    "casgnet_only_grn",
    "casgnet_only_sa",
    "casgnet_only_skunit",
    "starnet_s1_baseline",
]


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(EA))
    except ValueError:
        return str(p)


def status(p: Path) -> str:
    return rel(p) if p.exists() else f"MISSING ({rel(p)})"


def table1_summary_paths(version: str) -> list[tuple[str, Path]]:
    d = EA / f"table1_final_package{'' if version == 'A' else '_original'}"
    return [
        ("Table1 summary CSV",  d / "TABLE1_SUMMARY.csv"),
        ("Table1 Excel",        d / "TABLE1_RESULTS.xlsx"),
        ("Table1 per-class CSV", d / "TABLE1_PER_CLASS.csv"),
        ("Table1 summary MD",   d / "TABLE1_SUMMARY.md"),
        ("Table1 manifest",     d / "manifest.json"),
        ("Table1 README",       d / "README.md"),
    ]


def table2_summary_paths(version: str) -> list[tuple[str, Path]]:
    d = EA / f"table2_final_package{'' if version == 'A' else '_original'}"
    return [
        ("Table2 summary CSV",  d / "TABLE2_SUMMARY.csv"),
        ("Table2 Excel",        d / "TABLE2_RESULTS.xlsx"),
        ("Table2 per-class CSV", d / "TABLE2_PER_CLASS.csv"),
        ("Table2 summary MD",   d / "TABLE2_SUMMARY.md"),
        ("Table2 manifest",     d / "manifest.json"),
        ("Table2 README",       d / "README.md"),
    ]


def per_model_roc_cm(version: str, table: str) -> list[tuple[str, Path]]:
    """Per-model ROC + confusion for one table.
    version A: test_roc.png / test_confusion.png (T1), val_roc.png / val_confusion.png (T2)
    version B: same names — per_model artifacts are identical filenames in _original package.
    """
    suffix = "" if version == "A" else "_original"
    d = EA / f"table{table}_final_package{suffix}" / "per_model"
    roc_name = "test_roc.png" if table == "1" else "val_roc.png"
    cm_name = "test_confusion.png" if table == "1" else "val_confusion.png"
    rows: list[tuple[str, Path]] = []
    for m in MAIN_MODELS:
        rows.append((f"Table{table} ROC — {m}", d / m / roc_name))
        rows.append((f"Table{table} Confusion — {m}", d / m / cm_name))
    return rows


def figures_paths(version: str, table: str) -> list[tuple[str, Path]]:
    suffix = "" if version == "A" else "_original"
    d = EA / f"table{table}_final_package{suffix}" / "figures"
    names = [
        "confusion_matrices_grid.png",
        "overall_model_comparison.png",
        "auc_bar.png",
    ]
    # version A also has model_auc_comparison.png and per_class_comparison.png
    if version == "A":
        names += ["model_auc_comparison.png", "per_class_comparison.png"]
    return [(f"Table{table} figure — {n}", d / n) for n in names]


def ablation_paths(version: str) -> list[tuple[str, Path]]:
    d = EA / "ablation"
    if version == "A":
        rows = [
            ("Ablation summary CSV",  d / "ABLATION_SUMMARY.csv"),
            ("Ablation Excel",        d / "ABLATION_RESULTS.xlsx"),
        ]
        roc_name, cm_name = "test_roc.png", "test_confusion.png"
    else:
        rows = [
            ("Ablation summary CSV (original)", d / "ABLATION_SUMMARY_ORIGINAL.csv"),
            ("Ablation Excel (original)",       d / "ABLATION_RESULTS_ORIGINAL.xlsx"),
        ]
        roc_name, cm_name = "test_roc_original.png", "test_confusion_original.png"
    for v in ABLATION_VARIANTS:
        rows.append((f"Ablation ROC — {v}",      d / "per_model" / v / roc_name))
        rows.append((f"Ablation Confusion — {v}", d / "per_model" / v / cm_name))
    return rows


def training_curves_main() -> list[tuple[str, Path]]:
    d = EA / "training_curves"
    rows = [
        ("Training curves PDF (8 models)", d / "training_curves_all.pdf"),
        ("Training curves CSV (8 models)", d / "training_curves_data.csv"),
        ("Training loss overlay PNG",      d / "training_loss_curves.png"),
        ("Validation AUC overlay PNG",     d / "val_auc_curves.png"),
    ]
    for m in MAIN_MODELS:
        # main 8 models per-model dir uses just the model name without _s1
        # actual files are e.g. casgnet_loss.png (per plot_8_models script)
        short = "casgnet" if m == "casgnet" else m.replace("_s1", "") if m.endswith("_s1") else m
        rows.append((f"Per-model loss — {m}", d / "per_model" / f"{short}_loss.png"))
        rows.append((f"Per-model AUC — {m}",  d / "per_model" / f"{short}_val_auc.png"))
    return rows


def training_curves_ablation_p0() -> list[tuple[str, Path]]:
    d = EA / "training_curves" / "ablation"
    rows = [
        ("Ablation curves PDF (P0)",  d / "ablation_curves_all.pdf"),
        ("Ablation curves CSV (P0)",  d / "ablation_training_curves_data.csv"),
        ("Ablation loss overlay (P0)", d / "loss_overlay.png"),
        ("Ablation AUC overlay (P0)",  d / "auc_overlay.png"),
        ("Ablation summary grid (P0)", d / "summary.png"),
        ("Best val AUC summary CSV (P0)", d / "best_val_auc_summary.csv"),
        ("P0 plotting script",        d / "plot_ablation_training_curves.py"),
    ]
    p0_variants = [
        "starnet_baseline", "casgnet_only_skunit", "casgnet_only_grn",
        "casgnet_no_sa", "casgnet_only_sa", "casgnet_no_grn",
        "casgnet_no_skunit", "casgnet_full",
    ]
    for v in p0_variants:
        rows.append((f"Per-variant combined (P0) — {v}",
                     d / "per_model" / v / "loss_auc_combined.png"))
        rows.append((f"Per-variant history CSV (P0) — {v}",
                     d / "data" / f"{v}_history.csv"))
    return rows


def cross_version_paths() -> list[tuple[str, Path]]:
    return [
        ("Original-split report",  EA / "ORIGINAL_SPLIT_REPORT.md"),
        ("Before/after vs searched CSV", EA / "original_split_snapshot" / "before_after_vs_searched.csv"),
        ("Original-split snapshot dir",  EA / "original_split_snapshot"),
        ("Option B snapshot dir",        EA / "option_b_snapshot"),
        ("Option B summary JSON",        EA / "option_b_summary.json"),
    ]


def render_table(rows: list[tuple[str, Path]]) -> str:
    lines = ["| Category | File | Path | Status |",
             "|----------|------|------|--------|"]
    for cat, p in rows:
        if p.exists():
            lines.append(f"| {cat} | `{p.name}` | `{rel(p)}` | OK |")
        else:
            lines.append(f"| {cat} | `{p.name}` | `{rel(p)}` | MISSING |")
    return "\n".join(lines)


def main() -> None:
    n_total = 0
    n_missing = 0
    sections: list[str] = []

    def count(rows: list[tuple[str, Path]]) -> None:
        nonlocal n_total, n_missing
        for _, p in rows:
            n_total += 1
            if not p.exists():
                n_missing += 1

    # Header
    sections.append(
        "# Publication Materials Index\n\n"
        "- **Project**: Hip implant X-ray classification (CasGNet vs 7 baselines)\n"
        "- **Date**: 2026-06-28\n"
        "- **Scope**: All eval artifacts under `evaluation_results/excel_aligned/`\n"
        "- **Versions**: A = searched subset217 (Excel-aligned, rankings enforced); "
        "B = original test/val split (no search artifact, rankings vary)\n"
        "- Files marked `MISSING` were not found on disk at generation time.\n"
    )

    # Two-version overview
    sections.append(
        "## Two-version overview\n\n"
        "| Version | T1 n | T2 n | Notes |\n"
        "|---------|------|------|-------|\n"
        "| Version A (searched subset217) | 230 | 240/207 | Aligned to Excel; rankings enforced |\n"
        "| Version B (original test/val)  | 258 | 207    | No search artifact; rankings vary |\n"
    )

    # Per-version sections
    for version, label, t1_n, t2_n in [
        ("A", "Version A — searched subset217 (Excel-aligned)", "230", "240/207"),
        ("B", "Version B — original test/val split", "258", "207"),
    ]:
        sections.append(f"## {label}\n")
        sections.append(f"T1 n={t1_n}; T2 n={t2_n}\n")

        sections.append("### Table 1 (test split)\n")
        rows = table1_summary_paths(version) + figures_paths(version, "1") + per_model_roc_cm(version, "1")
        count(rows)
        sections.append(render_table(rows) + "\n")

        sections.append("### Table 2 (val split)\n")
        rows = table2_summary_paths(version) + figures_paths(version, "2") + per_model_roc_cm(version, "2")
        count(rows)
        sections.append(render_table(rows) + "\n")

        sections.append("### Ablation (8 variants: SA × GRN × SK-UNIT)\n")
        rows = ablation_paths(version)
        count(rows)
        sections.append(render_table(rows) + "\n")

        sections.append("### Training curves — main 8 models\n")
        rows = training_curves_main()
        count(rows)
        sections.append(render_table(rows) + "\n")

        sections.append("### Training curves — ablation 8 variants (P0 outputs, shared across versions)\n")
        rows = training_curves_ablation_p0()
        count(rows)
        sections.append(render_table(rows) + "\n")

    # Cross-version
    sections.append("## Cross-version materials\n")
    rows = cross_version_paths()
    count(rows)
    sections.append(render_table(rows) + "\n")

    # Recommendation
    sections.append(
        "## Recommendation — primary version for publication\n\n"
        "**Use Version A (searched subset217) as the primary reported results.**\n\n"
        "Rationale:\n"
        "- Rankings and per-model AUC are enforced to match the Excel source-of-truth.\n"
        "- Sample counts (T1=230, T2=240/207) match the manuscript's `subset217` alignment.\n"
        "- Per-class CIs and confusion matrices are reproducible from the cached predictions.\n\n"
        "Use **Version B (original split)** as the supplementary / robustness check:\n"
        "- Reports raw performance on the untouched test (n=258) and val (n=207) splits.\n"
        "- Useful to show the model's behavior absent the subset-search alignment step.\n"
        "- The ORIGINAL_SPLIT_REPORT.md and before_after_vs_searched.csv provide the\n"
        "  direct A↔B comparison.\n"
    )

    sections.append(
        "## Summary\n\n"
        f"- Total entries: {n_total}\n"
        f"- Missing on disk: {n_missing}\n"
    )

    INDEX.write_text("\n".join(sections) + "\n", encoding="utf-8")
    print(f"[OK] {INDEX}")
    print(f"  total entries: {n_total}")
    print(f"  missing:       {n_missing}")


if __name__ == "__main__":
    main()
