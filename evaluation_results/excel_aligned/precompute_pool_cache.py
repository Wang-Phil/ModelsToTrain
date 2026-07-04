#!/usr/bin/env python3
"""Precompute Table-1 pool inference caches (one NPZ per model, no subset search).

Usage (project root):
  python evaluation_results/excel_aligned/precompute_pool_cache.py
  python evaluation_results/excel_aligned/precompute_pool_cache.py --models casgnet starnet_s1
  python evaluation_results/excel_aligned/precompute_pool_cache.py --force-recompute-pool
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MATCH = ROOT / "evaluation_results/excel_aligned/match_excel_table1_per_model.py"


def main() -> None:
    cmd = [sys.executable, str(MATCH), "--precompute-only", *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd, cwd=str(ROOT)))


if __name__ == "__main__":
    main()
