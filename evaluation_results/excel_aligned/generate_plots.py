#!/usr/bin/env python3
"""
从 evaluation_results/excel_aligned/caches/*.npz 重新生成全部 ROC / 混淆矩阵图。

用法:
  python evaluation_results/excel_aligned/generate_plots.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE / "run_all_models_eval.py"


def main() -> None:
    subprocess.run([sys.executable, str(RUN), "--skip-inference"], check=True)


if __name__ == "__main__":
    main()
