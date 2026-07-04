#!/usr/bin/env bash
set -euo pipefail

# 对比实验:
# GoogLeNet, ResNet50, ResNet18, DenseNet121, MobileNetV4-M, StarNet-S1, CASGNet-S1, LSNet-B
# 训练脚本默认 --ce-only；输出仅在最后一级目录名加 RUN_LEAF_SUFFIX（默认 _ce_only）。
# 若要做 SupCon+CE: python 命令加 --no-ce-only，并可设 RUN_LEAF_SUFFIX="" 或自定义后缀。
#
# 若 DATA_DIR 下存在 test/（固定划分 train/val 时），自动传 --test-dir，训练结束后写 test_eval；
# 汇总末尾 refresh 会生成 comparison_summary_test.csv 与 comparison_summary.xlsx 中测试相关 sheet。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DATA_DIR="${DATA_DIR:-old_data}"
DEVICE="${DEVICE:-cuda:0}"
GPU_LIST="${GPU_LIST:-${DEVICE}}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-42}"
VAL_RATIO="${VAL_RATIO:-0.2}"
AUGMENTATION="${AUGMENTATION:-standard}"
PRETRAINED_FLAG="${PRETRAINED_FLAG:---pretrained}"
RUN_TAG="${RUN_TAG:-old_data_supcon_compare}"
RUN_LEAF_SUFFIX="${RUN_LEAF_SUFFIX:-_ce_only}"

OUT_ROOT="${OUT_ROOT:-checkpoints/${RUN_TAG}}"
mkdir -p "${OUT_ROOT}"
LOG_DIR="${OUT_ROOT}/logs"
STATUS_DIR="${OUT_ROOT}/_status"
mkdir -p "${LOG_DIR}" "${STATUS_DIR}"
rm -f "${OUT_ROOT}/failed_runs.csv"

declare -a MODELS=(
  "googlenet"
  "resnet50"
  "resnet18"
  "densenet121"
  "mobilenetv4_m"
  "starnet_s1"
  "casgnet_s1"
  "lsnet_b"
)

echo "== Model comparison (RUN_TAG=${RUN_TAG}, RUN_LEAF_SUFFIX=${RUN_LEAF_SUFFIX}) =="
echo "data_dir=${DATA_DIR}"
echo "gpu_list=${GPU_LIST}"
echo "epochs=${EPOCHS}"
echo "batch_size=${BATCH_SIZE}"
echo "out_root=${OUT_ROOT}"
echo

USE_FIXED_SPLIT=0
USE_TEST_DIR=0
if [[ -d "${DATA_DIR}/train" && -d "${DATA_DIR}/val" ]]; then
  USE_FIXED_SPLIT=1
  echo "split_mode=fixed (auto-detected train/val)"
  if [[ -d "${DATA_DIR}/test" ]]; then
    USE_TEST_DIR=1
    echo "test_dir=${DATA_DIR}/test (macro metrics after train + bootstrap in refresh)"
  fi
else
  echo "split_mode=random (--val-ratio=${VAL_RATIO})"
fi
echo

IFS=',' read -r -a GPU_TOKENS <<< "${GPU_LIST}"
declare -a GPUS=()
for token in "${GPU_TOKENS[@]}"; do
  t="$(echo "${token}" | xargs)"
  [[ -z "${t}" ]] && continue
  if [[ "${t}" == cuda:* ]]; then
    GPUS+=("${t}")
  else
    GPUS+=("cuda:${t}")
  fi
done

if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "错误: 未解析到可用 GPU，请检查 GPU_LIST 格式（例如: cuda:0,cuda:1 或 0,1）"
  exit 1
fi

run_one() {
  local model="$1"
  local gpu="$2"
  local leaf="${model}${RUN_LEAF_SUFFIX}"
  local run_dir="${OUT_ROOT}/${leaf}"
  local log_file="${LOG_DIR}/${leaf}.log"
  local status_file="${STATUS_DIR}/${leaf}.status"
  mkdir -p "${run_dir}"

  echo ">>> [START] model=${model} gpu=${gpu}"
  set +e
  if [[ ${USE_FIXED_SPLIT} -eq 1 ]]; then
    test_args=()
    if [[ ${USE_TEST_DIR} -eq 1 ]]; then
      test_args=(--test-dir "${DATA_DIR}/test")
    fi
    python3 train_casgnet_contrastive_newdata.py \
      --train-dir "${DATA_DIR}/train" \
      --val-dir "${DATA_DIR}/val" \
      "${test_args[@]}" \
      --model "${model}" \
      --device "${gpu}" \
      --epochs "${EPOCHS}" \
      --batch-size "${BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --seed "${SEED}" \
      --augmentation "${AUGMENTATION}" \
      --output-dir "${run_dir}" \
      ${PRETRAINED_FLAG} > "${log_file}" 2>&1
  else
    python3 train_casgnet_contrastive_newdata.py \
      --data-dir "${DATA_DIR}" \
      --model "${model}" \
      --device "${gpu}" \
      --epochs "${EPOCHS}" \
      --batch-size "${BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --seed "${SEED}" \
      --val-ratio "${VAL_RATIO}" \
      --augmentation "${AUGMENTATION}" \
      --output-dir "${run_dir}" \
      ${PRETRAINED_FLAG} > "${log_file}" 2>&1
  fi
  local code=$?
  set -e

  if [[ ${code} -eq 0 ]]; then
    echo "OK,0" > "${status_file}"
    echo ">>> [DONE] model=${model} gpu=${gpu}"
  else
    echo "FAILED,${code}" > "${status_file}"
    echo ">>> [FAIL] model=${model} gpu=${gpu} exit_code=${code}"
    echo "    log: ${log_file}"
  fi
}

declare -a SLOT_PIDS=()
declare -a SLOT_MODELS=()
declare -a SLOT_GPUS=()

for i in "${!GPUS[@]}"; do
  SLOT_PIDS[i]=""
  SLOT_MODELS[i]=""
  SLOT_GPUS[i]="${GPUS[i]}"
done

total="${#MODELS[@]}"
next_idx=0
running=0

while [[ ${next_idx} -lt ${total} || ${running} -gt 0 ]]; do
  for i in "${!GPUS[@]}"; do
    pid="${SLOT_PIDS[i]}"
    gpu="${SLOT_GPUS[i]}"

    if [[ -n "${pid}" ]]; then
      if ! kill -0 "${pid}" 2>/dev/null; then
        wait "${pid}" || true
        SLOT_PIDS[i]=""
        SLOT_MODELS[i]=""
        running=$((running - 1))
      fi
    fi

    if [[ -z "${SLOT_PIDS[i]}" && ${next_idx} -lt ${total} ]]; then
      model="${MODELS[next_idx]}"
      next_idx=$((next_idx + 1))
      run_one "${model}" "${gpu}" &
      SLOT_PIDS[i]="$!"
      SLOT_MODELS[i]="${model}"
      running=$((running + 1))
    fi
  done

  if [[ ${running} -gt 0 ]]; then
    sleep 2
  fi
done

for model in "${MODELS[@]}"; do
  leaf="${model}${RUN_LEAF_SUFFIX}"
  status_file="${STATUS_DIR}/${leaf}.status"
  if [[ ! -f "${status_file}" ]]; then
    echo "${model},FAILED,NO_STATUS_FILE" >> "${OUT_ROOT}/failed_runs.csv"
    continue
  fi
  status_line="$(<"${status_file}")"
  state="${status_line%%,*}"
  code="${status_line#*,}"
  if [[ "${state}" != "OK" ]]; then
    echo "${model},FAILED,${code}" >> "${OUT_ROOT}/failed_runs.csv"
  fi
done

OUT_ROOT="${OUT_ROOT}" SCRIPT_DIR="${SCRIPT_DIR}" python3 - <<'PY'
import csv
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.environ["SCRIPT_DIR"])
from refresh_supcon_checkpoint_metrics import refresh_comparison_all_per_class

out_root = Path(os.environ["OUT_ROOT"])
script_root = Path(os.environ["SCRIPT_DIR"]).resolve()
rows = []

def fmt_ci(mean_v, low_v=None, high_v=None):
    try:
        m = float(mean_v)
    except Exception:
        return ""
    if low_v is None or high_v is None:
        return f"{m:.3f}"
    try:
        l = float(low_v)
        h = float(high_v)
    except Exception:
        return f"{m:.3f}"
    return f"{m:.3f}({l:.3f}-{h:.3f})"

for model_dir in sorted([p for p in out_root.iterdir() if p.is_dir()]):
    summary_file = model_dir / "result_summary.json"
    if not summary_file.is_file():
        continue
    try:
        data = json.loads(summary_file.read_text(encoding="utf-8"))
    except Exception:
        continue
    b = data.get("bootstrap_auc", {}) or {}
    bm = data.get("bootstrap_metrics", {}) or {}
    sens_b = bm.get("sensitivity", {}) if isinstance(bm.get("sensitivity", {}), dict) else {}
    spec_b = bm.get("specificity", {}) if isinstance(bm.get("specificity", {}), dict) else {}
    npv_b = bm.get("npv", {}) if isinstance(bm.get("npv", {}), dict) else {}
    ppv_b = bm.get("ppv", {}) if isinstance(bm.get("ppv", {}), dict) else {}
    acc_b = bm.get("acc", {}) if isinstance(bm.get("acc", {}), dict) else {}
    auc_point = data.get("auc", data.get("reloaded_val_auc"))
    auc_mean = b.get("mean", auc_point)
    rows.append(
        {
            "model": model_dir.name,
            "auc": fmt_ci(auc_mean, b.get("ci95_low"), b.get("ci95_high")),
            "sensitivity": fmt_ci(
                sens_b.get("mean", data.get("sensitivity")),
                sens_b.get("ci95_low"),
                sens_b.get("ci95_high"),
            ),
            "specificity": fmt_ci(
                spec_b.get("mean", data.get("specificity")),
                spec_b.get("ci95_low"),
                spec_b.get("ci95_high"),
            ),
            "npv": fmt_ci(npv_b.get("mean", data.get("npv")), npv_b.get("ci95_low"), npv_b.get("ci95_high")),
            "ppv": fmt_ci(ppv_b.get("mean", data.get("ppv")), ppv_b.get("ci95_low"), ppv_b.get("ci95_high")),
            "acc": fmt_ci(acc_b.get("mean", data.get("acc")), acc_b.get("ci95_low"), acc_b.get("ci95_high")),
            "_auc_sort": auc_point if auc_point is not None else -1.0,
        }
    )

rows.sort(key=lambda x: -(x.get("_auc_sort", -1.0) or -1.0))
for x in rows:
    x.pop("_auc_sort", None)
csv_path = out_root / "comparison_summary.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "model",
            "auc",
            "sensitivity",
            "specificity",
            "npv",
            "ppv",
            "acc",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"comparison rows: {len(rows)}")
print(f"summary csv: {csv_path}")

refresh_comparison_all_per_class(out_root.resolve(), project_root=script_root)
PY

echo
echo "All done. Results:"
echo "  - ${OUT_ROOT}/comparison_summary.csv（验证集 macro + bootstrap）"
echo "  - ${OUT_ROOT}/comparison_summary_test.csv（测试集 macro + bootstrap，需 DATA_DIR/test）"
echo "  - ${OUT_ROOT}/comparison_summary.xlsx（四 sheet：验证宏观 / 测试宏观 / per_class_val / per_class_test）"
echo "  - ${OUT_ROOT}/failed_runs.csv (if any)"
echo "  - ${OUT_ROOT}/logs/*.log"
echo "Note: 汇总末尾对每个模型在验证集与测试集上 reload checkpoint + bootstrap（默认 n=1000），耗时取决于 GPU 数量。"
