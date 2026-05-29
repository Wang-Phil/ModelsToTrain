#!/usr/bin/env bash
# 单独训练一个模型，输出目录与 run_compare_old_data_supcon.sh 相同，便于汇总进
#   checkpoints/old_data_supcon_compare/
#
# 用法:
#   ./train_single_compare_model.sh lsnet_b
#   MODEL=starnet_s1 DEVICE=cuda:1 EPOCHS=50 ./train_single_compare_model.sh
#
# 子目录名: ${MODEL}${RUN_LEAF_SUFFIX}，默认 RUN_LEAF_SUFFIX=_ce_only
# 与批量脚本一致的环境变量: DATA_DIR, DEVICE, EPOCHS, BATCH_SIZE, RUN_TAG, RUN_LEAF_SUFFIX, 等

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

MODEL="${MODEL:-${1:-}}"
if [[ -z "${MODEL}" ]]; then
  echo "用法: $0 <model_name>"
  echo "示例: $0 lsnet_b"
  echo "或在前面设置: MODEL=resnet50 $0"
  exit 1
fi

DATA_DIR="${DATA_DIR:-old_data}"
DEVICE="${DEVICE:-cuda:0}"
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
LEAF="${MODEL}${RUN_LEAF_SUFFIX}"
RUN_DIR="${OUT_ROOT}/${LEAF}"
LOG_DIR="${OUT_ROOT}/logs"
STATUS_DIR="${OUT_ROOT}/_status"
mkdir -p "${RUN_DIR}" "${LOG_DIR}" "${STATUS_DIR}"

LOG_FILE="${LOG_DIR}/${LEAF}.log"
STATUS_FILE="${STATUS_DIR}/${LEAF}.status"

USE_FIXED_SPLIT=0
USE_TEST_DIR=0
if [[ -d "${DATA_DIR}/train" && -d "${DATA_DIR}/val" ]]; then
  USE_FIXED_SPLIT=1
  if [[ -d "${DATA_DIR}/test" ]]; then
    USE_TEST_DIR=1
  fi
fi

echo "== 单模型训练 =="
echo "model=${MODEL}"
echo "out_dir=${RUN_DIR}"
echo "log=${LOG_FILE}"
echo "data_dir=${DATA_DIR}  fixed_split=${USE_FIXED_SPLIT}"
echo "device=${DEVICE}  epochs=${EPOCHS}  batch_size=${BATCH_SIZE}"
echo

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
    --model "${MODEL}" \
    --device "${DEVICE}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --seed "${SEED}" \
    --augmentation "${AUGMENTATION}" \
    --output-dir "${RUN_DIR}" \
    ${PRETRAINED_FLAG} 2>&1 | tee "${LOG_FILE}"
else
  python3 train_casgnet_contrastive_newdata.py \
    --data-dir "${DATA_DIR}" \
    --model "${MODEL}" \
    --device "${DEVICE}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --seed "${SEED}" \
    --val-ratio "${VAL_RATIO}" \
    --augmentation "${AUGMENTATION}" \
    --output-dir "${RUN_DIR}" \
    ${PRETRAINED_FLAG} 2>&1 | tee "${LOG_FILE}"
fi
code=$?
set -e

if [[ ${code} -eq 0 ]]; then
  echo "OK,0" > "${STATUS_FILE}"
  echo ">>> 完成: ${RUN_DIR}"
else
  echo "FAILED,${code}" > "${STATUS_FILE}"
  echo ">>> 失败 exit=${code}，见 ${LOG_FILE}"
  exit "${code}"
fi
