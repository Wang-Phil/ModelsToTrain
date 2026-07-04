#!/usr/bin/env bash
# CASGNet（models/casgnet.py）SA / GRN / 末 stage SelectiveKernel(SKSGBlock) 三模块 2^3=8 组全因子消融。
# 训练与 checkpoints/old_data_supcon_compare_v3 一致：train_casgnet_contrastive_newdata.py + 末尾 refresh。
#
# 命名 casgnet_s1_abXYZ：X=空间注意力 SA，Y=GRN，Z=末 stage SK（与代码开关一致）。
#
# 用法:
#   export GPUS=0,1,2,3
#   bash scripts/run_casgnet_sa_grn_sk_ablation_parallel.sh
#
# 环境变量（与 run_compare_old_data_supcon.sh 对齐）:
#   DATA_DIR, GPU_LIST 或 GPUS, EPOCHS, BATCH_SIZE, RUN_TAG, RUN_LEAF_SUFFIX, PRETRAINED_FLAG, SKIP_REFRESH

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SCRIPT_DIR}"

DATA_DIR="${DATA_DIR:-old_data}"
if [[ -z "${GPU_LIST:-}" ]]; then
  if [[ -n "${GPUS:-}" ]]; then
    GPU_LIST=""
    IFS=',' read -r -a _GPUS_TOK <<< "${GPUS}"
    for _id in "${_GPUS_TOK[@]}"; do
      _id="$(echo "${_id}" | xargs)"
      [[ -z "${_id}" ]] && continue
      [[ -n "${GPU_LIST}" ]] && GPU_LIST+=","
      if [[ "${_id}" == cuda:* ]]; then GPU_LIST+="${_id}"
      else GPU_LIST+="cuda:${_id}"; fi
    done
  else
    GPU_LIST="cuda:0"
  fi
fi

EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-42}"
VAL_RATIO="${VAL_RATIO:-0.2}"
AUGMENTATION="${AUGMENTATION:-standard}"
PRETRAINED_FLAG="${PRETRAINED_FLAG:---pretrained}"
RUN_TAG="${RUN_TAG:-casgnet_sa_grn_sk_ablation}"
RUN_LEAF_SUFFIX="${RUN_LEAF_SUFFIX:-_ce_only}"
SKIP_REFRESH="${SKIP_REFRESH:-0}"

OUT_ROOT="${OUT_ROOT:-checkpoints/${RUN_TAG}}"
mkdir -p "${OUT_ROOT}"
LOG_DIR="${OUT_ROOT}/logs"
STATUS_DIR="${OUT_ROOT}/_status"
mkdir -p "${LOG_DIR}" "${STATUS_DIR}"
rm -f "${OUT_ROOT}/failed_runs.csv"

declare -a MODELS=(
  "casgnet_s1_ab000"
  "casgnet_s1_ab100"
  "casgnet_s1_ab010"
  "casgnet_s1_ab001"
  "casgnet_s1_ab110"
  "casgnet_s1_ab101"
  "casgnet_s1_ab011"
  "casgnet_s1_ab111"
)

echo "== CASGNet SA/GRN/SK 2^3 ablation (train_casgnet_contrastive_newdata.py) =="
echo "RUN_TAG=${RUN_TAG}  RUN_LEAF_SUFFIX=${RUN_LEAF_SUFFIX}"
echo "data_dir=${DATA_DIR}  out_root=${OUT_ROOT}"
echo "gpu_list=${GPU_LIST}  epochs=${EPOCHS}"
echo

USE_FIXED_SPLIT=0
USE_TEST_DIR=0
if [[ -d "${DATA_DIR}/train" && -d "${DATA_DIR}/val" ]]; then
  USE_FIXED_SPLIT=1
  echo "split_mode=fixed"
  [[ -d "${DATA_DIR}/test" ]] && USE_TEST_DIR=1 && echo "test_dir=${DATA_DIR}/test"
else
  echo "split_mode=random val_ratio=${VAL_RATIO}"
fi
echo

IFS=',' read -r -a GPU_TOKENS <<< "${GPU_LIST}"
declare -a GPUS_ARR=()
for token in "${GPU_TOKENS[@]}"; do
  t="$(echo "${token}" | xargs)"
  [[ -z "${t}" ]] && continue
  if [[ "${t}" == cuda:* ]]; then GPUS_ARR+=("${t}")
  else GPUS_ARR+=("cuda:${t}"); fi
done
if [[ ${#GPUS_ARR[@]} -eq 0 ]]; then echo "错误: 无可用 GPU_LIST/GPUS"; exit 1; fi

run_one() {
  local model="$1"
  local gpu="$2"
  local leaf="${model}${RUN_LEAF_SUFFIX}"
  local run_dir="${OUT_ROOT}/${leaf}"
  local log_file="${LOG_DIR}/${leaf}.log"
  local status_file="${STATUS_DIR}/${leaf}.status"
  mkdir -p "${run_dir}"
  echo ">>> [START] ${model} gpu=${gpu}"
  set +e
  if [[ ${USE_FIXED_SPLIT} -eq 1 ]]; then
    test_args=()
    [[ ${USE_TEST_DIR} -eq 1 ]] && test_args=(--test-dir "${DATA_DIR}/test")
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
      ${PRETRAINED_FLAG} >"${log_file}" 2>&1
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
      ${PRETRAINED_FLAG} >"${log_file}" 2>&1
  fi
  local code=$?
  set -e
  if [[ ${code} -eq 0 ]]; then echo "OK,0" >"${status_file}"; echo ">>> [DONE] ${model}"
  else echo "FAILED,${code}" >"${status_file}"; echo ">>> [FAIL] ${model} log=${log_file}"; fi
}

declare -a SLOT_PIDS=()
for i in "${!GPUS_ARR[@]}"; do SLOT_PIDS[i]=""; done
total="${#MODELS[@]}"
next_idx=0
running=0
while [[ ${next_idx} -lt ${total} || ${running} -gt 0 ]]; do
  for i in "${!GPUS_ARR[@]}"; do
    pid="${SLOT_PIDS[i]}"
    gpu="${GPUS_ARR[i]}"
    if [[ -n "${pid}" ]]; then
      if ! kill -0 "${pid}" 2>/dev/null; then
        wait "${pid}" || true
        SLOT_PIDS[i]=""
        running=$((running - 1))
      fi
    fi
    if [[ -z "${SLOT_PIDS[i]}" && ${next_idx} -lt ${total} ]]; then
      model="${MODELS[next_idx]}"
      next_idx=$((next_idx + 1))
      run_one "${model}" "${gpu}" &
      SLOT_PIDS[i]="$!"
      running=$((running + 1))
    fi
  done
  [[ ${running} -gt 0 ]] && sleep 2
done

for model in "${MODELS[@]}"; do
  leaf="${model}${RUN_LEAF_SUFFIX}"
  sf="${STATUS_DIR}/${leaf}.status"
  if [[ ! -f "${sf}" ]]; then echo "${model},FAILED,NO_STATUS" >>"${OUT_ROOT}/failed_runs.csv"; continue; fi
  sl="$(<"${sf}")"
  [[ "${sl%%,*}" != "OK" ]] && echo "${model},FAILED,${sl#*,}" >>"${OUT_ROOT}/failed_runs.csv"
done

if [[ "${SKIP_REFRESH}" != "1" ]]; then
  python3 refresh_supcon_checkpoint_metrics.py --comparison-refresh-all "${OUT_ROOT}"
else
  echo "SKIP_REFRESH=1，跳过 refresh"
fi

python3 scripts/summarize_casgnet_module_ablation_excel.py \
  --experiment-root "${OUT_ROOT}" \
  --run-leaf-suffix "${RUN_LEAF_SUFFIX}" \
  --output "${OUT_ROOT}/summary_casgnet_sa_grn_sk_ablation.xlsx"

echo "完成: ${OUT_ROOT}/summary_casgnet_sa_grn_sk_ablation.xlsx"
