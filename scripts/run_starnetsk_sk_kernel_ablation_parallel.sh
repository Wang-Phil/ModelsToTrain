#!/usr/bin/env bash
# StarNet-SK（models/starnetsk.py）CompleteSKUnit 两分支「名义卷积核」消融。
# 训练流程与 checkpoints/old_data_supcon_compare_v3 一致：
#   使用 train_casgnet_contrastive_newdata.py（同 run_compare_old_data_supcon.sh / train_single_compare_model.sh）
#
# 用法（在 ModelsTotrain 根目录）:
#   bash scripts/run_starnetsk_sk_kernel_ablation_parallel.sh
#
# 与 v3 批量对比相同的环境变量（节选）:
#   DATA_DIR, GPU_LIST 或 GPUS, EPOCHS, BATCH_SIZE, NUM_WORKERS, SEED, AUGMENTATION, PRETRAINED_FLAG
#   GPUS=4,5,6,7,8,9  仅数字时自动转为 GPU_LIST=cuda:4,cuda:5,...（若已设 GPU_LIST 则优先用 GPU_LIST）
#   RUN_TAG（默认 starnetsk_sk_kernel_supcon）→ 输出 checkpoints/${RUN_TAG}/
#   RUN_LEAF_SUFFIX（默认 _ce_only）→ 子目录名 ${model}${RUN_LEAF_SUFFIX}
#   SKIP_REFRESH=1  跳过末尾 refresh（省时间；可自行再跑 refresh）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SCRIPT_DIR}"

DATA_DIR="${DATA_DIR:-old_data}"
# GPU_LIST 优先；否则用 GPUS（如 export GPUS=4,5,6,7,8,9）
if [[ -z "${GPU_LIST:-}" ]]; then
  if [[ -n "${GPUS:-}" ]]; then
    GPU_LIST=""
    IFS=',' read -r -a _GPUS_TOK <<< "${GPUS}"
    for _id in "${_GPUS_TOK[@]}"; do
      _id="$(echo "${_id}" | xargs)"
      [[ -z "${_id}" ]] && continue
      [[ -n "${GPU_LIST}" ]] && GPU_LIST+=","
      if [[ "${_id}" == cuda:* ]]; then
        GPU_LIST+="${_id}"
      else
        GPU_LIST+="cuda:${_id}"
      fi
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
RUN_TAG="${RUN_TAG:-starnetsk_sk_kernel_supcon}"
RUN_LEAF_SUFFIX="${RUN_LEAF_SUFFIX:-_ce_only}"
SKIP_REFRESH="${SKIP_REFRESH:-0}"

OUT_ROOT="${OUT_ROOT:-checkpoints/${RUN_TAG}}"
mkdir -p "${OUT_ROOT}"
LOG_DIR="${OUT_ROOT}/logs"
STATUS_DIR="${OUT_ROOT}/_status"
mkdir -p "${LOG_DIR}" "${STATUS_DIR}"
rm -f "${OUT_ROOT}/failed_runs.csv"

# --model 传给 train_casgnet_contrastive_newdata.py（与 classic_models / SUPPORTED_SUPCON_MODELS 一致）
declare -a MODELS=(
  "starnet_s1_sk13"
  "starnet_s1_sk15"
  "starnet_s1_sk17"
  "starnet_s1_sk19"
  "starnet_s1_sk35"
  "starnet_s1_sk37"
  "starnet_s1_sk39"
  "starnet_s1_sk57"
  "starnet_s1_sk59"
  "starnet_s1_sk79"
)

echo "== StarNet-SK kernel ablation (same train script as old_data_supcon_compare_v3) =="
echo "RUN_TAG=${RUN_TAG}  RUN_LEAF_SUFFIX=${RUN_LEAF_SUFFIX}"
echo "data_dir=${DATA_DIR}  out_root=${OUT_ROOT}"
echo "gpu_list=${GPU_LIST}  epochs=${EPOCHS}  batch_size=${BATCH_SIZE}"
echo

USE_FIXED_SPLIT=0
USE_TEST_DIR=0
if [[ -d "${DATA_DIR}/train" && -d "${DATA_DIR}/val" ]]; then
  USE_FIXED_SPLIT=1
  echo "split_mode=fixed (train/val)"
  if [[ -d "${DATA_DIR}/test" ]]; then
    USE_TEST_DIR=1
    echo "test_dir=${DATA_DIR}/test"
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
  echo "错误: 未解析到可用 GPU。请设置 GPU_LIST=cuda:4,cuda:5 或 GPUS=4,5,6,7"
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

  echo ">>> [START] model=${model} gpu=${gpu} -> ${run_dir}"
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

  if [[ ${code} -eq 0 ]]; then
    echo "OK,0" >"${status_file}"
    echo ">>> [DONE] model=${model}"
  else
    echo "FAILED,${code}" >"${status_file}"
    echo ">>> [FAIL] model=${model} exit=${code}  log=${log_file}"
  fi
}

declare -a SLOT_PIDS=()
for i in "${!GPUS[@]}"; do
  SLOT_PIDS[i]=""
done

total="${#MODELS[@]}"
next_idx=0
running=0

while [[ ${next_idx} -lt ${total} || ${running} -gt 0 ]]; do
  for i in "${!GPUS[@]}"; do
    pid="${SLOT_PIDS[i]}"
    gpu="${GPUS[i]}"

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

  if [[ ${running} -gt 0 ]]; then
    sleep 2
  fi
done

for model in "${MODELS[@]}"; do
  leaf="${model}${RUN_LEAF_SUFFIX}"
  status_file="${STATUS_DIR}/${leaf}.status"
  if [[ ! -f "${status_file}" ]]; then
    echo "${model},FAILED,NO_STATUS_FILE" >>"${OUT_ROOT}/failed_runs.csv"
    continue
  fi
  status_line="$(<"${status_file}")"
  state="${status_line%%,*}"
  code="${status_line#*,}"
  if [[ "${state}" != "OK" ]]; then
    echo "${model},FAILED,${code}" >>"${OUT_ROOT}/failed_runs.csv"
  fi
done

if [[ "${SKIP_REFRESH}" != "1" ]]; then
  echo "== refresh comparison（与 v3 批量末尾一致：bootstrap + comparison_summary*.csv/xlsx） =="
  python3 refresh_supcon_checkpoint_metrics.py --comparison-refresh-all "${OUT_ROOT}"
else
  echo "SKIP_REFRESH=1，跳过 refresh_supcon_checkpoint_metrics.py"
fi

echo "== 消融专用 Excel（SK 核组合 + val/test 指标） =="
python3 scripts/summarize_starnetsk_sk_ablation_excel.py \
  --experiment-root "${OUT_ROOT}" \
  --output "${OUT_ROOT}/summary_sk_kernel_ablation.xlsx"

echo
echo "完成。输出:"
echo "  - ${OUT_ROOT}/<model>${RUN_LEAF_SUFFIX}/（含 result_summary.json、best_auc_model.pth）"
echo "  - ${OUT_ROOT}/comparison_summary*.csv / comparison_summary.xlsx（若已 refresh）"
echo "  - ${OUT_ROOT}/summary_sk_kernel_ablation.xlsx"
echo "  - ${OUT_ROOT}/failed_runs.csv（若有失败）"
