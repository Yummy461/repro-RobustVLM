#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="${ROBUSTVLM_REPO_DIR:-/root/repro-RobustVLM}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/robustvlm-openclip-py310/bin/python}"
DATA_ROOT="${ROBUSTVLM_DATA_ROOT:-/root/autodl-tmp/data}"
AUTODL_PUB_ROOT="${ROBUSTVLM_AUTODL_PUB_ROOT:-/root/autodl-pub}"
OUTPUT_ROOT="${ROBUSTVLM_OUTPUT_ROOT:-/root/autodl-tmp/robustvlm_outputs/e0b_revised}"

MODE="subset"
ATTACK="pgd20"
for ((i = 1; i <= $#; i++)); do
  arg="${!i}"
  if [[ "$arg" == "--full" ]]; then
    MODE="full"
  fi
  if [[ "$arg" == "--attack" ]]; then
    next_index=$((i + 1))
    if [[ "$next_index" -le "$#" ]]; then
      ATTACK="${!next_index}"
    fi
  fi
done

mkdir -p "$OUTPUT_ROOT"

REPORT_PATH="reproduce/e0b_revised_subset_report.md"
PREFLIGHT_ARGS=()
if [[ "$MODE" == "full" ]]; then
  REPORT_PATH="reproduce/e0b_revised_full_cifar_report.md"
elif [[ "$ATTACK" == *"apgd"* ]]; then
  REPORT_PATH="reproduce/e0b_revised_apgd_subset_report.md"
else
  PREFLIGHT_ARGS=(--write-preflight reproduce/e0b_revised_preflight.md)
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${OUTPUT_ROOT}/e0b_revised_${MODE}_${ATTACK}_${TIMESTAMP}.log"

export HF_HOME="${HF_HOME:-/root/autodl-tmp/model_cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-/root/autodl-tmp/model_cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/root/autodl-tmp/model_cache/xdg}"

cd "$REPO_DIR"

echo "E0B_REVISED_LOG ${LOG_PATH}"

"$PYTHON_BIN" tools/openclip_e0b_attack_calibration.py \
  --data-root "$DATA_ROOT" \
  --autodl-pub-root "$AUTODL_PUB_ROOT" \
  --output-jsonl reproduce/e0b_revised_results.jsonl \
  --write-protocol reproduce/e0b_revised_attack_protocol.md \
  --write-report "$REPORT_PATH" \
  "${PREFLIGHT_ARGS[@]}" \
  "$@" 2>&1 | tee "$LOG_PATH"
