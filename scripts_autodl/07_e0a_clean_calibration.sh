#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="${ROBUSTVLM_REPO_DIR:-/root/repro-RobustVLM}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/robustvlm-openclip-py310/bin/python}"
DATA_ROOT="${ROBUSTVLM_DATA_ROOT:-/root/autodl-tmp/data}"
IMAGENET_VAL="${ROBUSTVLM_IMAGENET_VAL:-}"
MODELS="${ROBUSTVLM_E0A_MODELS:-vanilla}"
SUBSET_SIZE="${ROBUSTVLM_E0A_SUBSET_SIZE:-64}"
BATCH_SIZE="${ROBUSTVLM_E0A_BATCH_SIZE:-16}"
NUM_WORKERS="${ROBUSTVLM_E0A_NUM_WORKERS:-4}"
TEMPLATE_MODE="${ROBUSTVLM_E0A_TEMPLATE_MODE:-ensemble}"
FULL=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --full)
      FULL=1
      shift
      ;;
    --model|--models)
      MODELS="${2:?missing value for $1}"
      shift 2
      ;;
    --max-samples|--subset-size)
      SUBSET_SIZE="${2:?missing value for $1}"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="${2:?missing value for $1}"
      shift 2
      ;;
    --num-workers)
      NUM_WORKERS="${2:?missing value for $1}"
      shift 2
      ;;
    --imagenet-val-dir|--imagenet-val)
      IMAGENET_VAL="${2:?missing value for $1}"
      shift 2
      ;;
    --template-mode)
      TEMPLATE_MODE="${2:?missing value for $1}"
      shift 2
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

export HF_HOME="${HF_HOME:-/root/autodl-tmp/model_cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-/root/autodl-tmp/model_cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/root/autodl-tmp/model_cache/xdg}"

cd "$REPO_DIR"

CMD=(
  "$PYTHON_BIN" tools/openclip_clean_calibration.py
  --models "$MODELS"
  --data-root "$DATA_ROOT"
  --subset-size "$SUBSET_SIZE"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --template-mode "$TEMPLATE_MODE"
  --write-markdown reproduce/e0a_report.md
  --write-jsonl reproduce/e0a_results.jsonl
)

if [ -n "$IMAGENET_VAL" ]; then
  CMD+=(--imagenet-val "$IMAGENET_VAL")
fi

if [ "$FULL" = "1" ]; then
  CMD+=(--full)
else
  echo "Running E0a dry-run/subset mode. Use --full only after ImageNet val is verified."
fi

"${CMD[@]}"
