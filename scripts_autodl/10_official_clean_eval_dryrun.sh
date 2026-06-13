#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="${ROBUSTVLM_REPO_DIR:-/root/repro-RobustVLM}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/robustvlm-openclip-py310/bin/python}"
DATASET_ROOT="${ROBUSTVLM_IMAGENET_ROOT:-/root/autodl-tmp/data/imagenet}"
OUTPUT_DIR="${ROBUSTVLM_OFFICIAL_EVAL_OUTPUT_DIR:-${REPO_DIR}/reproduce}"
MODEL_KEY="${ROBUSTVLM_OFFICIAL_EVAL_MODEL:-fare4}"
N_SAMPLES="${ROBUSTVLM_OFFICIAL_EVAL_N_SAMPLES:-11}"
BATCH_SIZE="${ROBUSTVLM_OFFICIAL_EVAL_BATCH_SIZE:-64}"
NUM_WORKERS="${ROBUSTVLM_OFFICIAL_EVAL_NUM_WORKERS:-4}"

case "$MODEL_KEY" in
  fare4)
    MODEL_NAME="hf-hub:chs20/FARE4-ViT-B-16-laion2B-s34B-b88K"
    ;;
  tecoa4)
    MODEL_NAME="hf-hub:chs20/TeCoA4-ViT-B-16-laion2B-s34B-b88K"
    ;;
  *)
    echo "ERROR: unsupported ROBUSTVLM_OFFICIAL_EVAL_MODEL: $MODEL_KEY" >&2
    exit 2
    ;;
esac

export HF_HOME="${HF_HOME:-/root/autodl-tmp/model_cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-/root/autodl-tmp/model_cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/root/autodl-tmp/model_cache/xdg}"
export PYTHONPATH="${REPO_DIR}/CLIP_benchmark:${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "$OUTPUT_DIR"
cd "$REPO_DIR"

"$PYTHON_BIN" -m clip_benchmark.cli eval \
  --dataset imagenet1k-unverified \
  --dataset_root "$DATASET_ROOT" \
  --split val \
  --task zeroshot_classification \
  --model "$MODEL_NAME" \
  --pretrained "" \
  --attack none \
  --batch_size "$BATCH_SIZE" \
  --num_workers "$NUM_WORKERS" \
  --n_samples "$N_SAMPLES" \
  --output "${OUTPUT_DIR}/e0a_official_clean_eval_${MODEL_KEY}_${N_SAMPLES}.json"
