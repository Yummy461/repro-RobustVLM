#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="${ROBUSTVLM_REPO_DIR:-/root/repro-RobustVLM}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/robustvlm-openclip-py310/bin/python}"
DATA_ROOT="${ROBUSTVLM_DATA_ROOT:-/root/autodl-tmp/data}"
AUTODL_PUB_ROOT="${ROBUSTVLM_AUTODL_PUB_ROOT:-/root/autodl-pub}"
MODEL_KEY="${ROBUSTVLM_ATTACK_MODEL:-vanilla}"
SUBSET_SIZE="${ROBUSTVLM_ATTACK_SUBSET_SIZE:-512}"
BATCH_SIZE="${ROBUSTVLM_ATTACK_BATCH_SIZE:-32}"
STEPS="${ROBUSTVLM_ATTACK_STEPS:-5}"

export HF_HOME="${HF_HOME:-/root/autodl-tmp/model_cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-/root/autodl-tmp/model_cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/root/autodl-tmp/model_cache/xdg}"

cd "$REPO_DIR"

"$PYTHON_BIN" tools/openclip_attack_sanity.py \
  --model-key "$MODEL_KEY" \
  --data-root "$DATA_ROOT" \
  --autodl-pub-root "$AUTODL_PUB_ROOT" \
  --subset-size "$SUBSET_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --steps "$STEPS" \
  --write-markdown reproduce/attack_sanity_openclip.md \
  --write-json reproduce/attack_sanity_openclip.json
