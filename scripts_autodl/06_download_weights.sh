#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="${ROBUSTVLM_REPO_DIR:-/root/repro-RobustVLM}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/robustvlm-openclip-py310/bin/python}"

export HF_HOME="${HF_HOME:-/root/autodl-tmp/model_cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-/root/autodl-tmp/model_cache/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/root/autodl-tmp/model_cache/xdg}"

mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TORCH_HOME" "$XDG_CACHE_HOME"
cd "$REPO_DIR"

echo "Loading OpenCLIP / RobustVLM weights into data-disk cache"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "HF_HOME=$HF_HOME"
echo "TORCH_HOME=$TORCH_HOME"

"$PYTHON_BIN" tools/openclip_loader.py \
  --models all \
  --device cpu \
  --write-markdown reproduce/fare4_b16_loading_notes.md \
  --write-json reproduce/fare4_b16_loading_notes.json
