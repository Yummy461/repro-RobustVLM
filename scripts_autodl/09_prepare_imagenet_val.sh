#!/usr/bin/env bash

set -euo pipefail

DATA_ROOT="${ROBUSTVLM_DATA_ROOT:-/root/autodl-tmp/data}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/robustvlm-openclip-py310/bin/python}"
AUTODL_PUB_ROOT="${ROBUSTVLM_AUTODL_PUB_ROOT:-/root/autodl-pub}"
IMAGENET_VAL_DIR="${IMAGENET_VAL_DIR:-${DATA_ROOT}/imagenet/val}"
IMAGENET_VAL_SOURCE="${IMAGENET_VAL_SOURCE:-${AUTODL_PUB_ROOT}/ImageNet/ILSVRC2012}"
IMAGENET_TAR_PATH="${IMAGENET_TAR_PATH:-${IMAGENET_VAL_SOURCE}/ILSVRC2012_img_val.tar}"
IMAGENET_DEVKIT_PATH="${IMAGENET_DEVKIT_PATH:-${IMAGENET_VAL_SOURCE}/ILSVRC2012_devkit_t12.tar.gz}"
IMAGENET_STAGING_DIR="${IMAGENET_STAGING_DIR:-${DATA_ROOT}/imagenet/val_raw}"
DEVKIT_STAGING_DIR="${DEVKIT_STAGING_DIR:-${DATA_ROOT}/imagenet/devkit}"
EXTRACT=0
PREPARE=0

usage() {
  cat <<'USAGE'
Usage: scripts_autodl/09_prepare_imagenet_val.sh [--check] [--extract] [--prepare]

Default behavior is check-only. This script never downloads ImageNet and never
deletes source tar files. Long operations require explicit flags.

Environment variables:
  IMAGENET_VAL_DIR       Target ImageFolder directory, default /root/autodl-tmp/data/imagenet/val
  IMAGENET_TAR_PATH      Source val tar, default /root/autodl-pub/ImageNet/ILSVRC2012/ILSVRC2012_img_val.tar
  IMAGENET_DEVKIT_PATH   Devkit tar with validation ground truth and meta.mat
  IMAGENET_STAGING_DIR   Flat extracted val staging directory
USAGE
}

is_imagefolder() {
  local path="$1"
  [ -d "$path" ] || return 1
  local classes
  classes="$(find "$path" -mindepth 1 -maxdepth 1 -type d | wc -l)"
  [ "$classes" -ge 1000 ] || return 1
  local sample
  sample="$(find "$path" -mindepth 2 -maxdepth 2 -type f \( -iname '*.JPEG' -o -iname '*.jpg' -o -iname '*.png' \) -print -quit)"
  [ -n "$sample" ]
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --check)
      shift
      ;;
    --extract)
      EXTRACT=1
      shift
      ;;
    --prepare)
      PREPARE=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

echo "IMAGENET_VAL_DIR=$IMAGENET_VAL_DIR"
echo "IMAGENET_TAR_PATH=$IMAGENET_TAR_PATH"
echo "IMAGENET_DEVKIT_PATH=$IMAGENET_DEVKIT_PATH"
echo "IMAGENET_STAGING_DIR=$IMAGENET_STAGING_DIR"
echo "PYTHON_BIN=$PYTHON_BIN"

if is_imagefolder "$IMAGENET_VAL_DIR"; then
  echo "STATUS=ready_imagefolder"
  exit 0
fi

if [ ! -f "$IMAGENET_TAR_PATH" ]; then
  echo "STATUS=missing_val_tar"
  echo "ERROR: ImageNet val tar not found: $IMAGENET_TAR_PATH" >&2
  exit 3
fi

if [ ! -f "$IMAGENET_DEVKIT_PATH" ]; then
  echo "STATUS=missing_devkit"
  echo "ERROR: ImageNet devkit tar not found: $IMAGENET_DEVKIT_PATH" >&2
  exit 3
fi

if [ "$EXTRACT" = "0" ] && [ "$PREPARE" = "0" ]; then
  echo "STATUS=needs_extract_or_prepare"
  echo "Run with --extract to extract flat val images to staging."
  echo "Run with --prepare after extraction to build ImageFolder class directories."
  exit 0
fi

if [ "$EXTRACT" = "1" ]; then
  mkdir -p "$IMAGENET_STAGING_DIR"
  existing="$(find "$IMAGENET_STAGING_DIR" -maxdepth 1 -type f -iname '*.JPEG' | wc -l)"
  if [ "$existing" -ge 50000 ]; then
    echo "STATUS=staging_already_extracted count=$existing"
  else
    echo "STATUS=extracting_val_tar"
    tar -xf "$IMAGENET_TAR_PATH" -C "$IMAGENET_STAGING_DIR"
    count="$(find "$IMAGENET_STAGING_DIR" -maxdepth 1 -type f -iname '*.JPEG' | wc -l)"
    echo "STATUS=extracted count=$count"
  fi
fi

if [ "$PREPARE" = "1" ]; then
  mkdir -p "$DEVKIT_STAGING_DIR"
  if [ ! -f "$DEVKIT_STAGING_DIR/ILSVRC2012_devkit_t12/data/meta.mat" ]; then
    tar -xzf "$IMAGENET_DEVKIT_PATH" -C "$DEVKIT_STAGING_DIR"
  fi
  "$PYTHON_BIN" - "$IMAGENET_STAGING_DIR" "$IMAGENET_VAL_DIR" "$DEVKIT_STAGING_DIR/ILSVRC2012_devkit_t12/data/meta.mat" "$DEVKIT_STAGING_DIR/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt" <<'PY'
import os
import re
import shutil
import sys
from pathlib import Path

from scipy.io import loadmat

staging = Path(sys.argv[1])
target = Path(sys.argv[2])
meta_path = Path(sys.argv[3])
gt_path = Path(sys.argv[4])

if not staging.exists():
    raise SystemExit(f"staging directory missing: {staging}")

meta = loadmat(meta_path, squeeze_me=True)["synsets"]
idx_to_wnid = {}
for item in meta:
    ilsvrc_id = int(item["ILSVRC2012_ID"])
    if 1 <= ilsvrc_id <= 1000:
        idx_to_wnid[ilsvrc_id] = str(item["WNID"])
if len(idx_to_wnid) != 1000:
    raise SystemExit(f"expected 1000 validation synsets, got {len(idx_to_wnid)}")

labels = [int(line.strip()) for line in gt_path.read_text().splitlines() if line.strip()]
if len(labels) != 50000:
    raise SystemExit(f"expected 50000 validation labels, got {len(labels)}")

target.mkdir(parents=True, exist_ok=True)
linked = 0
missing = 0
for path in sorted(staging.glob("ILSVRC2012_val_*.JPEG")):
    match = re.search(r"_(\d+)\.JPEG$", path.name)
    if not match:
        continue
    image_index = int(match.group(1))
    if image_index < 1 or image_index > len(labels):
        missing += 1
        continue
    wnid = idx_to_wnid[labels[image_index - 1]]
    class_dir = target / wnid
    class_dir.mkdir(parents=True, exist_ok=True)
    dst = class_dir / path.name
    if dst.exists():
        continue
    try:
        os.link(path, dst)
    except OSError:
        shutil.copy2(path, dst)
    linked += 1

classes = [p for p in target.iterdir() if p.is_dir()]
images = list(target.glob("*/*.JPEG"))
print(f"STATUS=prepared classes={len(classes)} images={len(images)} linked_or_copied={linked} missing={missing}")
if len(classes) != 1000 or len(images) != 50000:
    raise SystemExit("prepared ImageFolder is incomplete")
PY
fi

if is_imagefolder "$IMAGENET_VAL_DIR"; then
  echo "STATUS=ready_imagefolder"
else
  echo "STATUS=not_ready"
  exit 4
fi
