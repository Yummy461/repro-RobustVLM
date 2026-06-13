#!/usr/bin/env python

import argparse
import json
import platform
import time
from pathlib import Path

import torch

from openclip_common import MODEL_SPECS, describe_model, load_openclip_model, parse_model_keys


def parse_args():
    parser = argparse.ArgumentParser(description="Load RobustVLM/OpenCLIP model variants and record public-safe metadata.")
    parser.add_argument("--models", default="all", help="all or comma-separated keys: vanilla,fare4,tecoa4")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--write-markdown", default="")
    parser.add_argument("--write-json", default="")
    return parser.parse_args()


def markdown_report(records):
    lines = [
        "# FARE4 / TeCoA4 ViT-B/16 Loading Notes",
        "",
        f"Run timestamp: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
        "This public-safe note records model loading, preprocessing, tokenizer, and lineage details for E0a calibration. It does not contain private route analysis.",
        "",
        "## Environment",
        "",
        f"- Python: `{platform.python_version()}`",
        f"- Torch: `{torch.__version__}`",
        f"- CUDA available: `{torch.cuda.is_available()}`",
    ]
    try:
        import torchvision
        lines.append(f"- Torchvision: `{torchvision.__version__}`")
    except Exception as exc:
        lines.append(f"- Torchvision: unavailable ({exc})")
    try:
        import open_clip
        version = getattr(open_clip, "__version__", "unknown")
        lines.append(f"- OpenCLIP: `{version}`")
    except Exception as exc:
        lines.append(f"- OpenCLIP: unavailable ({exc})")

    lines.extend([
        "",
        "## Model Loading Summary",
        "",
        "| Key | Model | Pretrained | HF Hub | Non-QuickGELU B/16 | Image size | Mean | Std | Tokenizer | logit_scale.exp |",
        "|---|---|---|---|---|---|---|---|---|---:|",
    ])
    for item in records:
        lines.append(
            f"| {item['key']} | `{item['model_name']}` | `{item['pretrained']}` | {item['hf_hub']} | "
            f"{item['maps_to_non_quickgelu_vit_b16']} | `{item['image_size']}` | `{item['mean']}` | "
            f"`{item['std']}` | `{item['tokenizer_type']}` | {item['logit_scale_exp']:.6f} |"
        )

    lines.extend([
        "",
        "## Preprocessing Details",
        "",
    ])
    for item in records:
        lines.extend([
            f"### {item['key']}",
            "",
            f"- Label: {item['label']}",
            f"- Reference clean accuracy: `{item['reference_clean']}` from {item['reference_source']}",
            f"- Matches LIFT CLIP normalization: `{item['matches_lift_normalization']}`",
            f"- LIFT mean/std: `{item['lift_mean']}` / `{item['lift_std']}`",
            f"- OpenCLIP val preprocess: `{json.dumps(item['preprocess_val'])}`",
            "",
        ])

    lines.extend([
        "## Public-Safe Risk Notes",
        "",
        "- Epsilon comparability depends on applying perturbations in pixel space after resize/crop and before normalization.",
        "- If any model reports a different mean/std or resize/crop behavior, clean and robust numbers must be interpreted with that preprocessing difference recorded.",
        "- Full ImageNet calibration remains pending until a valid extracted ImageNet validation directory is available.",
        "",
    ])
    return "\n".join(lines)


def main():
    args = parse_args()
    records = []
    for key in parse_model_keys(args.models):
        spec = MODEL_SPECS[key]
        print(f"[load] {key}: {spec.model_name} pretrained={spec.pretrained}", flush=True)
        model, preprocess_train, preprocess_val, tokenizer = load_openclip_model(spec, device=args.device)
        record = describe_model(spec, model, preprocess_train, preprocess_val, tokenizer)
        records.append(record)
        print("OPENCLIP_LOAD_RESULT " + json.dumps(record, sort_keys=True), flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.write_json:
        Path(args.write_json).write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.write_markdown:
        Path(args.write_markdown).write_text(markdown_report(records), encoding="utf-8")


if __name__ == "__main__":
    main()

