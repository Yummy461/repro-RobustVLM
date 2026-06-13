#!/usr/bin/env python

import argparse
import json
import os
from pathlib import Path

import torch

from openclip_common import MODEL_SPECS, describe_model, load_openclip_model, parse_model_keys


def count_parameters(module):
    return int(sum(param.numel() for param in module.parameters()))


def tensor_shape_from_state(state_dict, key):
    value = state_dict.get(key)
    if value is None:
        return None
    return list(value.shape)


def detect_visual_signature(model):
    visual = getattr(model, "visual", None)
    if visual is None:
        return {"status": "missing_visual"}
    state = visual.state_dict()
    resblocks = getattr(getattr(visual, "transformer", None), "resblocks", None)
    image_size = getattr(visual, "image_size", None)
    if isinstance(image_size, tuple):
        image_size = list(image_size)
    conv1_shape = tensor_shape_from_state(state, "conv1.weight")
    signature = {
        "visual_class": visual.__class__.__name__,
        "image_size": image_size,
        "conv1_weight_shape": conv1_shape,
        "projection_shape": tensor_shape_from_state(state, "proj"),
        "ln_post_weight_shape": tensor_shape_from_state(state, "ln_post.weight"),
        "resblock_count": len(resblocks) if resblocks is not None else None,
        "state_key_examples": list(state.keys())[:12],
    }
    signature["looks_like_vit_b16"] = (
        signature["visual_class"] == "VisionTransformer"
        and conv1_shape == [768, 3, 16, 16]
        and signature["resblock_count"] == 12
    )
    signature["looks_like_convnext"] = "convnext" in signature["visual_class"].lower()
    return signature


def detect_text_signature(model):
    state = model.state_dict()
    transformer = getattr(model, "transformer", None)
    resblocks = getattr(transformer, "resblocks", None)
    return {
        "text_tower_class": transformer.__class__.__name__ if transformer is not None else None,
        "token_embedding_shape": tensor_shape_from_state(state, "token_embedding.weight"),
        "text_projection_shape": tensor_shape_from_state(state, "text_projection"),
        "text_resblock_count": len(resblocks) if resblocks is not None else None,
    }


def repo_id_from_model_name(model_name):
    if model_name.startswith("hf-hub:"):
        return model_name.split(":", 1)[1]
    return None


def scan_hf_cache(repo_id):
    if not repo_id:
        return None
    try:
        from huggingface_hub import scan_cache_dir
    except Exception as exc:
        return {"status": "unavailable", "error": repr(exc)}

    cache_dir = os.environ.get("HUGGINGFACE_HUB_CACHE") or os.environ.get("HF_HOME")
    try:
        info = scan_cache_dir(cache_dir=cache_dir) if cache_dir else scan_cache_dir()
    except Exception as exc:
        return {"status": "scan_failed", "cache_dir": cache_dir, "error": repr(exc)}

    matches = []
    for repo in getattr(info, "repos", []):
        if getattr(repo, "repo_id", None) != repo_id:
            continue
        revisions = []
        for rev in getattr(repo, "revisions", []):
            files = []
            for file_info in getattr(rev, "files", []):
                file_name = getattr(file_info, "file_name", None)
                if file_name:
                    files.append({
                        "file_name": file_name,
                        "size_on_disk": getattr(file_info, "size_on_disk", None),
                    })
            revisions.append({
                "commit_hash": getattr(rev, "commit_hash", None),
                "refs": sorted(str(item) for item in getattr(rev, "refs", [])),
                "snapshot_path": str(getattr(rev, "snapshot_path", "")),
                "files": sorted(files, key=lambda item: item["file_name"]),
            })
        matches.append({
            "repo_id": getattr(repo, "repo_id", None),
            "repo_type": getattr(repo, "repo_type", None),
            "repo_path": str(getattr(repo, "repo_path", "")),
            "revisions": revisions,
        })
    return {
        "status": "found" if matches else "not_found",
        "cache_dir": cache_dir,
        "repos": matches,
    }


def diagnose_model(key):
    spec = MODEL_SPECS[key]
    model, preprocess_train, preprocess_val, tokenizer = load_openclip_model(spec, device="cpu")
    metadata = describe_model(spec, model, preprocess_train, preprocess_val, tokenizer)
    visual = getattr(model, "visual", None)
    record = {
        "model_key": key,
        "model_label": spec.label,
        "open_clip_model_name": spec.model_name,
        "pretrained": spec.pretrained,
        "hf_repo_id": repo_id_from_model_name(spec.model_name),
        "metadata": metadata,
        "visual_signature": detect_visual_signature(model),
        "text_signature": detect_text_signature(model),
        "parameter_counts": {
            "total": count_parameters(model),
            "visual": count_parameters(visual) if visual is not None else None,
        },
    }
    record["hf_cache"] = scan_hf_cache(record["hf_repo_id"])
    return record


def markdown(records):
    lines = [
        "# OpenCLIP Model Identity Diagnostics",
        "",
        "This public-safe diagnostic records model identity and cache metadata for E0a mismatch analysis. It does not contain private route analysis.",
        "",
        "| Model | OpenCLIP model name | HF repo | Visual class | ViT-B/16 signature | ConvNeXt signature | Visual params | Total params | Image size | Logit scale exp |",
        "|---|---|---|---|---:|---:|---:|---:|---|---:|",
    ]
    for item in records:
        visual = item["visual_signature"]
        metadata = item["metadata"]
        counts = item["parameter_counts"]
        lines.append(
            f"| {item['model_key']} | `{item['open_clip_model_name']}` | "
            f"`{item['hf_repo_id'] or ''}` | `{visual.get('visual_class')}` | "
            f"{visual.get('looks_like_vit_b16')} | {visual.get('looks_like_convnext')} | "
            f"{counts['visual']} | {counts['total']} | `{metadata.get('image_size')}` | "
            f"{metadata.get('logit_scale_exp'):.6f} |"
        )
    lines.extend(["", "## Cache Revisions", ""])
    for item in records:
        lines.append(f"### {item['model_key']}")
        cache = item.get("hf_cache") or {}
        lines.append(f"- HF cache status: `{cache.get('status')}`.")
        lines.append(f"- HF cache dir: `{cache.get('cache_dir')}`.")
        for repo in cache.get("repos", []):
            for rev in repo.get("revisions", []):
                lines.append(f"- revision: `{rev.get('commit_hash')}`, refs: `{rev.get('refs')}`")
                file_names = [file_info["file_name"] for file_info in rev.get("files", [])]
                lines.append(f"- files: `{file_names}`")
        lines.append("")
    lines.extend(["## Raw JSON", "", "```json", json.dumps(records, indent=2, sort_keys=True), "```", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Inspect OpenCLIP model identity and cached HF snapshot metadata.")
    parser.add_argument("--models", default="all")
    parser.add_argument("--write-json", default="")
    parser.add_argument("--write-markdown", default="")
    args = parser.parse_args()

    records = [diagnose_model(key) for key in parse_model_keys(args.models)]
    if args.write_json:
        Path(args.write_json).write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.write_markdown:
        Path(args.write_markdown).write_text(markdown(records), encoding="utf-8")
    print(json.dumps(records, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
