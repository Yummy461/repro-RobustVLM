#!/usr/bin/env python

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from openclip_common import (
    MODEL_SPECS,
    build_text_features,
    describe_model,
    load_openclip_model,
    locate_imagenet_val,
    logits_from_pixel_images,
    parse_model_keys,
    preprocess_without_normalize,
    read_imagenet_classnames,
    read_imagenet_templates,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Small OpenCLIP ImageNet clean calibration runner.")
    parser.add_argument("--models", default="vanilla", help="all or comma-separated keys")
    parser.add_argument("--data-root", default="/root/autodl-tmp/data")
    parser.add_argument("--imagenet-val", default="")
    parser.add_argument("--subset-size", "--max-samples", dest="subset_size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--write-markdown", default="")
    parser.add_argument("--write-jsonl", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--template-mode", choices=["ensemble", "std"], default="ensemble")
    return parser.parse_args()


def evaluate_model(spec, args, imagenet_val):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, preprocess_train, preprocess_val, tokenizer = load_openclip_model(spec, device="cpu")
    metadata = describe_model(spec, model, preprocess_train, preprocess_val, tokenizer)
    model.to(device)
    model.eval()

    classnames = read_imagenet_classnames()
    if args.template_mode == "ensemble":
        templates = read_imagenet_templates()
        prompt_template_status = "Uses CLIP_eval/zeroshot-templates.json imagenet1k ensemble, matching RobustVLM clip_robustbench.py template=ensemble path."
    else:
        templates = ["This is a photo of a {}"]
        prompt_template_status = "Uses RobustVLM std template: This is a photo of a {}."
    text_features = build_text_features(model, tokenizer, classnames, templates, device)
    transform = preprocess_without_normalize(preprocess_val)
    normalize = None
    for item in getattr(preprocess_val, "transforms", [preprocess_val]):
        if item.__class__.__name__ == "Normalize":
            normalize = item
    if normalize is None:
        raise RuntimeError("Normalize transform not found in OpenCLIP preprocess")

    dataset = ImageFolder(imagenet_val, transform=transform)
    class_count = len(dataset.classes)
    if class_count != 1000:
        raise RuntimeError(f"ImageNet val ImageFolder must contain 1000 class directories, found {class_count}: {imagenet_val}")
    total_requested = len(dataset) if args.full else min(args.subset_size, len(dataset))
    subset = Subset(dataset, list(range(total_requested)))
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    correct = 0
    total = 0
    start = time.time()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = logits_from_pixel_images(model, normalize, text_features, images)
            pred = logits.argmax(dim=1)
            correct += int(pred.eq(labels).sum().item())
            total += int(labels.numel())
    runtime_s = time.time() - start
    clean_acc = 100.0 * correct / total if total else 0.0
    gap = clean_acc - spec.reference_clean if spec.reference_clean is not None else None
    mismatch = bool(args.full and gap is not None and abs(gap) > 1.0)

    return {
        "status": "mismatch" if mismatch else ("passed" if total > 0 else "partial"),
        "model_key": spec.key,
        "model_label": spec.label,
        "mode": "full" if args.full else "dry_run_subset",
        "imagenet_val": imagenet_val,
        "num_samples": total,
        "num_classes": class_count,
        "correct": correct,
        "clean_accuracy": clean_acc,
        "reference_clean": spec.reference_clean,
        "gap_pp": gap,
        "runtime_s": runtime_s,
        "metadata": metadata,
        "template_mode": args.template_mode,
        "prompt_template_status": prompt_template_status,
        "class_order_risk": "ImageFolder class order must match ImageNet class index order for formal calibration.",
    }


def markdown_report(results, blocker=None):
    lines = [
        "# E0a Clean Calibration Report",
        "",
        "This public-safe report records E0a clean calibration status for OpenCLIP LAION2B / FARE4 / TeCoA4. Full ImageNet calibration must use an extracted ImageNet validation directory and `--full`.",
        "",
        "## Reference Targets",
        "",
        "| Model | Reference clean top-1 | Source |",
        "|---|---:|---|",
        "| vanilla LAION2B ViT-B/16 | 70.23 | OpenCLIP official classification CSV |",
        "| FARE4 ViT-B/16 LAION2B | 56.6 | RobustVLM README Additional Models table |",
        "| TeCoA4 ViT-B/16 LAION2B | 51.5 | RobustVLM README Additional Models table |",
        "",
    ]
    if blocker:
        lines.extend([
            "## Current Status",
            "",
            f"- Status: partial / blocked",
            f"- Blocker: {blocker}",
            "- Full clean calibration was not run.",
            "",
        ])
    if results:
        lines.extend([
            "## Measured Results",
            "",
            "| Model | Mode | Samples | Clean acc | Reference | Gap pp | Runtime | Status |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ])
        for item in results:
            gap = "" if item["gap_pp"] is None else f"{item['gap_pp']:.4f}"
            lines.append(
                f"| {item['model_key']} | {item['mode']} | {item['num_samples']} | "
                f"{item['clean_accuracy']:.4f} | {item['reference_clean']} | {gap} | "
                f"{item['runtime_s']:.2f}s | {item['status']} |"
            )
    lines.extend([
        "",
        "## Stop Condition",
        "",
        "- If any full clean calibration differs from its reference by more than 1.0 percentage point, stop before full attacks and investigate prompt templates, preprocessing, weight loading, tokenizer, and class ordering.",
        "",
    ])
    mismatches = [item for item in results if item.get("status") == "mismatch"]
    if mismatches:
        lines.extend([
            "## Mismatch Report",
            "",
            "At least one full clean calibration result differs from its reference by more than 1.0 percentage point.",
            "",
            "Suspect order:",
            "",
            "1. prompt templates;",
            "2. preprocessing;",
            "3. weight loading;",
            "4. tokenizer;",
            "5. class ordering.",
            "",
        ])
    return "\n".join(lines)


def main():
    args = parse_args()
    set_seed(args.seed)
    imagenet_val = locate_imagenet_val(args.data_root, args.imagenet_val or None)
    results = []
    blocker = None
    if imagenet_val is None:
        blocker = f"no extracted ImageNet val ImageFolder found under {args.data_root}"
    else:
        for key in parse_model_keys(args.models):
            results.append(evaluate_model(MODEL_SPECS[key], args, imagenet_val))
            print("E0A_CLEAN_RESULT " + json.dumps(results[-1], sort_keys=True), flush=True)
            if results[-1].get("status") == "mismatch":
                break

    if args.write_jsonl:
        with Path(args.write_jsonl).open("w", encoding="utf-8") as handle:
            for item in results:
                handle.write(json.dumps(item, sort_keys=True) + "\n")
            if blocker:
                handle.write(json.dumps({"status": "partial", "blocker": blocker}, sort_keys=True) + "\n")
    if args.write_markdown:
        Path(args.write_markdown).write_text(markdown_report(results, blocker), encoding="utf-8")
    if blocker:
        print("E0A_CLEAN_RESULT " + json.dumps({"status": "partial", "blocker": blocker}, sort_keys=True), flush=True)
    if any(item.get("status") == "mismatch" for item in results):
        sys.exit(3)


if __name__ == "__main__":
    main()
