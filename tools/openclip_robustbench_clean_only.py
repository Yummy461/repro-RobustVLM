#!/usr/bin/env python

import argparse
import json
import time
from pathlib import Path

import torch
from robustbench.data import load_clean_dataset
from robustbench.model_zoo.enums import BenchmarkDataset

from openclip_common import (
    MODEL_SPECS,
    build_text_features,
    find_normalize,
    load_openclip_model,
    logits_from_pixel_images,
    preprocess_without_normalize,
    read_imagenet_classnames,
    read_imagenet_templates,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Clean-only ImageNet eval using RobustBench sample selection.")
    parser.add_argument("--models", default="fare4")
    parser.add_argument("--data-dir", default="/root/autodl-tmp/data/imagenet")
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--template-mode", choices=["ensemble", "std"], default="ensemble")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--write-jsonl", default="")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def parse_models(value):
    if value == "all":
        return ["fare4", "tecoa4"]
    keys = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [key for key in keys if key not in MODEL_SPECS]
    if unknown:
        raise ValueError(f"unknown model keys: {unknown}")
    return keys


def evaluate(spec, args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, _, preprocess_val, tokenizer = load_openclip_model(spec, device="cpu")
    model.to(device)
    model.eval()
    transform = preprocess_without_normalize(preprocess_val)
    normalize = find_normalize(preprocess_val)
    if normalize is None:
        raise RuntimeError("Normalize transform not found in OpenCLIP preprocess")
    if args.template_mode == "ensemble":
        templates = read_imagenet_templates()
    else:
        templates = ["This is a photo of a {}"]
    classnames = read_imagenet_classnames()
    text_features = build_text_features(model, tokenizer, classnames, templates, device)

    x_test, y_test = load_clean_dataset(
        BenchmarkDataset("imagenet"),
        n_examples=args.n_samples,
        data_dir=args.data_dir,
        prepr=transform,
    )
    total = 0
    correct = 0
    start = time.time()
    with torch.no_grad():
        for start_idx in range(0, x_test.shape[0], args.batch_size):
            end_idx = min(start_idx + args.batch_size, x_test.shape[0])
            images = x_test[start_idx:end_idx].to(device)
            labels = y_test[start_idx:end_idx].to(device)
            logits = logits_from_pixel_images(model, normalize, text_features, images)
            pred = logits.argmax(dim=1)
            correct += int(pred.eq(labels).sum().item())
            total += int(labels.numel())
    runtime_s = time.time() - start
    clean_acc = 100.0 * correct / total if total else 0.0
    gap = clean_acc - spec.reference_clean if spec.reference_clean is not None else None
    return {
        "status": "passed" if gap is not None and abs(gap) <= 1.0 else "mismatch",
        "model_key": spec.key,
        "model_label": spec.label,
        "protocol": "robustbench_clean_only",
        "template_mode": args.template_mode,
        "data_dir": args.data_dir,
        "n_samples": total,
        "correct": correct,
        "clean_accuracy": clean_acc,
        "reference_clean": spec.reference_clean,
        "gap_pp": gap,
        "runtime_s": runtime_s,
        "note": "Uses RobustBench load_clean_dataset sample selection and no attack.",
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    records = [evaluate(MODEL_SPECS[key], args) for key in parse_models(args.models)]
    if args.write_jsonl:
        with Path(args.write_jsonl).open("w", encoding="utf-8") as handle:
            for item in records:
                handle.write(json.dumps(item, sort_keys=True) + "\n")
    for item in records:
        print("E0A_RB_CLEAN_ONLY_RESULT " + json.dumps(item, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
