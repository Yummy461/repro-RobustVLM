#!/usr/bin/env python

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

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
    parser = argparse.ArgumentParser(description="Evaluate a vanilla text-tower + robust visual transplant variant.")
    parser.add_argument("--visual-source", choices=["fare4", "tecoa4"], default="fare4")
    parser.add_argument("--imagenet-val", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--subset-size", type=int, default=64)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--write-json", default="")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    base_spec = MODEL_SPECS["vanilla"]
    visual_spec = MODEL_SPECS[args.visual_source]
    base_model, _, base_preprocess_val, base_tokenizer = load_openclip_model(base_spec, device="cpu")
    visual_model, _, _, _ = load_openclip_model(visual_spec, device="cpu")
    base_model.visual.load_state_dict(visual_model.visual.state_dict())
    base_model.to(device)
    base_model.eval()

    normalize = find_normalize(base_preprocess_val)
    if normalize is None:
        raise RuntimeError("Normalize transform not found in vanilla OpenCLIP preprocess")
    transform = preprocess_without_normalize(base_preprocess_val)
    classnames = read_imagenet_classnames()
    templates = read_imagenet_templates()
    text_features = build_text_features(base_model, base_tokenizer, classnames, templates, device)

    dataset = ImageFolder(args.imagenet_val, transform=transform)
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
            logits = logits_from_pixel_images(base_model, normalize, text_features, images)
            pred = logits.argmax(dim=1)
            correct += int(pred.eq(labels).sum().item())
            total += int(labels.numel())
    runtime_s = time.time() - start
    clean_acc = 100.0 * correct / total if total else 0.0
    record = {
        "status": "completed",
        "variant": f"vanilla_text_logit_scale_plus_{args.visual_source}_visual",
        "visual_source": args.visual_source,
        "base_text_model": "vanilla LAION2B ViT-B/16",
        "imagenet_val": args.imagenet_val,
        "mode": "full" if args.full else "dry_run_subset",
        "num_samples": total,
        "correct": correct,
        "clean_accuracy": clean_acc,
        "runtime_s": runtime_s,
        "interpretation": "If this matches direct hf-hub robust-model clean accuracy, the text tower and logit scale are unlikely to explain the mismatch.",
    }
    if args.write_json:
        Path(args.write_json).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("E0A_VISUAL_TRANSPLANT_RESULT " + json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
