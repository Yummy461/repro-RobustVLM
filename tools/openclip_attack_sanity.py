#!/usr/bin/env python

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from openclip_common import (
    MODEL_SPECS,
    CIFAR100PythonTest,
    build_text_features,
    ensure_cifar100_from_autodl,
    load_openclip_model,
    logits_from_pixel_images,
    pixel_preprocess_for_size,
    read_cifar100_classnames,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="OpenCLIP CIFAR-100 PGD sanity check.")
    parser.add_argument("--model-key", default="vanilla", choices=sorted(MODEL_SPECS.keys()))
    parser.add_argument("--data-root", default="/root/autodl-tmp/data")
    parser.add_argument("--autodl-pub-root", default="/root/autodl-pub")
    parser.add_argument("--subset-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--write-markdown", default="")
    parser.add_argument("--write-json", default="")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def clamp_pixel(x):
    return x.clamp(0.0, 1.0)


def project_linf(x_adv, x_orig, eps):
    return torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps).clamp(0.0, 1.0)


def pgd_attack(model, normalize, text_features, images, labels, eps, steps, step_size, random_start):
    x_orig = images.detach()
    if eps == 0:
        x_adv = x_orig.clone()
    elif random_start:
        x_adv = clamp_pixel(x_orig + torch.empty_like(x_orig).uniform_(-eps, eps))
    else:
        x_adv = x_orig.clone()
    gradient_ok = False
    max_abs_grad = 0.0
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        logits = logits_from_pixel_images(model, normalize, text_features, x_adv)
        loss = F.cross_entropy(logits.float(), labels)
        grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
        max_abs_grad = max(max_abs_grad, float(grad.detach().abs().max().item()))
        gradient_ok = gradient_ok or (torch.isfinite(grad).all().item() and max_abs_grad > 0.0)
        if eps > 0:
            x_adv = x_adv.detach() + step_size * grad.detach().sign()
            x_adv = project_linf(x_adv, x_orig, eps)
        else:
            x_adv = x_orig.clone()
    return x_adv.detach(), gradient_ok, max_abs_grad


def evaluate_images(model, normalize, text_features, images, labels):
    with torch.no_grad():
        logits = logits_from_pixel_images(model, normalize, text_features, images)
        pred = logits.argmax(dim=1)
        return int(pred.eq(labels).sum().item())


def run_case(model, normalize, text_features, loader, device, eps, steps):
    step_size = 0.0 if eps == 0 else eps / 2.0
    random_start = eps > 0
    clean_correct = 0
    adv_correct = 0
    total = 0
    gradient_ok = eps == 0
    max_abs_grad = 0.0
    start = time.time()
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        clean_correct += evaluate_images(model, normalize, text_features, images, labels)
        adv_images, grad_ok, grad_max = pgd_attack(
            model, normalize, text_features, images, labels, eps, steps, step_size, random_start
        )
        gradient_ok = gradient_ok or grad_ok
        max_abs_grad = max(max_abs_grad, grad_max)
        adv_correct += evaluate_images(model, normalize, text_features, adv_images, labels)
        total += int(labels.numel())
    runtime_s = time.time() - start
    return {
        "epsilon_pixel_domain": eps,
        "epsilon_domain": "pixel_after_resize_before_normalization",
        "steps": steps,
        "step_size_pixel_domain": step_size,
        "random_start": random_start,
        "subset_size": total,
        "clean_correct": clean_correct,
        "adv_correct": adv_correct,
        "clean_accuracy": 100.0 * clean_correct / total if total else 0.0,
        "robust_accuracy": 100.0 * adv_correct / total if total else 0.0,
        "gradient_ok": bool(gradient_ok),
        "max_abs_input_grad": max_abs_grad,
        "runtime_s": runtime_s,
    }


def markdown_report(model_key, cifar_status, results, monotonic_ok):
    collapse_note = ""
    if len(results) > 1 and results[1]["robust_accuracy"] <= 0.0:
        collapse_note = (
            "- Observed eps 1/255 robust accuracy is `0.0`; keep this as an attack-interface "
            "stress signal only. Verify loss, labels, preprocessing, and official ImageNet "
            "protocol before treating it as a formal robustness number."
        )
    lines = [
        "# OpenCLIP Attack Sanity Report",
        "",
        "This public-safe report records a small CIFAR-100 attack-pipeline sanity check. It is not a formal robustness result.",
        "",
        f"- Model key: `{model_key}`",
        f"- CIFAR-100 status: `{json.dumps(cifar_status, sort_keys=True)}`",
        "- Dataset: CIFAR-100 test subset",
        "- Text side: fixed precomputed class features",
        "- Epsilon domain: pixel space after resize/crop and before normalization",
        "- Attack: untargeted image-only L_inf PGD",
        "",
        "| Epsilon | Steps | Step size | Clean acc | Robust acc | Grad ok | Runtime |",
        "|---:|---:|---:|---:|---:|---|---:|",
    ]
    for item in results:
        lines.append(
            f"| {item['epsilon_pixel_domain']:.8f} | {item['steps']} | {item['step_size_pixel_domain']:.8f} | "
            f"{item['clean_accuracy']:.4f} | {item['robust_accuracy']:.4f} | {item['gradient_ok']} | {item['runtime_s']:.2f}s |"
        )
    lines.extend([
        "",
        f"- Monotonic non-increasing robust accuracy: `{monotonic_ok}`",
        "- If epsilon 0 does not exactly match clean accuracy, treat the attack pipeline as invalid.",
        "- If eps 4/255 does not substantially reduce vanilla accuracy, inspect preprocessing, gradients, class order, and text features before running larger attacks.",
        "",
    ])
    if collapse_note:
        lines.insert(-1, collapse_note)
    return "\n".join(lines)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cifar_status = ensure_cifar100_from_autodl(args.data_root, args.autodl_pub_root)
    transform = pixel_preprocess_for_size(224)
    dataset = CIFAR100PythonTest(args.data_root, transform)
    subset = Subset(dataset, list(range(min(args.subset_size, len(dataset)))))
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    spec = MODEL_SPECS[args.model_key]
    model, _, preprocess_val, tokenizer = load_openclip_model(spec, device="cpu")
    model.to(device)
    model.eval()
    normalize = None
    for item in getattr(preprocess_val, "transforms", [preprocess_val]):
        if item.__class__.__name__ == "Normalize":
            normalize = item
    if normalize is None:
        raise RuntimeError("Normalize transform not found in OpenCLIP preprocess")
    classnames = read_cifar100_classnames(args.data_root)
    templates = ["a photo of a {c}."]
    text_features = build_text_features(model, tokenizer, classnames, templates, device)

    results = []
    for eps in [0.0, 1.0 / 255.0, 2.0 / 255.0, 4.0 / 255.0]:
        item = run_case(model, normalize, text_features, loader, device, eps, args.steps)
        item["model_key"] = args.model_key
        results.append(item)
        print("ATTACK_SANITY_RESULT " + json.dumps(item, sort_keys=True), flush=True)

    robust_values = [item["robust_accuracy"] for item in results]
    monotonic_ok = all(robust_values[i] >= robust_values[i + 1] - 1e-9 for i in range(len(robust_values) - 1))
    summary = {
        "status": "passed" if results[0]["clean_accuracy"] == results[0]["robust_accuracy"] and monotonic_ok else "partial",
        "model_key": args.model_key,
        "monotonic_ok": monotonic_ok,
        "results": results,
    }
    print("ATTACK_SANITY_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)
    if args.write_json:
        Path(args.write_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.write_markdown:
        Path(args.write_markdown).write_text(markdown_report(args.model_key, cifar_status, results, monotonic_ok), encoding="utf-8")


if __name__ == "__main__":
    main()
