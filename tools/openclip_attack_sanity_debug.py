#!/usr/bin/env python

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from openclip_common import (
    CIFAR100PythonTest,
    MODEL_SPECS,
    build_text_features,
    ensure_cifar100_from_autodl,
    find_normalize,
    load_openclip_model,
    logits_from_pixel_images,
    pixel_preprocess_for_size,
    read_cifar100_classnames,
    set_seed,
)


EPS_SWEEP = [
    0.0,
    0.125 / 255.0,
    0.25 / 255.0,
    0.5 / 255.0,
    1.0 / 255.0,
    2.0 / 255.0,
    4.0 / 255.0,
]
STEP_SWEEP = [1, 2, 5, 10]
NOISE_EPS = [1.0 / 255.0, 2.0 / 255.0, 4.0 / 255.0]
FARE4_EPS = [1.0 / 255.0, 2.0 / 255.0]


def parse_args():
    parser = argparse.ArgumentParser(description="OpenCLIP E0b-lite attack protocol debug.")
    parser.add_argument("--data-root", default="/root/autodl-tmp/data")
    parser.add_argument("--autodl-pub-root", default="/root/autodl-pub")
    parser.add_argument("--subset-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--write-markdown", default="")
    parser.add_argument("--write-json", default="")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def clamp_pixel(x):
    return x.clamp(0.0, 1.0)


def project_linf(x_adv, x_orig, eps):
    return torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps).clamp(0.0, 1.0)


def step_size_for(eps, steps):
    if eps == 0:
        return 0.0
    if steps == 1:
        return eps
    return eps / 2.0


def tensor_stats(tensor):
    data = tensor.detach().float().cpu()
    return {
        "shape": list(data.shape),
        "min": float(data.min().item()),
        "max": float(data.max().item()),
        "mean": float(data.mean().item()),
        "std": float(data.std(unbiased=False).item()),
    }


def evaluate_images(model, normalize, text_features, images, labels):
    with torch.no_grad():
        logits = logits_from_pixel_images(model, normalize, text_features, images)
        pred = logits.argmax(dim=1)
        return int(pred.eq(labels).sum().item())


def pgd_attack_primary(model, normalize, text_features, images, labels, eps, steps, step_size, random_start):
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
        grad_abs = float(grad.detach().abs().max().item())
        max_abs_grad = max(max_abs_grad, grad_abs)
        gradient_ok = gradient_ok or (torch.isfinite(grad).all().item() and grad_abs > 0.0)
        if eps > 0:
            x_adv = x_adv.detach() + step_size * grad.detach().sign()
            x_adv = project_linf(x_adv, x_orig, eps)
        else:
            x_adv = x_orig.clone()
    return x_adv.detach(), gradient_ok, max_abs_grad


def pgd_attack_minimal(model, normalize, text_features, images, labels, eps, steps, step_size, random_start):
    x_orig = images.detach()
    if eps == 0:
        x_adv = x_orig.clone()
    elif random_start:
        x_adv = clamp_pixel(x_orig + torch.empty_like(x_orig).uniform_(-eps, eps))
    else:
        x_adv = x_orig.clone()

    gradient_ok = False
    max_abs_grad = 0.0
    scale = model.logit_scale.detach().exp() if hasattr(model, "logit_scale") else 1.0
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        image_features = model.encode_image(normalize(x_adv), normalize=True)
        logits = (image_features @ text_features) * scale
        loss = F.cross_entropy(logits.float(), labels)
        loss.backward()
        grad = x_adv.grad.detach()
        grad_abs = float(grad.abs().max().item())
        max_abs_grad = max(max_abs_grad, grad_abs)
        gradient_ok = gradient_ok or (torch.isfinite(grad).all().item() and grad_abs > 0.0)
        if eps > 0:
            x_adv = x_adv.detach() + step_size * grad.sign()
            x_adv = project_linf(x_adv, x_orig, eps)
        else:
            x_adv = x_orig.clone()
    return x_adv.detach(), gradient_ok, max_abs_grad


def random_noise_attack(images, eps):
    if eps == 0:
        return images.detach().clone()
    return clamp_pixel(images.detach() + torch.empty_like(images).uniform_(-eps, eps))


def run_case(model, normalize, text_features, loader, device, attack_kind, eps, steps):
    step_size = step_size_for(eps, steps)
    random_start = bool(eps > 0 and attack_kind == "pgd_primary" and steps > 1)
    clean_correct = 0
    adv_correct = 0
    total = 0
    gradient_ok = eps == 0 or attack_kind == "random_noise"
    max_abs_grad = 0.0
    start = time.time()

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        clean_correct += evaluate_images(model, normalize, text_features, images, labels)
        if attack_kind == "pgd_primary":
            adv_images, grad_ok, grad_max = pgd_attack_primary(
                model, normalize, text_features, images, labels, eps, steps, step_size, random_start
            )
            gradient_ok = gradient_ok or grad_ok
            max_abs_grad = max(max_abs_grad, grad_max)
        elif attack_kind == "pgd_minimal":
            adv_images, grad_ok, grad_max = pgd_attack_minimal(
                model, normalize, text_features, images, labels, eps, steps, step_size, random_start
            )
            gradient_ok = gradient_ok or grad_ok
            max_abs_grad = max(max_abs_grad, grad_max)
        elif attack_kind == "random_noise":
            adv_images = random_noise_attack(images, eps)
        else:
            raise ValueError(f"unknown attack kind: {attack_kind}")
        adv_correct += evaluate_images(model, normalize, text_features, adv_images, labels)
        total += int(labels.numel())

    return {
        "attack_kind": attack_kind,
        "epsilon_pixel_domain": eps,
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
        "runtime_s": time.time() - start,
    }


def first_batch_debug(model, normalize, text_features, loader, device):
    images, labels = next(iter(loader))
    images = images.to(device)
    labels = labels.to(device)
    eps = 1.0 / 255.0
    steps = 5
    step_size = step_size_for(eps, steps)
    adv_images, grad_ok, grad_max = pgd_attack_primary(
        model, normalize, text_features, images, labels, eps, steps, step_size, random_start=True
    )
    delta = adv_images - images
    normalized = normalize(images)
    normalized_adv = normalize(adv_images)
    delta_absmax = float(delta.detach().abs().max().item())
    return {
        "setting": "vanilla PGD-5 eps=1/255 first batch",
        "x_pixel": tensor_stats(images),
        "x_adv_pixel": tensor_stats(adv_images),
        "delta": tensor_stats(delta),
        "delta_absmax": delta_absmax,
        "normalized_input": tensor_stats(normalized),
        "normalized_adv_input": tensor_stats(normalized_adv),
        "gradient_ok": bool(grad_ok),
        "max_abs_input_grad": grad_max,
        "dataloader_returns_pixel_space": bool(images.min().item() >= -1e-6 and images.max().item() <= 1.0 + 1e-6),
        "adv_clamped_to_pixel_space": bool(adv_images.min().item() >= -1e-6 and adv_images.max().item() <= 1.0 + 1e-6),
        "delta_within_eps": bool(delta_absmax <= eps + 1e-6),
        "epsilon_domain": "pixel_after_resize_before_normalization",
    }


def prepare_model(model_key, device, data_root):
    spec = MODEL_SPECS[model_key]
    model, _, preprocess_val, tokenizer = load_openclip_model(spec, device="cpu")
    model.to(device)
    model.eval()
    normalize = find_normalize(preprocess_val)
    if normalize is None:
        raise RuntimeError("Normalize transform not found in OpenCLIP preprocess")
    classnames = read_cifar100_classnames(data_root)
    templates = ["a photo of a {c}."]
    text_features = build_text_features(model, tokenizer, classnames, templates, device)
    return model, normalize, text_features


def run_vanilla_debug(loader, device, data_root):
    model, normalize, text_features = prepare_model("vanilla", device, data_root)
    first_stats = first_batch_debug(model, normalize, text_features, loader, device)

    eps_sweep = []
    for eps in EPS_SWEEP:
        item = run_case(model, normalize, text_features, loader, device, "pgd_primary", eps, 5)
        item["model_key"] = "vanilla"
        item["sweep"] = "epsilon"
        eps_sweep.append(item)
        print("E0B_LITE_DEBUG_RESULT " + json.dumps(item, sort_keys=True), flush=True)

    steps_sweep = []
    for steps in STEP_SWEEP:
        item = run_case(model, normalize, text_features, loader, device, "pgd_primary", 1.0 / 255.0, steps)
        item["model_key"] = "vanilla"
        item["sweep"] = "steps"
        item["attack_label"] = "fgsm" if steps == 1 else f"pgd_{steps}"
        steps_sweep.append(item)
        print("E0B_LITE_DEBUG_RESULT " + json.dumps(item, sort_keys=True), flush=True)

    noise = []
    for eps in NOISE_EPS:
        item = run_case(model, normalize, text_features, loader, device, "random_noise", eps, 0)
        item["model_key"] = "vanilla"
        item["sweep"] = "random_noise"
        noise.append(item)
        print("E0B_LITE_DEBUG_RESULT " + json.dumps(item, sort_keys=True), flush=True)

    cross_check = run_case(model, normalize, text_features, loader, device, "pgd_minimal", 1.0 / 255.0, 5)
    cross_check["model_key"] = "vanilla"
    cross_check["sweep"] = "cross_check"
    print("E0B_LITE_DEBUG_RESULT " + json.dumps(cross_check, sort_keys=True), flush=True)

    del model, normalize, text_features
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "first_batch_stats": first_stats,
        "epsilon_sweep": eps_sweep,
        "steps_sweep": steps_sweep,
        "random_noise": noise,
        "minimal_pgd_cross_check": cross_check,
    }


def run_fare4_debug(loader, device, data_root):
    model, normalize, text_features = prepare_model("fare4", device, data_root)
    results = []
    for eps in FARE4_EPS:
        item = run_case(model, normalize, text_features, loader, device, "pgd_primary", eps, 5)
        item["model_key"] = "fare4"
        item["sweep"] = "fare4_small"
        results.append(item)
        print("E0B_LITE_DEBUG_RESULT " + json.dumps(item, sort_keys=True), flush=True)
    del model, normalize, text_features
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def fmt_pct(value):
    return f"{value:.4f}"


def eps_label(eps):
    if eps == 0:
        return "0"
    return f"{eps * 255.0:.3g}/255"


def table_for(items, include_attack=False):
    if include_attack:
        lines = [
            "| Model | Setting | Epsilon | Steps | Step size | Clean acc | Attacked/noisy acc | Grad ok | Runtime |",
            "|---|---|---:|---:|---:|---:|---:|---|---:|",
        ]
    else:
        lines = [
            "| Epsilon | Steps | Step size | Clean acc | Attacked/noisy acc | Grad ok | Runtime |",
            "|---:|---:|---:|---:|---:|---|---:|",
        ]
    for item in items:
        if include_attack:
            setting = item.get("attack_label") or item.get("attack_kind") or item.get("sweep")
            lines.append(
                f"| {item.get('model_key', '')} | {setting} | {eps_label(item['epsilon_pixel_domain'])} | "
                f"{item['steps']} | {eps_label(item['step_size_pixel_domain'])} | "
                f"{fmt_pct(item['clean_accuracy'])} | {fmt_pct(item['robust_accuracy'])} | "
                f"{item['gradient_ok']} | {item['runtime_s']:.2f}s |"
            )
        else:
            lines.append(
                f"| {eps_label(item['epsilon_pixel_domain'])} | {item['steps']} | "
                f"{eps_label(item['step_size_pixel_domain'])} | {fmt_pct(item['clean_accuracy'])} | "
                f"{fmt_pct(item['robust_accuracy'])} | {item['gradient_ok']} | {item['runtime_s']:.2f}s |"
            )
    return lines


def stats_table(stats):
    rows = [
        ("x_pixel", stats["x_pixel"]),
        ("x_adv_pixel", stats["x_adv_pixel"]),
        ("delta", stats["delta"]),
        ("normalized_input", stats["normalized_input"]),
        ("normalized_adv_input", stats["normalized_adv_input"]),
    ]
    lines = [
        "| Tensor | Min | Max | Mean | Std | Shape |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, item in rows:
        lines.append(
            f"| {name} | {item['min']:.8f} | {item['max']:.8f} | {item['mean']:.8f} | "
            f"{item['std']:.8f} | `{item['shape']}` |"
        )
    return lines


def answer_questions(summary):
    eps_sweep = summary["vanilla"]["epsilon_sweep"]
    steps_sweep = summary["vanilla"]["steps_sweep"]
    noise = summary["vanilla"]["random_noise"]
    fare4 = summary["fare4"]
    first = summary["vanilla"]["first_batch_stats"]
    cross = summary["vanilla"]["minimal_pgd_cross_check"]

    clean = eps_sweep[0]["clean_accuracy"]
    eps0_exact = eps_sweep[0]["clean_accuracy"] == eps_sweep[0]["robust_accuracy"]
    eps_one = next(item for item in eps_sweep if abs(item["epsilon_pixel_domain"] - 1.0 / 255.0) < 1e-12)
    first_nonzero = next(item for item in eps_sweep if item["epsilon_pixel_domain"] > 0)
    fgsm = next(item for item in steps_sweep if item["steps"] == 1)
    noise_one = next(item for item in noise if abs(item["epsilon_pixel_domain"] - 1.0 / 255.0) < 1e-12)
    fare4_one = next(item for item in fare4 if abs(item["epsilon_pixel_domain"] - 1.0 / 255.0) < 1e-12)

    noise_weaker = noise_one["robust_accuracy"] > eps_one["robust_accuracy"] + 10.0
    immediate_collapse = first_nonzero["robust_accuracy"] <= max(5.0, clean * 0.1)
    fgsm_collapse = fgsm["robust_accuracy"] <= max(5.0, clean * 0.1)
    fare4_more_robust = fare4_one["robust_accuracy"] > eps_one["robust_accuracy"] + 5.0
    independent_delta_pp = abs(cross["robust_accuracy"] - eps_one["robust_accuracy"])
    independent_matches = independent_delta_pp <= 1.0
    inputs_pixel = first["dataloader_returns_pixel_space"]
    clamp_ok = first["adv_clamped_to_pixel_space"] and first["delta_within_eps"]

    if not inputs_pixel or not clamp_ok or not eps0_exact or not independent_matches:
        verdict = "bug_found_or_protocol_invalid"
    elif noise_weaker and fare4_more_robust:
        verdict = "suspicious_but_reproducible_no_local_bug_found"
    else:
        verdict = "plausible_but_requires_official_imagenet_check"

    return {
        "eps0_exact": eps0_exact,
        "random_noise_much_weaker_than_pgd": noise_weaker,
        "collapse_pattern": "immediate" if immediate_collapse else "gradual",
        "fgsm_already_collapses": fgsm_collapse,
        "fare4_more_robust_than_vanilla": fare4_more_robust,
        "inputs_definitely_pixel_space": inputs_pixel,
        "epsilon_after_resize_before_normalization": clamp_ok,
        "previous_eps_1_255_zero_likely": verdict,
        "independent_pgd_matches": independent_matches,
        "independent_pgd_delta_pp": independent_delta_pp,
        "remaining_issue": "Official ImageNet E0a clean calibration and RobustVLM CLIP_eval/AutoAttack path remain unverified because extracted ImageNet val is missing.",
        "verdict": verdict,
    }


def markdown_report(summary):
    answers = summary["answers"]
    lines = [
        "# OpenCLIP E0b-Lite Attack Debug",
        "",
        "This public-safe report debugs the CIFAR-100 attack sanity pipeline. It is not a formal RobustVLM robustness result.",
        "",
        "## Run Context",
        "",
        f"- Dataset status: `{json.dumps(summary['cifar_status'], sort_keys=True)}`",
        f"- Dataset: CIFAR-100 test subset, `{summary['subset_size']}` samples",
        "- Models: vanilla LAION2B ViT-B/16 and FARE4 ViT-B/16 LAION2B",
        "- Text side: fixed zero-shot class features with template `a photo of a {c}.`",
        "- Epsilon domain: pixel space after resize/center-crop and before CLIP normalization",
        "- Attack: untargeted image-only L_inf PGD; no AutoAttack and no full E0b",
        "",
        "## First Batch Pixel / Normalization Checks",
        "",
        *stats_table(summary["vanilla"]["first_batch_stats"]),
        "",
        f"- Dataloader returns [0,1] pixel tensors: `{answers['inputs_definitely_pixel_space']}`",
        f"- PGD adversarial pixels are clamped to [0,1]: `{summary['vanilla']['first_batch_stats']['adv_clamped_to_pixel_space']}`",
        f"- First-batch delta absmax: `{summary['vanilla']['first_batch_stats']['delta_absmax']:.8f}`",
        f"- First-batch delta within eps=1/255: `{summary['vanilla']['first_batch_stats']['delta_within_eps']}`",
        "",
        "## Vanilla Epsilon Sweep",
        "",
        *table_for(summary["vanilla"]["epsilon_sweep"]),
        "",
        "## Vanilla Steps Sweep at eps=1/255",
        "",
        *table_for(summary["vanilla"]["steps_sweep"], include_attack=True),
        "",
        "## Vanilla Random Noise Baselines",
        "",
        *table_for(summary["vanilla"]["random_noise"], include_attack=True),
        "",
        "## Independent PGD Cross-Check",
        "",
        *table_for([summary["vanilla"]["minimal_pgd_cross_check"]], include_attack=True),
        "",
        "## FARE4 Small Sanity",
        "",
        *table_for(summary["fare4"], include_attack=True),
        "",
        "## Debug Answers",
        "",
        f"1. eps=0 exactly matches clean: `{answers['eps0_exact']}`.",
        f"2. Random noise is much weaker than PGD: `{answers['random_noise_much_weaker_than_pgd']}`.",
        f"3. Collapse pattern as eps increases: `{answers['collapse_pattern']}`.",
        f"4. FGSM/PGD-1 already collapses to near 0: `{answers['fgsm_already_collapses']}`.",
        f"5. FARE4 behaves more robustly than vanilla on this subset: `{answers['fare4_more_robust_than_vanilla']}`.",
        f"6. Inputs are pixel-space tensors before normalization: `{answers['inputs_definitely_pixel_space']}`.",
        f"7. Epsilon is applied after resize/crop and before normalization: `{answers['epsilon_after_resize_before_normalization']}`.",
        f"8. Previous eps=1/255 -> 0% status: `{answers['previous_eps_1_255_zero_likely']}`. Independent PGD delta: `{answers['independent_pgd_delta_pp']:.4f}` pp.",
        f"9. Remaining official E0b issue: {answers['remaining_issue']}",
        "",
        "## Verdict",
        "",
        f"`{answers['verdict']}`",
        "",
        "The local attack path is internally consistent if eps=0 matches clean, random noise is weaker than PGD, the independent PGD cross-check matches, and first-batch clamp/statistics are valid. "
        "The very fast collapse remains suspicious for a formal robustness claim and must be rechecked through the official ImageNet/RobustVLM evaluation path once extracted ImageNet val is available.",
        "",
    ]
    return "\n".join(lines)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cifar_status = ensure_cifar100_from_autodl(args.data_root, args.autodl_pub_root)
    transform = pixel_preprocess_for_size(224)
    dataset = CIFAR100PythonTest(args.data_root, transform)
    subset_size = min(args.subset_size, len(dataset))
    subset = Subset(dataset, list(range(subset_size)))
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    summary = {
        "status": "completed",
        "subset_size": subset_size,
        "batch_size": args.batch_size,
        "device": str(device),
        "cifar_status": cifar_status,
        "epsilon_domain": "pixel_after_resize_before_normalization",
    }
    summary["vanilla"] = run_vanilla_debug(loader, device, args.data_root)
    summary["fare4"] = run_fare4_debug(loader, device, args.data_root)
    summary["answers"] = answer_questions(summary)
    print("E0B_LITE_DEBUG_SUMMARY " + json.dumps(summary["answers"], sort_keys=True), flush=True)

    if args.write_json:
        Path(args.write_json).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.write_markdown:
        Path(args.write_markdown).write_text(markdown_report(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
