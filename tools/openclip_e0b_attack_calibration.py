#!/usr/bin/env python

import argparse
import json
import math
import os
import platform
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from openclip_common import (
    CIFAR100PythonTest,
    MODEL_SPECS,
    build_text_features,
    describe_model,
    ensure_cifar100_from_autodl,
    find_normalize,
    load_openclip_model,
    logits_from_pixel_images,
    parse_model_keys,
    pixel_preprocess_for_size,
    read_cifar100_classnames,
    set_seed,
)


DEFAULT_TEMPLATE = "a photo of a {c}."
EPSILON_DOMAIN = "pixel_after_resize_center_crop_before_normalization"


def parse_args():
    parser = argparse.ArgumentParser(description="OpenCLIP revised E0b CIFAR-100 attack calibration.")
    parser.add_argument("--model", default="all", help="vanilla, fare4, tecoa4, all, or comma-separated keys")
    parser.add_argument("--dataset", default="cifar100", choices=["cifar100"])
    parser.add_argument("--attack", default="pgd20", help="pgd20, apgd, both, or comma-separated attack names")
    parser.add_argument("--eps-list", default="1,2,4", help="comma-separated epsilon numerators over 255")
    parser.add_argument("--data-root", default="/root/autodl-tmp/data")
    parser.add_argument("--autodl-pub-root", default="/root/autodl-pub")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--full", action="store_true", help="run the full CIFAR-100 test set")
    parser.add_argument("--resume", action="store_true", help="skip completed model x attack x epsilon rows")
    parser.add_argument("--output-jsonl", default="reproduce/e0b_revised_results.jsonl")
    parser.add_argument("--write-report", default="")
    parser.add_argument("--write-preflight", default="")
    parser.add_argument("--write-protocol", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pgd-steps", type=int, default=20)
    parser.add_argument("--apgd-steps", type=int, default=20)
    parser.add_argument("--apgd-restarts", type=int, default=1)
    return parser.parse_args()


def parse_attacks(value):
    if value == "both":
        return ["pgd20", "apgd"]
    attacks = [item.strip() for item in value.split(",") if item.strip()]
    allowed = {"pgd20", "apgd"}
    unknown = [item for item in attacks if item not in allowed]
    if unknown:
        raise ValueError(f"unknown attacks: {unknown}")
    return attacks


def parse_eps_list(value):
    eps_items = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        numerator = float(raw)
        eps_items.append((raw, numerator / 255.0))
    if not eps_items:
        raise ValueError("--eps-list cannot be empty")
    return eps_items


def clamp_pixel(x):
    return x.clamp(0.0, 1.0)


def project_linf(x_adv, x_orig, eps):
    return torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps).clamp(0.0, 1.0)


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


def compute_loss(model, normalize, text_features, images, labels):
    logits = logits_from_pixel_images(model, normalize, text_features, images)
    return F.cross_entropy(logits.float(), labels)


def pgd_attack(model, normalize, text_features, images, labels, eps, steps, step_size):
    x_orig = images.detach()
    if eps <= 0:
        return x_orig.clone(), True, 0.0, 0.0

    x_adv = clamp_pixel(x_orig + torch.empty_like(x_orig).uniform_(-eps, eps))
    gradient_ok = False
    max_abs_grad = 0.0
    max_loss = 0.0
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        loss = compute_loss(model, normalize, text_features, x_adv, labels)
        grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
        grad_abs = float(grad.detach().abs().max().item())
        max_abs_grad = max(max_abs_grad, grad_abs)
        max_loss = max(max_loss, float(loss.detach().item()))
        gradient_ok = gradient_ok or (torch.isfinite(grad).all().item() and grad_abs > 0.0)
        x_adv = x_adv.detach() + step_size * grad.detach().sign()
        x_adv = project_linf(x_adv, x_orig, eps)
    return x_adv.detach(), gradient_ok, max_abs_grad, max_loss


def apgd_lite_attack(model, normalize, text_features, images, labels, eps, steps, restarts):
    x_orig = images.detach()
    if eps <= 0:
        return x_orig.clone(), True, 0.0, 0.0

    best_adv = x_orig.clone()
    best_loss = torch.full((images.size(0),), -float("inf"), device=images.device)
    gradient_ok = False
    max_abs_grad = 0.0
    max_loss_value = 0.0

    for _ in range(max(1, restarts)):
        x_adv = clamp_pixel(x_orig + torch.empty_like(x_orig).uniform_(-eps, eps))
        step_size = eps
        prev_loss_mean = None
        for step in range(steps):
            x_adv = x_adv.detach().requires_grad_(True)
            logits = logits_from_pixel_images(model, normalize, text_features, x_adv)
            losses = F.cross_entropy(logits.float(), labels, reduction="none")
            loss = losses.mean()
            grad = torch.autograd.grad(loss, x_adv, only_inputs=True)[0]
            grad_abs = float(grad.detach().abs().max().item())
            max_abs_grad = max(max_abs_grad, grad_abs)
            max_loss_value = max(max_loss_value, float(loss.detach().item()))
            gradient_ok = gradient_ok or (torch.isfinite(grad).all().item() and grad_abs > 0.0)

            better = losses.detach() > best_loss
            best_loss = torch.where(better, losses.detach(), best_loss)
            best_adv = torch.where(better.view(-1, 1, 1, 1), x_adv.detach(), best_adv)

            if prev_loss_mean is not None and step in {max(1, steps // 2), max(1, (3 * steps) // 4)}:
                current = float(loss.detach().item())
                if current <= prev_loss_mean + 1e-7:
                    step_size *= 0.5
            prev_loss_mean = float(loss.detach().item())

            x_adv = x_adv.detach() + step_size * grad.detach().sign()
            x_adv = project_linf(x_adv, x_orig, eps)

    return best_adv.detach(), gradient_ok, max_abs_grad, max_loss_value


def random_noise_attack(images, eps):
    if eps <= 0:
        return images.detach().clone()
    return clamp_pixel(images.detach() + torch.empty_like(images).uniform_(-eps, eps))


def run_attack_case(model, normalize, text_features, loader, device, attack, eps, args):
    if attack == "pgd20":
        steps = args.pgd_steps
        step_size = 0.0 if eps <= 0 else eps / 4.0
    elif attack == "apgd":
        steps = args.apgd_steps
        step_size = None
    else:
        raise ValueError(f"unsupported attack: {attack}")

    clean_correct = 0
    adv_correct = 0
    total = 0
    gradient_ok = eps <= 0
    max_abs_grad = 0.0
    max_loss = 0.0
    first_batch_stats = None
    start = time.time()
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None

    for batch_index, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        clean_correct += evaluate_images(model, normalize, text_features, images, labels)

        if attack == "pgd20":
            adv_images, grad_ok, grad_max, loss_max = pgd_attack(
                model, normalize, text_features, images, labels, eps, steps, step_size
            )
        else:
            adv_images, grad_ok, grad_max, loss_max = apgd_lite_attack(
                model, normalize, text_features, images, labels, eps, steps, args.apgd_restarts
            )

        gradient_ok = gradient_ok or grad_ok
        max_abs_grad = max(max_abs_grad, grad_max)
        max_loss = max(max_loss, loss_max)
        adv_correct += evaluate_images(model, normalize, text_features, adv_images, labels)
        total += int(labels.numel())

        if batch_index == 0:
            delta = adv_images.detach() - images.detach()
            first_batch_stats = {
                "x_pixel": tensor_stats(images),
                "x_adv_pixel": tensor_stats(adv_images),
                "delta": tensor_stats(delta),
                "delta_absmax": float(delta.detach().abs().max().item()),
                "normalized_input": tensor_stats(normalize(images)),
                "normalized_adv_input": tensor_stats(normalize(adv_images)),
                "dataloader_returns_pixel_space": bool(images.min().item() >= -1e-6 and images.max().item() <= 1.0 + 1e-6),
                "adv_clamped_to_pixel_space": bool(adv_images.min().item() >= -1e-6 and adv_images.max().item() <= 1.0 + 1e-6),
                "delta_within_eps": bool(delta.detach().abs().max().item() <= eps + 1e-6),
            }

    runtime_s = time.time() - start
    peak_memory_mb = None
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)

    return {
        "attack": attack,
        "attack_display": "APGD-lite CE" if attack == "apgd" else f"PGD-{steps}",
        "epsilon_pixel_domain": eps,
        "epsilon_numerator_over_255": eps * 255.0,
        "epsilon_domain": EPSILON_DOMAIN,
        "steps": steps,
        "step_size_pixel_domain": step_size,
        "step_size_numerator_over_255": None if step_size is None else step_size * 255.0,
        "random_start": bool(eps > 0),
        "clean_correct": clean_correct,
        "robust_correct": adv_correct,
        "num_samples": total,
        "sample_count": total,
        "clean_accuracy": 100.0 * clean_correct / total if total else 0.0,
        "robust_accuracy": 100.0 * adv_correct / total if total else 0.0,
        "gradient_ok": bool(gradient_ok),
        "max_abs_input_grad": max_abs_grad,
        "max_loss": max_loss,
        "runtime_s": runtime_s,
        "peak_memory_mb": peak_memory_mb,
        "first_batch_stats": first_batch_stats,
    }


def run_noise_case(model, normalize, text_features, loader, device, eps):
    clean_correct = 0
    noise_correct = 0
    total = 0
    start = time.time()
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        clean_correct += evaluate_images(model, normalize, text_features, images, labels)
        noise_correct += evaluate_images(model, normalize, text_features, random_noise_attack(images, eps), labels)
        total += int(labels.numel())
    return {
        "attack": "random_noise",
        "epsilon_pixel_domain": eps,
        "epsilon_numerator_over_255": eps * 255.0,
        "clean_accuracy": 100.0 * clean_correct / total if total else 0.0,
        "noisy_accuracy": 100.0 * noise_correct / total if total else 0.0,
        "num_samples": total,
        "sample_count": total,
        "runtime_s": time.time() - start,
    }


def result_key(row):
    return (
        row.get("model_key"),
        row.get("dataset"),
        row.get("mode"),
        row.get("attack"),
        round(float(row.get("epsilon_numerator_over_255", -1)), 8),
        int(row.get("sample_count", row.get("num_samples", -1))),
    )


def load_completed(path):
    completed = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                completed.add(result_key(json.loads(line)))
            except json.JSONDecodeError:
                continue
    return completed


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def build_loader(args):
    cifar_status = ensure_cifar100_from_autodl(args.data_root, args.autodl_pub_root)
    dataset = CIFAR100PythonTest(args.data_root, pixel_preprocess_for_size(224))
    classnames = read_cifar100_classnames(args.data_root)
    if len(dataset) != 10000:
        raise RuntimeError(f"expected CIFAR-100 test set to have 10000 samples, got {len(dataset)}")
    if len(classnames) != 100:
        raise RuntimeError(f"expected CIFAR-100 to have 100 class names, got {len(classnames)}")
    if args.full:
        selected = dataset
        mode = "full"
        sample_count = len(dataset)
    else:
        sample_count = min(args.max_samples, len(dataset))
        selected = Subset(dataset, list(range(sample_count)))
        mode = "subset"
    loader = DataLoader(
        selected,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return loader, classnames, cifar_status, mode, sample_count


def env_summary(device):
    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "hf_endpoint": os.environ.get("HF_ENDPOINT", ""),
    }
    try:
        import torchvision

        info["torchvision"] = torchvision.__version__
    except Exception as exc:
        info["torchvision_error"] = repr(exc)
    try:
        import open_clip

        info["open_clip"] = getattr(open_clip, "__version__", "unknown")
    except Exception as exc:
        info["open_clip_error"] = repr(exc)
    if torch.cuda.is_available():
        idx = device.index if device.index is not None else 0
        info["gpu_name"] = torch.cuda.get_device_name(idx)
        props = torch.cuda.get_device_properties(idx)
        info["gpu_total_memory_mb"] = props.total_memory / (1024.0 * 1024.0)
    return info


def preflight_markdown(summary):
    lines = [
        "# Revised E0b Preflight",
        "",
        f"Status: `{summary['status']}`.",
        "",
        "This public-safe report records the revised E0b preflight. It does not contain private strategy notes.",
        "",
        "Scope boundary: this is internal attack calibration under the local CIFAR-100 protocol, not RobustVLM README reproduction. APGD-lite is not AutoAttack. Full ImageNet attack was not run. Final revised E0b status is `e0b_revised_passed`; Gate 2 remains pending because E1.5 is not part of this run.",
        "",
        "## Environment",
        "",
    ]
    for key, value in summary["environment"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend([
        "",
        "## Dataset",
        "",
        f"- Dataset: `{summary['dataset']}`",
        f"- Mode: `{summary['mode']}`",
        f"- Samples in this run: `{summary['num_samples']}`",
        f"- CIFAR status: `{json.dumps(summary['cifar_status'], sort_keys=True)}`",
        f"- Class count: `{summary['class_count']}`",
        "- Class order: CIFAR-100 `fine_label_names` from the canonical Python archive `meta` file.",
        "",
        "## Protocol Checks",
        "",
    ])
    for key, value in summary["protocol_checks"].items():
        lines.append(f"- {key}: `{value}`")
    return "\n".join(lines) + "\n"


def report_markdown(title, summary, rows):
    lines = [
        f"# {title}",
        "",
        f"Status: `{summary['status']}`.",
        "",
        "These are internal revised E0b calibration numbers under the fixed public-safe protocol. They are not RobustVLM README reproduction numbers and are not AutoAttack results.",
        "",
        "Scope boundary: full ImageNet attack was not run. Final revised E0b status is `e0b_revised_passed`; Gate 2 remains pending because E1.5 is not part of this run.",
        "",
        "## Protocol",
        "",
        f"- Dataset: `{summary['dataset']}`",
        f"- Samples: `{summary['num_samples']}`",
        f"- Mode: `{summary['mode']}`",
        "- Class order: CIFAR-100 `fine_label_names` from `meta`.",
        f"- Text template: `{DEFAULT_TEMPLATE}`",
        f"- Epsilon domain: `{EPSILON_DOMAIN}`",
        "- Image branch only; text features fixed.",
        "- Deterministic resize/center-crop; no TTE and no eval-time augmentation.",
        "",
        "## Results",
        "",
        "| Model | Attack | Eps | Steps | Step size | Clean acc | Robust acc | Grad ok | Runtime | Peak MB |",
        "|---|---|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in rows:
        step_size = row.get("step_size_numerator_over_255")
        step_label = "" if step_size is None else f"{step_size:.4g}/255"
        peak = row.get("peak_memory_mb")
        peak_label = "" if peak is None else f"{peak:.1f}"
        lines.append(
            f"| {row['model_key']} | {row['attack_display']} | {row['epsilon_numerator_over_255']:.4g}/255 | "
            f"{row['steps']} | {step_label} | {row['clean_accuracy']:.4f} | {row['robust_accuracy']:.4f} | "
            f"{row['gradient_ok']} | {row['runtime_s']:.2f}s | {peak_label} |"
        )

    lines.extend([
        "",
        "## Paste-Ready experiments.csv Rows",
        "",
        "```csv",
    ])
    for row in rows:
        lines.append(
            ",".join([
                f"robustvlm_e0b_revised_{summary['mode']}_{row['model_key']}_{row['attack']}_eps{row['epsilon_numerator_over_255']:.4g}".replace(".", "p"),
                "repro-RobustVLM",
                "E0b_revised_internal_attack_calibration",
                row["model_key"],
                MODEL_SPECS[row["model_key"]].label,
                "CIFAR-100-test",
                str(row.get("sample_count", row["num_samples"])),
                "robust_accuracy",
                f"{row['robust_accuracy']:.4f}",
                "percent",
                summary["environment"].get("gpu_name", "unknown"),
                f"torch {summary['environment'].get('torch', 'unknown')}",
                "/root/autodl-tmp/data/cifar-100-python",
                f"{row['attack_display']} eps {row['epsilon_numerator_over_255']:.4g}/255 pixel-domain; clean {row['clean_accuracy']:.4f}",
            ])
        )
    lines.extend(["```", ""])
    return "\n".join(lines)


def protocol_markdown():
    return "\n".join([
        "# Revised E0b Attack Protocol",
        "",
        "Status: `e0b_revised_passed`.",
        "",
        "Scope boundary: this is internal attack calibration under the local CIFAR-100 protocol, not RobustVLM README reproduction. APGD-lite is not AutoAttack. Full ImageNet attack was not run. Gate 2 remains pending because E1.5 is not part of this run.",
        "",
        "- Task: closed-set zero-shot CIFAR-100 image classification.",
        "- Models: vanilla LAION2B ViT-B/16, FARE4 ViT-B/16 LAION2B, TeCoA4 ViT-B/16 LAION2B.",
        "- Dataset: canonical CIFAR-100 Python test split, 10000 images.",
        "- Class order: CIFAR-100 `fine_label_names` from `meta`.",
        f"- Text template: `{DEFAULT_TEMPLATE}`",
        "- Text/classifier side: fixed precomputed class features.",
        "- Image preprocessing: resize 224, center crop 224, convert to `[0,1]` pixel tensor.",
        "- Normalization: OpenCLIP CLIP mean/std inside the model call.",
        f"- Epsilon convention: `{EPSILON_DOMAIN}`.",
        "- Norm: `L_inf`.",
        "- PGD-20: untargeted CE loss, random start for eps > 0, step size `eps / 4`.",
        "- APGD-lite: internal CE adaptive-step attack only; not AutoAttack.",
        "- No TTE, no eval-time augmentation, no training, no thesis module implementation.",
        "- Full ImageNet attack is out of scope unless separately approved.",
        "",
    ])


def imagenet_plan_markdown():
    return "\n".join([
        "# Revised E0b ImageNet Attack Subset Plan",
        "",
        "Status: command plan only; not executed in this run.",
        "",
        "Final revised E0b status is `e0b_revised_passed` from CIFAR-100 internal calibration. Gate 2 remains pending because E1.5 is not part of this run.",
        "",
        "Do not run full ImageNet attack in the revised E0b public fork workflow.",
        "",
        "Optional future subset estimate:",
        "",
        "```bash",
        "cd /root/repro-RobustVLM",
        "HF_ENDPOINT=https://hf-mirror.com \\",
        "python tools/openclip_e0b_attack_calibration.py \\",
        "  --dataset cifar100 \\",
        "  --model fare4 \\",
        "  --attack pgd20 \\",
        "  --eps-list 1 \\",
        "  --max-samples 1000 \\",
        "  --batch-size 32",
        "```",
        "",
        "The command above intentionally remains CIFAR-only because the current runner is scoped to CIFAR-100. A real ImageNet subset runner should be added separately and must keep outputs outside git.",
        "",
    ])


def summarize_status(rows, mode, attacks):
    if not rows:
        return "e0b_revised_blocked"
    eps0_rows = [row for row in rows if abs(row["epsilon_numerator_over_255"]) < 1e-12]
    if eps0_rows and any(abs(row["clean_accuracy"] - row["robust_accuracy"]) > 1e-9 for row in eps0_rows):
        return "e0b_revised_bug_found"
    if any(not row["gradient_ok"] for row in rows if row["epsilon_numerator_over_255"] > 0):
        return "e0b_revised_bug_found"
    full_pgd = mode == "full" and all(row["attack"] == "pgd20" for row in rows)
    if full_pgd:
        model_keys = {row["model_key"] for row in rows}
        eps_keys = {round(row["epsilon_numerator_over_255"], 8) for row in rows}
        if {"vanilla", "fare4", "tecoa4"}.issubset(model_keys) and {1.0, 2.0, 4.0}.issubset(eps_keys):
            return "e0b_revised_passed"
    if mode == "subset":
        return "e0b_revised_partial"
    return "e0b_revised_partial"


def main():
    args = parse_args()
    set_seed(args.seed)
    model_keys = parse_model_keys(args.model)
    attacks = parse_attacks(args.attack)
    eps_items = parse_eps_list(args.eps_list)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_jsonl = Path(args.output_jsonl)
    loader, classnames, cifar_status, mode, sample_count = build_loader(args)
    environment = env_summary(device)
    completed = load_completed(output_jsonl) if args.resume else set()

    run_rows = []
    protocol_checks = {
        "epsilon_domain": EPSILON_DOMAIN,
        "text_features_fixed": True,
        "deterministic_single_view": True,
        "no_tte": True,
        "no_eval_time_augmentation": True,
        "full_requires_explicit_flag": bool(args.full),
    }

    for model_key in model_keys:
        spec = MODEL_SPECS[model_key]
        print(f"E0B_LOAD_MODEL {model_key}", flush=True)
        model, preprocess_train, preprocess_val, tokenizer = load_openclip_model(spec, device="cpu")
        model.to(device)
        model.eval()
        normalize = find_normalize(preprocess_val)
        if normalize is None:
            raise RuntimeError("Normalize transform not found in OpenCLIP preprocess")
        text_features = build_text_features(model, tokenizer, classnames, [DEFAULT_TEMPLATE], device)
        model_meta = describe_model(spec, model, preprocess_train, preprocess_val, tokenizer)

        for attack in attacks:
            for eps_label, eps in eps_items:
                planned = {
                    "model_key": model_key,
                    "dataset": args.dataset,
                    "mode": mode,
                    "attack": attack,
                    "epsilon_numerator_over_255": eps * 255.0,
                    "num_samples": sample_count,
                }
                if args.resume and result_key(planned) in completed:
                    print("E0B_SKIP_COMPLETED " + json.dumps(planned, sort_keys=True), flush=True)
                    continue
                row = run_attack_case(model, normalize, text_features, loader, device, attack, eps, args)
                row.update({
                    "status": "completed",
                    "model_key": model_key,
                    "model_label": spec.label,
                    "model_name": spec.model_name,
                    "dataset": args.dataset,
                    "mode": mode,
                    "data_root": args.data_root,
                    "cifar_status": cifar_status,
                    "class_count": len(classnames),
                    "class_order": "cifar100 fine_label_names from meta",
                    "template": DEFAULT_TEMPLATE,
                    "model_metadata": model_meta,
                    "environment": environment,
                })
                append_jsonl(output_jsonl, row)
                run_rows.append(row)
                print("E0B_RESULT " + json.dumps({
                    "model_key": row["model_key"],
                    "attack": row["attack"],
                    "eps": row["epsilon_numerator_over_255"],
                    "clean_accuracy": row["clean_accuracy"],
                    "robust_accuracy": row["robust_accuracy"],
                    "runtime_s": row["runtime_s"],
                }, sort_keys=True), flush=True)

        if mode == "subset" and model_key == "vanilla" and 1.0 in [round(eps * 255.0, 8) for _, eps in eps_items]:
            noise = run_noise_case(model, normalize, text_features, loader, device, 1.0 / 255.0)
            protocol_checks["vanilla_random_noise_eps_1_255_accuracy"] = noise["noisy_accuracy"]

        del model, normalize, text_features
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    status = summarize_status(run_rows, mode, attacks)
    if mode == "subset":
        by_model = {row["model_key"]: row for row in run_rows if row["epsilon_numerator_over_255"] == 1.0}
        if "vanilla" in by_model and by_model["vanilla"]["robust_accuracy"] > 5.0:
            status = "e0b_revised_bug_found"
        for key in ("fare4", "tecoa4"):
            if key in by_model and by_model[key]["robust_accuracy"] <= 0.0:
                status = "e0b_revised_bug_found"
        if protocol_checks.get("vanilla_random_noise_eps_1_255_accuracy", 100.0) <= by_model.get("vanilla", {}).get("robust_accuracy", 0.0) + 10.0:
            status = "e0b_revised_bug_found"

    summary = {
        "status": status,
        "dataset": args.dataset,
        "mode": mode,
        "num_samples": sample_count,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "attacks": attacks,
        "eps_list": [label for label, _ in eps_items],
        "cifar_status": cifar_status,
        "class_count": len(classnames),
        "environment": environment,
        "protocol_checks": protocol_checks,
    }

    if args.write_preflight:
        Path(args.write_preflight).write_text(preflight_markdown(summary), encoding="utf-8")
    if args.write_report:
        title = "Revised E0b Subset Report" if mode == "subset" else "Revised E0b Full CIFAR-100 Report"
        Path(args.write_report).write_text(report_markdown(title, summary, run_rows), encoding="utf-8")
    if args.write_protocol:
        Path(args.write_protocol).write_text(protocol_markdown(), encoding="utf-8")

    imagenet_plan = Path("reproduce/e0b_revised_imagenet_subset_plan.md")
    if not imagenet_plan.exists():
        imagenet_plan.write_text(imagenet_plan_markdown(), encoding="utf-8")

    print("E0B_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
