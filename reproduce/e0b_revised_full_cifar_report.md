# Revised E0b Full CIFAR-100 Report

Status: `e0b_revised_passed`.

These are internal revised E0b calibration numbers under the fixed public-safe protocol. They are not RobustVLM README reproduction numbers and are not AutoAttack results.

Scope boundary: full ImageNet attack was not run. Gate 2 remains pending because
E1.5 is not part of this run.

## Protocol

- Environment: Python `3.10.8`, torch `2.1.2+cu118`, torchvision `0.16.2+cu118`, RTX 4090.
- Dataset: `cifar100`
- Samples: `10000`
- Mode: `full`
- Models: vanilla `ViT-B-16/laion2b_s34b_b88k`, FARE4 `hf-hub:chs20/FARE4-ViT-B-16-laion2B-s34B-b88K`, TeCoA4 `hf-hub:chs20/TeCoA4-ViT-B-16-laion2B-s34B-b88K`.
- Class order: CIFAR-100 `fine_label_names` from `meta`.
- Text template: `a photo of a {c}.`
- Epsilon domain: `pixel_after_resize_center_crop_before_normalization`
- Normalization: CLIP mean `[0.48145466, 0.4578275, 0.40821073]`, std `[0.26862954, 0.26130258, 0.27577711]`.
- Image branch only; text features fixed.
- Deterministic resize/center-crop; no TTE and no eval-time augmentation.

Command:

```bash
HF_ENDPOINT=https://hf-mirror.com \
bash scripts_autodl/11_e0b_revised_cifar_full.sh \
  --model all \
  --dataset cifar100 \
  --attack pgd20 \
  --eps-list 1,2,4 \
  --batch-size 32 \
  --full
```

Log:

```text
/root/autodl-tmp/robustvlm_outputs/e0b_revised/e0b_revised_full_pgd20_20260612_220922.log
```

## Results

| Model | Attack | Eps | Steps | Step size | Clean acc | Robust acc | Grad ok | Runtime | Peak MB |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| vanilla | PGD-20 | 1/255 | 20 | 0.25/255 | 76.9800 | 0.0000 | True | 630.69s | 4619.2 |
| vanilla | PGD-20 | 2/255 | 20 | 0.5/255 | 76.9800 | 0.0000 | True | 631.16s | 4619.2 |
| vanilla | PGD-20 | 4/255 | 20 | 1/255 | 76.9800 | 0.0000 | True | 632.01s | 4619.2 |
| fare4 | PGD-20 | 1/255 | 20 | 0.25/255 | 48.4500 | 37.6100 | True | 630.97s | 4620.3 |
| fare4 | PGD-20 | 2/255 | 20 | 0.5/255 | 48.4500 | 27.7900 | True | 631.15s | 4620.3 |
| fare4 | PGD-20 | 4/255 | 20 | 1/255 | 48.4500 | 13.2200 | True | 630.69s | 4620.3 |
| tecoa4 | PGD-20 | 1/255 | 20 | 0.25/255 | 44.3500 | 36.6900 | True | 631.93s | 4620.3 |
| tecoa4 | PGD-20 | 2/255 | 20 | 0.5/255 | 44.3500 | 29.4800 | True | 631.62s | 4620.3 |
| tecoa4 | PGD-20 | 4/255 | 20 | 1/255 | 44.3500 | 17.6400 | True | 631.57s | 4620.3 |

## Paste-Ready experiments.csv Rows

```csv
robustvlm_e0b_revised_full_vanilla_pgd20_eps1,repro-RobustVLM,E0b_revised_internal_attack_calibration,vanilla,vanilla LAION2B ViT-B/16,CIFAR-100-test,10000,robust_accuracy,0.0000,percent,NVIDIA GeForce RTX 4090,torch 2.1.2+cu118,/root/autodl-tmp/data/cifar-100-python,PGD-20 eps 1/255 pixel-domain; clean 76.9800
robustvlm_e0b_revised_full_vanilla_pgd20_eps2,repro-RobustVLM,E0b_revised_internal_attack_calibration,vanilla,vanilla LAION2B ViT-B/16,CIFAR-100-test,10000,robust_accuracy,0.0000,percent,NVIDIA GeForce RTX 4090,torch 2.1.2+cu118,/root/autodl-tmp/data/cifar-100-python,PGD-20 eps 2/255 pixel-domain; clean 76.9800
robustvlm_e0b_revised_full_vanilla_pgd20_eps4,repro-RobustVLM,E0b_revised_internal_attack_calibration,vanilla,vanilla LAION2B ViT-B/16,CIFAR-100-test,10000,robust_accuracy,0.0000,percent,NVIDIA GeForce RTX 4090,torch 2.1.2+cu118,/root/autodl-tmp/data/cifar-100-python,PGD-20 eps 4/255 pixel-domain; clean 76.9800
robustvlm_e0b_revised_full_fare4_pgd20_eps1,repro-RobustVLM,E0b_revised_internal_attack_calibration,fare4,FARE4 ViT-B/16 LAION2B,CIFAR-100-test,10000,robust_accuracy,37.6100,percent,NVIDIA GeForce RTX 4090,torch 2.1.2+cu118,/root/autodl-tmp/data/cifar-100-python,PGD-20 eps 1/255 pixel-domain; clean 48.4500
robustvlm_e0b_revised_full_fare4_pgd20_eps2,repro-RobustVLM,E0b_revised_internal_attack_calibration,fare4,FARE4 ViT-B/16 LAION2B,CIFAR-100-test,10000,robust_accuracy,27.7900,percent,NVIDIA GeForce RTX 4090,torch 2.1.2+cu118,/root/autodl-tmp/data/cifar-100-python,PGD-20 eps 2/255 pixel-domain; clean 48.4500
robustvlm_e0b_revised_full_fare4_pgd20_eps4,repro-RobustVLM,E0b_revised_internal_attack_calibration,fare4,FARE4 ViT-B/16 LAION2B,CIFAR-100-test,10000,robust_accuracy,13.2200,percent,NVIDIA GeForce RTX 4090,torch 2.1.2+cu118,/root/autodl-tmp/data/cifar-100-python,PGD-20 eps 4/255 pixel-domain; clean 48.4500
robustvlm_e0b_revised_full_tecoa4_pgd20_eps1,repro-RobustVLM,E0b_revised_internal_attack_calibration,tecoa4,TeCoA4 ViT-B/16 LAION2B,CIFAR-100-test,10000,robust_accuracy,36.6900,percent,NVIDIA GeForce RTX 4090,torch 2.1.2+cu118,/root/autodl-tmp/data/cifar-100-python,PGD-20 eps 1/255 pixel-domain; clean 44.3500
robustvlm_e0b_revised_full_tecoa4_pgd20_eps2,repro-RobustVLM,E0b_revised_internal_attack_calibration,tecoa4,TeCoA4 ViT-B/16 LAION2B,CIFAR-100-test,10000,robust_accuracy,29.4800,percent,NVIDIA GeForce RTX 4090,torch 2.1.2+cu118,/root/autodl-tmp/data/cifar-100-python,PGD-20 eps 2/255 pixel-domain; clean 44.3500
robustvlm_e0b_revised_full_tecoa4_pgd20_eps4,repro-RobustVLM,E0b_revised_internal_attack_calibration,tecoa4,TeCoA4 ViT-B/16 LAION2B,CIFAR-100-test,10000,robust_accuracy,17.6400,percent,NVIDIA GeForce RTX 4090,torch 2.1.2+cu118,/root/autodl-tmp/data/cifar-100-python,PGD-20 eps 4/255 pixel-domain; clean 44.3500
```
