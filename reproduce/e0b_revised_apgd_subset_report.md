# Revised E0b APGD-Lite Subset Report

Status: `e0b_revised_passed` supporting APGD-lite subset.

These are internal revised E0b calibration numbers under the fixed public-safe protocol. They are not RobustVLM README reproduction numbers and are not AutoAttack results.

Scope boundary: APGD-lite is not AutoAttack and full ImageNet attack was not
run. This APGD-lite subset is supporting evidence for the final revised E0b
closeout; Gate 2 remains pending because E1.5 is not part of this run.

## Protocol

- Environment: Python `3.10.8`, torch `2.1.2+cu118`, torchvision `0.16.2+cu118`, RTX 4090.
- Dataset: `cifar100`
- Samples: `1000`
- Mode: `subset`
- Attack note: this is internal `APGD-lite CE`, not AutoAttack.
- Class order: CIFAR-100 `fine_label_names` from `meta`.
- Text template: `a photo of a {c}.`
- Epsilon domain: `pixel_after_resize_center_crop_before_normalization`
- Image branch only; text features fixed.
- Deterministic resize/center-crop; no TTE and no eval-time augmentation.

Command:

```bash
HF_ENDPOINT=https://hf-mirror.com \
bash scripts_autodl/11_e0b_revised_cifar_full.sh \
  --model fare4,tecoa4 \
  --dataset cifar100 \
  --attack apgd \
  --eps-list 1,2,4 \
  --max-samples 1000 \
  --batch-size 32
```

Log:

```text
/root/autodl-tmp/robustvlm_outputs/e0b_revised/e0b_revised_subset_apgd_20260612_234439.log
```

## Results

| Model | Attack | Eps | Steps | Step size | Clean acc | Robust acc | Grad ok | Runtime | Peak MB |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| fare4 | APGD-lite CE | 1/255 | 20 |  | 46.5000 | 36.6000 | True | 63.61s | 4637.6 |
| fare4 | APGD-lite CE | 2/255 | 20 |  | 46.5000 | 25.2000 | True | 63.38s | 4637.6 |
| fare4 | APGD-lite CE | 4/255 | 20 |  | 46.5000 | 13.3000 | True | 63.47s | 4637.6 |
| tecoa4 | APGD-lite CE | 1/255 | 20 |  | 42.4000 | 36.1000 | True | 63.48s | 4638.7 |
| tecoa4 | APGD-lite CE | 2/255 | 20 |  | 42.4000 | 29.0000 | True | 63.54s | 4638.7 |
| tecoa4 | APGD-lite CE | 4/255 | 20 |  | 42.4000 | 17.5000 | True | 63.49s | 4638.7 |

## Paste-Ready experiments.csv Rows

```csv
robustvlm_e0b_revised_subset_fare4_apgd_eps1,repro-RobustVLM,E0b_revised_internal_attack_calibration,fare4,FARE4 ViT-B/16 LAION2B,CIFAR-100-test,1000,robust_accuracy,36.6000,percent,NVIDIA GeForce RTX 4090,torch 2.1.2+cu118,/root/autodl-tmp/data/cifar-100-python,APGD-lite CE eps 1/255 pixel-domain; clean 46.5000
robustvlm_e0b_revised_subset_fare4_apgd_eps2,repro-RobustVLM,E0b_revised_internal_attack_calibration,fare4,FARE4 ViT-B/16 LAION2B,CIFAR-100-test,1000,robust_accuracy,25.2000,percent,NVIDIA GeForce RTX 4090,torch 2.1.2+cu118,/root/autodl-tmp/data/cifar-100-python,APGD-lite CE eps 2/255 pixel-domain; clean 46.5000
robustvlm_e0b_revised_subset_fare4_apgd_eps4,repro-RobustVLM,E0b_revised_internal_attack_calibration,fare4,FARE4 ViT-B/16 LAION2B,CIFAR-100-test,1000,robust_accuracy,13.3000,percent,NVIDIA GeForce RTX 4090,torch 2.1.2+cu118,/root/autodl-tmp/data/cifar-100-python,APGD-lite CE eps 4/255 pixel-domain; clean 46.5000
robustvlm_e0b_revised_subset_tecoa4_apgd_eps1,repro-RobustVLM,E0b_revised_internal_attack_calibration,tecoa4,TeCoA4 ViT-B/16 LAION2B,CIFAR-100-test,1000,robust_accuracy,36.1000,percent,NVIDIA GeForce RTX 4090,torch 2.1.2+cu118,/root/autodl-tmp/data/cifar-100-python,APGD-lite CE eps 1/255 pixel-domain; clean 42.4000
robustvlm_e0b_revised_subset_tecoa4_apgd_eps2,repro-RobustVLM,E0b_revised_internal_attack_calibration,tecoa4,TeCoA4 ViT-B/16 LAION2B,CIFAR-100-test,1000,robust_accuracy,29.0000,percent,NVIDIA GeForce RTX 4090,torch 2.1.2+cu118,/root/autodl-tmp/data/cifar-100-python,APGD-lite CE eps 2/255 pixel-domain; clean 42.4000
robustvlm_e0b_revised_subset_tecoa4_apgd_eps4,repro-RobustVLM,E0b_revised_internal_attack_calibration,tecoa4,TeCoA4 ViT-B/16 LAION2B,CIFAR-100-test,1000,robust_accuracy,17.5000,percent,NVIDIA GeForce RTX 4090,torch 2.1.2+cu118,/root/autodl-tmp/data/cifar-100-python,APGD-lite CE eps 4/255 pixel-domain; clean 42.4000
```
