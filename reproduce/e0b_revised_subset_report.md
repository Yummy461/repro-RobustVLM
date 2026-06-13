# Revised E0b Subset Report

Status: `e0b_revised_passed` supporting subset gate.

These are internal revised E0b calibration numbers under the fixed public-safe protocol. They are not RobustVLM README reproduction numbers and are not AutoAttack results.

Scope boundary: full ImageNet attack was not run. This subset gate is supporting
evidence for the final revised E0b closeout; Gate 2 remains pending because
E1.5 is not part of this run.

## Protocol

- Dataset: `cifar100`
- Samples: `512`
- Mode: `subset`
- Class order: CIFAR-100 `fine_label_names` from `meta`.
- Text template: `a photo of a {c}.`
- Epsilon domain: `pixel_after_resize_center_crop_before_normalization`
- Image branch only; text features fixed.
- Deterministic resize/center-crop; no TTE and no eval-time augmentation.

## Results

| Model | Attack | Eps | Steps | Step size | Clean acc | Robust acc | Grad ok | Runtime | Peak MB |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|
| vanilla | PGD-20 | 0/255 | 20 | 0/255 | 77.5391 | 77.5391 | True | 1.76s | 877.6 |
| vanilla | PGD-20 | 1/255 | 20 | 0.25/255 | 77.5391 | 0.0000 | True | 32.09s | 4619.2 |
| fare4 | PGD-20 | 0/255 | 20 | 0/255 | 49.0234 | 49.0234 | True | 1.52s | 886.3 |
| fare4 | PGD-20 | 1/255 | 20 | 0.25/255 | 49.0234 | 39.2578 | True | 32.40s | 4620.3 |
| tecoa4 | PGD-20 | 0/255 | 20 | 0/255 | 45.1172 | 45.1172 | True | 1.54s | 886.3 |
| tecoa4 | PGD-20 | 1/255 | 20 | 0.25/255 | 45.1172 | 38.8672 | True | 32.48s | 4620.3 |

## Paste-Ready experiments.csv Rows

```csv
robustvlm_e0b_revised_subset_vanilla_pgd20_eps0,repro-RobustVLM,E0b_revised_internal_attack_calibration,vanilla,vanilla LAION2B ViT-B/16,CIFAR-100-test,512,robust_accuracy,77.5391,percent,NVIDIA GeForce RTX 4090,torch 2.1.2+cu118,/root/autodl-tmp/data/cifar-100-python,PGD-20 eps 0/255 pixel-domain; clean 77.5391
robustvlm_e0b_revised_subset_vanilla_pgd20_eps1,repro-RobustVLM,E0b_revised_internal_attack_calibration,vanilla,vanilla LAION2B ViT-B/16,CIFAR-100-test,512,robust_accuracy,0.0000,percent,NVIDIA GeForce RTX 4090,torch 2.1.2+cu118,/root/autodl-tmp/data/cifar-100-python,PGD-20 eps 1/255 pixel-domain; clean 77.5391
robustvlm_e0b_revised_subset_fare4_pgd20_eps0,repro-RobustVLM,E0b_revised_internal_attack_calibration,fare4,FARE4 ViT-B/16 LAION2B,CIFAR-100-test,512,robust_accuracy,49.0234,percent,NVIDIA GeForce RTX 4090,torch 2.1.2+cu118,/root/autodl-tmp/data/cifar-100-python,PGD-20 eps 0/255 pixel-domain; clean 49.0234
robustvlm_e0b_revised_subset_fare4_pgd20_eps1,repro-RobustVLM,E0b_revised_internal_attack_calibration,fare4,FARE4 ViT-B/16 LAION2B,CIFAR-100-test,512,robust_accuracy,39.2578,percent,NVIDIA GeForce RTX 4090,torch 2.1.2+cu118,/root/autodl-tmp/data/cifar-100-python,PGD-20 eps 1/255 pixel-domain; clean 49.0234
robustvlm_e0b_revised_subset_tecoa4_pgd20_eps0,repro-RobustVLM,E0b_revised_internal_attack_calibration,tecoa4,TeCoA4 ViT-B/16 LAION2B,CIFAR-100-test,512,robust_accuracy,45.1172,percent,NVIDIA GeForce RTX 4090,torch 2.1.2+cu118,/root/autodl-tmp/data/cifar-100-python,PGD-20 eps 0/255 pixel-domain; clean 45.1172
robustvlm_e0b_revised_subset_tecoa4_pgd20_eps1,repro-RobustVLM,E0b_revised_internal_attack_calibration,tecoa4,TeCoA4 ViT-B/16 LAION2B,CIFAR-100-test,512,robust_accuracy,38.8672,percent,NVIDIA GeForce RTX 4090,torch 2.1.2+cu118,/root/autodl-tmp/data/cifar-100-python,PGD-20 eps 1/255 pixel-domain; clean 45.1172
```
