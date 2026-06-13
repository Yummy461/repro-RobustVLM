# Revised E0b Preflight

Status: `passed`.

This public-safe report records the revised E0b preflight. It does not contain private strategy notes.

Scope boundary: this is internal attack calibration under the local CIFAR-100
protocol, not RobustVLM README reproduction. APGD-lite is not AutoAttack. Full
ImageNet attack was not run. Final revised E0b status is `e0b_revised_passed`;
Gate 2 remains pending because E1.5 is not part of this run.

## Environment

- AutoDL env: `/root/miniconda3/envs/robustvlm-openclip-py310`
- Python: `3.10.8`
- Torch: `2.1.2+cu118`
- Torchvision: `0.16.2+cu118`
- OpenCLIP package: import passed; `__version__` reports `unknown`
- GPU: `NVIDIA GeForce RTX 4090`
- HF endpoint: `https://hf-mirror.com`

## Dataset

- Dataset: CIFAR-100 canonical Python test split.
- Path: `/root/autodl-tmp/data/cifar-100-python`
- Test samples: `10000`
- Class count: `100`
- Class order: CIFAR-100 `fine_label_names` from the canonical Python archive `meta` file.
- Subset gate samples: `512`

## Protocol Checks

- Dataloader path: resize to 224, center crop, then `[0,1]` pixel tensor before normalization.
- Epsilon domain: `pixel_after_resize_center_crop_before_normalization`.
- Text side: fixed zero-shot class features with template `a photo of a {c}.`
- Inference: deterministic single view; no TTE and no eval-time augmentation.
- Attack: image-only untargeted `L_inf` PGD-20, CE loss, random start for eps > 0.
- eps `0` exactly matched clean accuracy for vanilla, FARE4, and TeCoA4.
- Gradients flowed for eps `1/255` on all three models.
- Vanilla random noise eps `1/255` accuracy was `76.5625%`, much weaker than PGD-20 eps `1/255` robust accuracy `0.0000%`.
- FARE4 and TeCoA4 retained clearly non-zero robust accuracy on the same subset.

## Subset Gate Results

| Model | Eps | Clean acc | Robust acc | Runtime |
|---|---:|---:|---:|---:|
| vanilla | 0/255 | 77.5391 | 77.5391 | 1.76s |
| vanilla | 1/255 | 77.5391 | 0.0000 | 32.09s |
| FARE4 | 0/255 | 49.0234 | 49.0234 | 1.52s |
| FARE4 | 1/255 | 49.0234 | 39.2578 | 32.40s |
| TeCoA4 | 0/255 | 45.1172 | 45.1172 | 1.54s |
| TeCoA4 | 1/255 | 45.1172 | 38.8672 | 32.48s |

Log:

```text
/root/autodl-tmp/robustvlm_outputs/e0b_revised/e0b_revised_subset_pgd20_20260612_220706.log
```
