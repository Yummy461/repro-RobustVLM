# OpenCLIP E0b-Lite Attack Debug

This public-safe report debugs the CIFAR-100 attack sanity pipeline. It is not a formal RobustVLM robustness result.

## Run Context

- Dataset status: `{"path": "/root/autodl-tmp/data/cifar-100-python", "status": "exists"}`
- Dataset: CIFAR-100 test subset, `512` samples
- Models: vanilla LAION2B ViT-B/16 and FARE4 ViT-B/16 LAION2B
- Text side: fixed zero-shot class features with template `a photo of a {c}.`
- Epsilon domain: pixel space after resize/center-crop and before CLIP normalization
- Attack: untargeted image-only L_inf PGD; no AutoAttack and no full E0b

## First Batch Pixel / Normalization Checks

| Tensor | Min | Max | Mean | Std | Shape |
|---|---:|---:|---:|---:|---|
| x_pixel | 0.00000000 | 1.00000000 | 0.50537235 | 0.27131277 | `[32, 3, 224, 224]` |
| x_adv_pixel | 0.00000000 | 1.00000000 | 0.50533748 | 0.27123442 | `[32, 3, 224, 224]` |
| delta | -0.00392157 | 0.00392160 | -0.00003487 | 0.00282582 | `[32, 3, 224, 224]` |
| normalized_input | -1.79226255 | 2.14589691 | 0.20860890 | 1.00670707 | `[32, 3, 224, 224]` |
| normalized_adv_input | -1.79226255 | 2.14589691 | 0.20847911 | 1.00641227 | `[32, 3, 224, 224]` |

- Dataloader returns [0,1] pixel tensors: `True`
- PGD adversarial pixels are clamped to [0,1]: `True`
- First-batch delta absmax: `0.00392160`
- First-batch delta within eps=1/255: `True`

## Vanilla Epsilon Sweep

| Epsilon | Steps | Step size | Clean acc | Attacked/noisy acc | Grad ok | Runtime |
|---:|---:|---:|---:|---:|---|---:|
| 0 | 5 | 0 | 77.5391 | 77.5391 | True | 9.31s |
| 0.125/255 | 5 | 0.0625/255 | 77.5391 | 13.0859 | True | 9.44s |
| 0.25/255 | 5 | 0.125/255 | 77.5391 | 3.7109 | True | 9.37s |
| 0.5/255 | 5 | 0.25/255 | 77.5391 | 0.7812 | True | 9.41s |
| 1/255 | 5 | 0.5/255 | 77.5391 | 0.0000 | True | 9.45s |
| 2/255 | 5 | 1/255 | 77.5391 | 0.0000 | True | 9.46s |
| 4/255 | 5 | 2/255 | 77.5391 | 0.0000 | True | 9.46s |

## Vanilla Steps Sweep at eps=1/255

| Model | Setting | Epsilon | Steps | Step size | Clean acc | Attacked/noisy acc | Grad ok | Runtime |
|---|---|---:|---:|---:|---:|---:|---|---:|
| vanilla | fgsm | 1/255 | 1 | 1/255 | 77.5391 | 15.6250 | True | 3.21s |
| vanilla | pgd_2 | 1/255 | 2 | 0.5/255 | 77.5391 | 3.3203 | True | 4.82s |
| vanilla | pgd_5 | 1/255 | 5 | 0.5/255 | 77.5391 | 0.0000 | True | 9.53s |
| vanilla | pgd_10 | 1/255 | 10 | 0.5/255 | 77.5391 | 0.0000 | True | 17.27s |

## Vanilla Random Noise Baselines

| Model | Setting | Epsilon | Steps | Step size | Clean acc | Attacked/noisy acc | Grad ok | Runtime |
|---|---|---:|---:|---:|---:|---:|---|---:|
| vanilla | random_noise | 1/255 | 0 | 0.5/255 | 77.5391 | 76.7578 | True | 1.76s |
| vanilla | random_noise | 2/255 | 0 | 1/255 | 77.5391 | 72.6562 | True | 1.65s |
| vanilla | random_noise | 4/255 | 0 | 2/255 | 77.5391 | 69.1406 | True | 1.65s |

## Independent PGD Cross-Check

| Model | Setting | Epsilon | Steps | Step size | Clean acc | Attacked/noisy acc | Grad ok | Runtime |
|---|---|---:|---:|---:|---:|---:|---|---:|
| vanilla | pgd_minimal | 1/255 | 5 | 0.5/255 | 77.5391 | 0.1953 | True | 11.55s |

## FARE4 Small Sanity

| Model | Setting | Epsilon | Steps | Step size | Clean acc | Attacked/noisy acc | Grad ok | Runtime |
|---|---|---:|---:|---:|---:|---:|---|---:|
| fare4 | pgd_primary | 1/255 | 5 | 0.5/255 | 49.0234 | 39.2578 | True | 9.39s |
| fare4 | pgd_primary | 2/255 | 5 | 1/255 | 49.0234 | 26.9531 | True | 9.36s |

## Debug Answers

1. eps=0 exactly matches clean: `True`.
2. Random noise is much weaker than PGD: `True`.
3. Collapse pattern as eps increases: `gradual`.
4. FGSM/PGD-1 already collapses to near 0: `False`.
5. FARE4 behaves more robustly than vanilla on this subset: `True`.
6. Inputs are pixel-space tensors before normalization: `True`.
7. Epsilon is applied after resize/crop and before normalization: `True`.
8. Previous eps=1/255 -> 0% status: `suspicious_but_reproducible_no_local_bug_found`. Independent PGD delta: `0.1953` pp.
9. Remaining official E0b issue: Official ImageNet E0a clean calibration and RobustVLM CLIP_eval/AutoAttack path remain unverified because extracted ImageNet val is missing.

## Verdict

`suspicious_but_reproducible_no_local_bug_found`

The local attack path is internally consistent if eps=0 matches clean, random noise is weaker than PGD, the independent PGD cross-check matches, and first-batch clamp/statistics are valid. The very fast collapse remains suspicious for a formal robustness claim and must be rechecked through the official ImageNet/RobustVLM evaluation path once extracted ImageNet val is available.
