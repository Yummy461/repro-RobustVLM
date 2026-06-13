# RobustVLM / OpenCLIP Attack Protocol

Status: revised E0b internal attack calibration completed. This file keeps earlier sanity checks for traceability and points to the final revised E0b reports.

Scope boundary: revised E0b is internal CIFAR-100 attack calibration under the
protocol below. It is not RobustVLM README reproduction, APGD-lite is not
AutoAttack, and full ImageNet attack was not run. Gate 2 remains pending because
E1.5 is not part of this public-fork run.

## E0a Threat Model

- Task: closed-set zero-shot image classification.
- Backbone lineage: OpenCLIP LAION2B ViT-B/16.
- Robust weights:
  - `hf-hub:chs20/FARE4-ViT-B-16-laion2B-s34B-b88K`;
  - `hf-hub:chs20/TeCoA4-ViT-B-16-laion2B-s34B-b88K`.
- Vanilla counterpart: `ViT-B-16`, `pretrained=laion2b_s34b_b88k`.
- Norm: `L_inf`.
- Epsilon domain: `[0,1]` pixel space after resize/crop and before normalization.
- Image branch only; text features fixed and visible to attacker.
- Deterministic single-view inference; no TTE and no eval-time augmentation.

## Historical RobustVLM CLI Notes

These notes describe the upstream RobustVLM CLIP-only evaluation entry point
observed during preparation. They are retained for traceability and are not the
pass/fail contract for the revised E0b internal calibration.

Observed from RobustVLM CLIP-only evaluation code:

- Entry point: `python -m CLIP_eval.clip_robustbench`.
- Default ImageNet sample count: `5000`.
- Default batch size: `64`.
- Epsilon CLI value is divided by `255`.
- AutoAttack custom setting uses `apgd-ce` and targeted `apgd-t` when `blackbox_only=False`.
- CIFAR paths are supported by the same script, but E0a main-line calibration is ImageNet.

## E0a Preparation Run

- AutoDL image: PyTorch 2.1.2 / Python 3.10 / Ubuntu 22.04 / CUDA 11.8.
- Actual environment: Python `3.10.8`, torch `2.1.2+cu118`, torchvision `0.16.2+cu118`, RTX 4090.
- Dependency route: cloned AutoDL base env to `robustvlm-openclip-py310`; installed OpenCLIP/HF/timm helper dependencies only.
- Cache roots: `HF_HOME=/root/autodl-tmp/model_cache/huggingface`, `TORCH_HOME=/root/autodl-tmp/model_cache/torch`.
- HF access: direct `huggingface.co` timed out; `HF_ENDPOINT=https://hf-mirror.com` was used for public model weight caching.
- Loaded models: vanilla LAION2B ViT-B/16, FARE4 ViT-B/16 LAION2B, TeCoA4 ViT-B/16 LAION2B.
- Loading result: all three models load through OpenCLIP; image size is `[224, 224]`; CLIP mean/std match the LIFT CLIP normalization convention.
- E0a clean calibration: later completed under the revised scope. See `reproduce/e0a_report.md` and `reproduce/e0a_results.jsonl`.

## Attack Sanity Result

Command:

```bash
ROBUSTVLM_REPO_DIR=/root/repro-RobustVLM \
PYTHON_BIN=/root/miniconda3/envs/robustvlm-openclip-py310/bin/python \
bash scripts_autodl/08_attack_sanity_openclip.sh
```

Observed on CIFAR-100 test subset, 512 samples, vanilla LAION2B ViT-B/16:

| Epsilon | Steps | Step size | Clean acc | Robust acc | Grad ok |
|---:|---:|---:|---:|---:|---|
| 0 | 5 | 0 | 77.5391 | 77.5391 | True |
| 1/255 | 5 | 0.5/255 | 77.5391 | 0.0000 | True |
| 2/255 | 5 | 1/255 | 77.5391 | 0.0000 | True |
| 4/255 | 5 | 2/255 | 77.5391 | 0.0000 | True |

Interpretation:

- Epsilon `0` exactly matches clean accuracy, so the no-perturbation path is valid.
- Gradients flow through the image branch.
- Robust accuracy is monotonic non-increasing across tested eps values.
- The eps `1/255` collapse is unusually strong; treat it as an attack-interface stress result, not a formal adversarial robustness result.
- For README-style or official ImageNet robustness reproduction, verify official ImageNet preprocessing, class ordering, loss convention, and RobustVLM's own AutoAttack entry point separately.

## E0b-lite-debug status

Command:

```bash
ROBUSTVLM_REPO_DIR=/root/repro-RobustVLM \
PYTHON_BIN=/root/miniconda3/envs/robustvlm-openclip-py310/bin/python \
bash scripts_autodl/08b_attack_sanity_debug_openclip.sh
```

Observed on CIFAR-100 test subset, 512 samples.

Vanilla LAION2B ViT-B/16 PGD-5 epsilon sweep:

| Epsilon | Step size | Clean acc | Robust acc |
|---:|---:|---:|---:|
| 0 | 0 | 77.5391 | 77.5391 |
| 0.125/255 | 0.0625/255 | 77.5391 | 13.0859 |
| 0.25/255 | 0.125/255 | 77.5391 | 3.7109 |
| 0.5/255 | 0.25/255 | 77.5391 | 0.7812 |
| 1/255 | 0.5/255 | 77.5391 | 0.0000 |
| 2/255 | 1/255 | 77.5391 | 0.0000 |
| 4/255 | 2/255 | 77.5391 | 0.0000 |

Additional debug checks:

- eps `0` exactly matches clean accuracy.
- Dataloader output is confirmed as `[0,1]` pixel-space tensors before normalization.
- PGD perturbations are clamped in pixel space after resize/center-crop and before CLIP normalization.
- First-batch delta absmax at eps `1/255` is `0.00392160`, within eps.
- Random noise is much weaker than PGD: at eps `1/255`, random noise accuracy is `76.7578` while PGD-5 is `0.0000`.
- FGSM/PGD-1 at eps `1/255` is `15.6250`, so collapse is not immediate at one step.
- PGD-2 at eps `1/255` is `3.3203`; PGD-5 and PGD-10 are `0.0000`.
- Independent minimal PGD cross-check at eps `1/255`, steps `5` gives `0.1953`; delta from primary PGD is `0.1953` percentage points.
- FARE4 is more robust on the same subset: clean `49.0234`, PGD-5 eps `1/255` `39.2578`, PGD-5 eps `2/255` `26.9531`.

Verdict: `suspicious_but_reproducible_no_local_bug_found`.

Interpretation: the CIFAR-100 vanilla collapse is still suspicious for a formal robustness claim, but the local attack path is internally consistent. This sanity check was superseded by the revised E0b internal calibration below.

## Revised E0b Internal Calibration Closeout

Final revised E0b artifacts:

- `reproduce/e0b_revised_attack_protocol.md`
- `reproduce/e0b_revised_preflight.md`
- `reproduce/e0b_revised_subset_report.md`
- `reproduce/e0b_revised_full_cifar_report.md`
- `reproduce/e0b_revised_apgd_subset_report.md`
- `reproduce/e0b_revised_results.jsonl`

Preflight:

- CIFAR-100 test split: `10000` images, `100` classes.
- Subset gate: `512` samples.
- eps `0` exactly matched clean accuracy for vanilla, FARE4, and TeCoA4.
- Vanilla random noise eps `1/255`: `76.5625%`, much weaker than PGD-20 eps `1/255`: `0.0000%`.
- FARE4 and TeCoA4 retained clearly non-zero robust accuracy on the same subset.

Full CIFAR-100 PGD-20 internal calibration:

| Model | Clean acc | PGD-20 eps 1/255 | PGD-20 eps 2/255 | PGD-20 eps 4/255 |
|---|---:|---:|---:|---:|
| vanilla LAION2B ViT-B/16 | 76.9800 | 0.0000 | 0.0000 | 0.0000 |
| FARE4 ViT-B/16 LAION2B | 48.4500 | 37.6100 | 27.7900 | 13.2200 |
| TeCoA4 ViT-B/16 LAION2B | 44.3500 | 36.6900 | 29.4800 | 17.6400 |

APGD-lite subset calibration:

- This is an internal APGD-lite CE check, not AutoAttack.
- Subset size: `1000`.

| Model | Clean acc | APGD-lite eps 1/255 | APGD-lite eps 2/255 | APGD-lite eps 4/255 |
|---|---:|---:|---:|---:|
| FARE4 ViT-B/16 LAION2B | 46.5000 | 36.6000 | 25.2000 | 13.3000 |
| TeCoA4 ViT-B/16 LAION2B | 42.4000 | 36.1000 | 29.0000 | 17.5000 |

Status: `e0b_revised_passed`.

README `Additional Models` values are not used as revised E0b pass/fail targets.
If mentioned in E0a notes, they are external zero-shot average references only,
not CIFAR-100 attack calibration references.

## Historical Overnight E0b Launch Plan

Do not run yet.

Draft command shape:

```bash
python -m CLIP_eval.clip_robustbench \
  --clip_model_name ViT-B-16 \
  --pretrained <model-or-checkpoint> \
  --dataset imagenet \
  --imagenet_root <extracted-imagenet-root> \
  --wandb False \
  --norm linf \
  --eps <1|2|4> \
  --batch_size 64 \
  --n_samples_imagenet 5000
```

Historical stop conditions for README-style or official ImageNet robustness runs:

- clean calibration differs from reference by more than `1.0` percentage point;
- epsilon `0` attack sanity does not exactly match clean accuracy;
- vanilla robust accuracy is not generally non-increasing across eps `1/255`, `2/255`, `4/255`;
- preprocessing, tokenizer, or class ordering mismatch remains unresolved.

Expected storage:

- datasets under `/root/autodl-tmp/data`;
- model caches under `/root/autodl-tmp/model_cache`;
- run outputs and raw logs under `/root/autodl-tmp/robustvlm_outputs`;
- public-safe summaries under `reproduce/`.
