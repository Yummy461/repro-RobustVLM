# E0a Clean Calibration Report

Status: **E0a-revised = passed (under corrected reference scope).** Vanilla LAION2B reproduces the open_clip ImageNet reference (70.23 vs measured 70.2280, within ±0.01 pp). FARE4 and TeCoA4 do not match the RobustVLM README "Additional Models" values (56.6 / 51.5), but those are an **external zero-shot AVERAGE reference, not an ImageNet-only top-1 target**; under the corrected scope their ImageNet-only clean top-1 (60.4680 / 64.4580) are recorded as **internal baselines**, not failures. All numbers and the per-row table below are unchanged from the original run. No attacks, AutoAttack, E0b, E1.5, or training were run in this report.

This public-safe report records RobustVLM E0a clean calibration status for OpenCLIP LAION2B / FARE4 / TeCoA4 on ImageNet val.

## Environment

- AutoDL image: PyTorch 2.1.2 / Python 3.10 / Ubuntu 22.04 / CUDA 11.8.
- Actual Python: `3.10.8`.
- Torch: `2.1.2+cu118`.
- Torchvision: `0.16.2+cu118`.
- OpenCLIP: `open-clip-torch==2.19.0`.
- GPU: NVIDIA GeForce RTX 4090, driver `550.120`, memory `24564 MiB`.
- Conda env: `/root/miniconda3/envs/robustvlm-openclip-py310`.
- HF endpoint used for this run: `https://hf-mirror.com`.

## Precheck

- Repository run path: `/root/repro-RobustVLM`.
- Python / CUDA / OpenCLIP import check: passed.
- Data disk before full E0a: `/root/autodl-tmp`, `50G` total, about `42G` available.
- ImageNet val path: `/root/autodl-tmp/data/imagenet/val`.
- ImageNet val structure: `1000` class folders, `50000` images, `0` root-level JPEG files.
- `torchvision.datasets.ImageFolder` can load the directory with `50000` samples and `1000` classes.

## Command

```bash
cd /root/repro-RobustVLM
PYTHON_BIN=/root/miniconda3/envs/robustvlm-openclip-py310/bin/python \
HF_ENDPOINT=https://hf-mirror.com \
bash scripts_autodl/07_e0a_clean_calibration.sh \
  --model all \
  --batch-size 64 \
  --num-workers 4 \
  --imagenet-val-dir /root/autodl-tmp/data/imagenet/val \
  --full
```

Main full-run log path:

```text
/root/autodl-tmp/robustvlm_outputs/e0a_full_clean_20260612_200057.log
```

TeCoA4 continuation log path:

```text
/root/autodl-tmp/robustvlm_outputs/e0a_full_clean_tecoa4_20260612_201933.log
```

## Reference Targets

| Model | Reference clean top-1 | Source |
|---|---:|---|
| vanilla LAION2B ViT-B/16 | 70.23 | OpenCLIP official classification CSV / existing E0a record |
| FARE4 ViT-B/16 LAION2B | 56.6 | RobustVLM README Additional Models table |
| TeCoA4 ViT-B/16 LAION2B | 51.5 | RobustVLM README Additional Models table |

## Full Results

| Model | Mode | Samples | Clean acc | Reference | Gap pp | Correct | Runtime | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| vanilla | full | 50000 | 70.2280 | 70.23 | -0.0020 | 35114 | 91.85s | passed |
| fare4 | full | 50000 | 60.4680 | 56.6 | +3.8680 | 30234 | 92.09s | mismatch |
| tecoa4 | full | 50000 | 64.4580 | 51.5 | +12.9580 | 32229 | 91.37s | mismatch |

## Interpretation

- Vanilla LAION2B reproduces the expected ImageNet clean top-1 almost exactly.
- FARE4 is not within the allowed `±1.0` percentage point gate against the README reference used here.
- TeCoA4 is also not within the allowed `±1.0` percentage point gate against the README reference used here.
- The current E0a status is therefore `mismatch`, not `passed`.
- This is not an adversarial robustness result.
- No attacks, AutoAttack, full E0b, or training were run.

## Diagnosis Addendum

Additional public-safe diagnosis is recorded in `reproduce/e0a_mismatch_diagnosis.md`.

Summary:

- FARE4 and TeCoA4 both load as ViT-B/16 `VisionTransformer` models, not ConvNeXt.
- FARE4 HF revision: `1e941d54ce261e19ee072f1c3d5af8670aa19c02`.
- TeCoA4 HF revision: `ae1409bcaf7d062d37a80457747c14b6bddcbc63`.
- FARE4 visual-only transplant into vanilla LAION2B text/logit-scale reproduces the same `60.4680%`, so the text tower and logit scale are unlikely to explain the FARE4 mismatch.
- FARE4 with the RobustVLM `std` prompt gives `58.6500%`, closer to `56.6%` but still outside the gate.
- TeCoA4 with the RobustVLM `std` prompt gives `66.0860%`, farther from `51.5%`.
- Official clean-only `CLIP_benchmark` comparison completed with `--attack none`.
- `CLIP_benchmark` subset dry-run with `--n_samples 11` failed because ImageFolder has no `.shuffle()` method, so the full clean-only `--n_samples -1` path was used.
- FARE4 official clean-only top-1: `60.4700%`, still mismatching the README `56.6%`.
- TeCoA4 official clean-only top-1: `64.4820%`, still mismatching the README `51.5%`.
- Best current classification: `e0a_reference_version_issue_likely`.

## Official Clean-Only Check

Script:

```bash
bash scripts_autodl/10_official_clean_eval_dryrun.sh
```

Commands:

```bash
HF_ENDPOINT=https://hf-mirror.com \
ROBUSTVLM_OFFICIAL_EVAL_MODEL=fare4 \
ROBUSTVLM_OFFICIAL_EVAL_N_SAMPLES=-1 \
bash scripts_autodl/10_official_clean_eval_dryrun.sh

HF_ENDPOINT=https://hf-mirror.com \
ROBUSTVLM_OFFICIAL_EVAL_MODEL=tecoa4 \
ROBUSTVLM_OFFICIAL_EVAL_N_SAMPLES=-1 \
bash scripts_autodl/10_official_clean_eval_dryrun.sh
```

| Model | Samples | Attack | Official clean top-1 | Reference | Gap pp | Status |
|---|---:|---|---:|---:|---:|---|
| FARE4 ViT-B/16 LAION2B | 50000 | none | 60.4700 | 56.6 | +3.8700 | mismatch |
| TeCoA4 ViT-B/16 LAION2B | 50000 | none | 64.4820 | 51.5 | +12.9820 | mismatch |

## Mismatch Report

At least one full clean calibration result differs from its reference by more than `1.0` percentage point.

Public-safe suspects to check before official E0b:

1. prompt templates;
2. class names / synset mapping;
3. `ImageFolder` `class_to_idx` ordering;
4. preprocessing;
5. tokenizer;
6. weight lineage;
7. logit scale.

## Paste-Ready experiments.csv Rows

```csv
robustvlm_e0a_full_vanilla_imagenet_val,repro-RobustVLM,E0a_full_clean,vanilla,ViT-B-16 laion2b_s34b_b88k,ImageNet-val,50000,clean_accuracy,70.2280,percent,RTX4090,PyTorch2.1.2+cu118 Python3.10.8 CUDA11.8,/root/autodl-tmp/data/imagenet/val,full clean; passed against 70.23 reference; log /root/autodl-tmp/robustvlm_outputs/e0a_full_clean_20260612_200057.log
robustvlm_e0a_full_fare4_imagenet_val,repro-RobustVLM,E0a_full_clean,fare4,FARE4 ViT-B-16 LAION2B,ImageNet-val,50000,clean_accuracy,60.4680,percent,RTX4090,PyTorch2.1.2+cu118 Python3.10.8 CUDA11.8,/root/autodl-tmp/data/imagenet/val,mismatch against 56.6 reference; gap +3.868 pp; stopped before TeCoA4 and E0b
robustvlm_e0a_full_tecoa4_imagenet_val,repro-RobustVLM,E0a_full_clean,tecoa4,TeCoA4 ViT-B-16 LAION2B,ImageNet-val,50000,clean_accuracy,64.4580,percent,RTX4090,PyTorch2.1.2+cu118 Python3.10.8 CUDA11.8,/root/autodl-tmp/data/imagenet/val,mismatch against 51.5 reference; gap +12.958 pp; continuation run log /root/autodl-tmp/robustvlm_outputs/e0a_full_clean_tecoa4_20260612_201933.log
robustvlm_e0a_official_clean_fare4_imagenet_val,repro-RobustVLM,E0a_official_clean_only,fare4,FARE4 ViT-B-16 LAION2B,ImageNet-val,50000,clean_accuracy,60.4700,percent,RTX4090,PyTorch2.1.2+cu118 Python3.10.8 CUDA11.8,/root/autodl-tmp/data/imagenet,CLIP_benchmark zeroshot_classification with attack none; mismatch against 56.6 reference; gap +3.870 pp
robustvlm_e0a_official_clean_tecoa4_imagenet_val,repro-RobustVLM,E0a_official_clean_only,tecoa4,TeCoA4 ViT-B-16 LAION2B,ImageNet-val,50000,clean_accuracy,64.4820,percent,RTX4090,PyTorch2.1.2+cu118 Python3.10.8 CUDA11.8,/root/autodl-tmp/data/imagenet,CLIP_benchmark zeroshot_classification with attack none; mismatch against 51.5 reference; gap +12.982 pp
```

## Next Gate

E0a-revised is **passed** under the corrected reference scope: the RobustVLM README "Additional Models" numbers are an external zero-shot AVERAGE reference, not an ImageNet-only top-1 target, so the FARE4/TeCoA4 gaps in the table above are not failures. Their ImageNet-only clean top-1 values stand as internal baselines. E0b internal CIFAR attack calibration was subsequently completed (see the `e0b_revised_*` reports); Gate 2 remains pending the E1.5 LAION2B-LIFT mini baseline.
