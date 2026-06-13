# E0a Official Clean Eval Dry Run

Status: `e0a_reference_version_issue_likely`.

This public-safe note records the official clean-only eval path diagnosis for RobustVLM E0a. No training, PGD, AutoAttack, E0b, E1.5, or private method work was run.

## Dependency Status

The isolated AutoDL env was used:

```text
/root/miniconda3/envs/robustvlm-openclip-py310
```

Minimal official-eval dependencies were installed there only:

```bash
/root/miniconda3/envs/robustvlm-openclip-py310/bin/python -m pip install robustbench pycocoevalcap wandb
/root/miniconda3/envs/robustvlm-openclip-py310/bin/python -m pip install "protobuf<4" "wandb==0.15.12"
/root/miniconda3/envs/robustvlm-openclip-py310/bin/python -m pip install "transformers==4.30.2"
```

Reason for the second command: the newest `wandb` pulled a newer `protobuf` than `open-clip-torch==2.19.0` allows. The env was repaired to `protobuf==3.20.3`, `wandb==0.15.12`, and `pip check` passed.

The refreshed package list is recorded in `reproduce/env_freeze_openclip.txt`.

## Official Paths Inspected

- `CLIP_eval/clip_robustbench.py`: README CLIP ImageNet path, but the unchanged normal entrypoint computes clean accuracy and then proceeds to AutoAttack. The `full_benchmark` path calls RobustBench benchmark. It was not used as a clean-only result.
- `CLIP_benchmark`: README zero-shot path. `clip_benchmark.cli eval --task zeroshot_classification --attack none` is clean-only and can target the prepared ImageFolder ImageNet val directory.

## Dry-Run Result

Command:

```bash
cd /root/repro-RobustVLM
HF_ENDPOINT=https://hf-mirror.com bash scripts_autodl/10_official_clean_eval_dryrun.sh
```

Result:

- Model loading and tokenizer setup reached the metric path.
- The command used `--attack none`.
- The default subset setting `ROBUSTVLM_OFFICIAL_EVAL_N_SAMPLES=11` failed before evaluation because `CLIP_benchmark` calls `dataset.shuffle(...)` when `--n_samples > 0`, while the ImageFolder dataset for `imagenet1k-unverified` has no `.shuffle()` method.
- This is a subset-mode blocker, not a model-loading blocker.

Because the subset path is not usable on ImageFolder without changing official code, the full clean-only path was run with `--n_samples -1`.

## Full Clean-Only Commands

FARE4:

```bash
cd /root/repro-RobustVLM
HF_ENDPOINT=https://hf-mirror.com \
ROBUSTVLM_OFFICIAL_EVAL_MODEL=fare4 \
ROBUSTVLM_OFFICIAL_EVAL_N_SAMPLES=-1 \
bash scripts_autodl/10_official_clean_eval_dryrun.sh
```

TeCoA4:

```bash
cd /root/repro-RobustVLM
HF_ENDPOINT=https://hf-mirror.com \
ROBUSTVLM_OFFICIAL_EVAL_MODEL=tecoa4 \
ROBUSTVLM_OFFICIAL_EVAL_N_SAMPLES=-1 \
bash scripts_autodl/10_official_clean_eval_dryrun.sh
```

Logs stayed outside git:

- `/root/autodl-tmp/robustvlm_outputs/e0a_official_clean_eval_full_fare4.log`
- `/root/autodl-tmp/robustvlm_outputs/e0a_official_clean_eval_full_tecoa4.log`

## Results

| Model | Samples | Attack | Top-1 | Top-5 | Mean per-class recall | README clean ref | Gap pp |
|---|---:|---|---:|---:|---:|---:|---:|
| FARE4 ViT-B/16 LAION2B | 50000 | none | 60.4700 | 84.8200 | 60.4680 | 56.6 | +3.8700 |
| TeCoA4 ViT-B/16 LAION2B | 50000 | none | 64.4820 | 86.7500 | 64.4580 | 51.5 | +12.9820 |

Output JSON:

- `reproduce/e0a_official_clean_eval_fare4_-1.json`
- `reproduce/e0a_official_clean_eval_tecoa4_-1.json`

## Diagnosis Classification

Best current classification: `e0a_reference_version_issue_likely`.

Rationale:

- The official clean-only `CLIP_benchmark` path closely matches the earlier custom full-clean results.
- FARE4 differs from the custom result by only `0.002 pp`.
- TeCoA4 differs from the custom result by only `0.024 pp`.
- Both official clean-only results still miss the README Additional Models references by more than `1.0 pp`.

Remaining issue before official E0b:

- Confirm the exact checkpoint revision and evaluation protocol behind the README Additional Models clean references, especially whether they use a RobustBench 5000-sample subset, a different checkpoint revision, or a different command not represented by the unchanged clean-only `CLIP_benchmark` path.
