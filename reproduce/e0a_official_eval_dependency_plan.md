# E0a Official Clean Eval Dependency Plan

Status: dependency installation completed in isolated env only.

This public-safe note records the minimal dependency plan for official clean-only RobustVLM / CLIP evaluation diagnosis.

## Current Missing Dependencies

Checked in:

```text
/root/miniconda3/envs/robustvlm-openclip-py310
```

Initially missing packages:

- `robustbench`
- `pycocoevalcap`
- `wandb`

`autoattack` is not installed as a pip package, but this repository contains a local `autoattack/` module. This task must not run AutoAttack.

## Are They Required?

- `robustbench`: required for `CLIP_eval/clip_robustbench.py` and for the clean-only RobustBench sample-selection helper, because both need `robustbench.data.load_clean_dataset`.
- `pycocoevalcap`: required only because `CLIP_benchmark/clip_benchmark/cli.py` imports captioning modules at top level even when running zero-shot classification. It is not conceptually required for clean ImageNet classification, but it is required to run the official CLI unchanged.
- `wandb`: required because `CLIP_eval/clip_robustbench.py` imports `wandb` at top level even when `--wandb False` is used. It is not conceptually required for clean evaluation, but it is required to run the official module unchanged.

## Install Scope

Install only into the isolated conda env:

```bash
/root/miniconda3/envs/robustvlm-openclip-py310/bin/python -m pip install robustbench pycocoevalcap wandb
```

No base environment changes are needed.

No dataset, checkpoint, or cache directory should be committed.

## Stop Condition

If installation stalls for more than 10-15 minutes with no meaningful progress:

- stop installation attempts;
- do not continue to eval;
- record `e0a_dependency_blocked`;
- use a Mac-local wheel/source download and upload fallback only after explicit approval.

## Post-Install Checks

```bash
/root/miniconda3/envs/robustvlm-openclip-py310/bin/python - <<'PY'
import robustbench
import pycocoevalcap
import wandb
print("official eval dependencies import ok")
PY

/root/miniconda3/envs/robustvlm-openclip-py310/bin/python -m pip freeze > reproduce/env_freeze_openclip.txt
```

## Actual Follow-Up

The initial install pulled a newer `wandb` and `protobuf` combination that conflicted with `open-clip-torch==2.19.0`. The isolated env was repaired with:

```bash
/root/miniconda3/envs/robustvlm-openclip-py310/bin/python -m pip install "protobuf<4" "wandb==0.15.12"
/root/miniconda3/envs/robustvlm-openclip-py310/bin/python -m pip install "transformers==4.30.2"
```

Final public-safe import check:

- `open_clip`: ok
- `robustbench`: ok with repo `PYTHONPATH`
- `pycocoevalcap`: ok
- `wandb`: `0.15.12`
- `transformers`: `4.30.2`
- `protobuf`: `3.20.3`
- `pip check`: no broken requirements found
