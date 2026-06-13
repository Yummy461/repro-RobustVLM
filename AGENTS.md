# AGENTS.md

## Repository Purpose

This is a public fork of RobustVLM for public-safe reproduction and protocol
calibration only.

Allowed content:

- repository audits;
- environment notes;
- AutoDL image cards;
- smoke-test scripts;
- public-safe evaluation commands;
- public metrics and engineering reproducibility notes;
- small utilities for checking environment, paths, checkpoint loading, and attack
  protocol behavior.

Forbidden content:

- private thesis strategy;
- module A or module B design notes;
- unpublished method ideas;
- private route decisions;
- private dataset paths;
- credentials, tokens, SSH keys, or passwords;
- datasets, checkpoints, TensorBoard or wandb runs, raw large logs, or full
  output directories.

## Working Rules

1. Do not modify RobustVLM core algorithm code unless the user explicitly asks.
2. Do not train models in this public fork.
3. Do not run full PGD, AutoAttack, VLM, or large-dataset evaluations unless the
   user explicitly requests them.
4. Do not download ImageNet, COCO, TextVQA, OpenFlamingo, LLaVA, robust CLIP
   checkpoints, or other large assets without explicit user confirmation.
5. Keep dataset roots, checkpoint paths, and output paths configurable with
   placeholders or environment variables.
6. Commit only lightweight scripts and documentation.
7. Keep all outputs public-safe and reproducible from public commands.

## File Placement

- Public-safe reports belong under `reproduce/`.
- AutoDL shell scripts belong under `scripts_autodl/`.
- Public-safe helper utilities belong under `tools/`.
- Temporary run outputs should stay outside git-tracked files.

## Remote Sync Rules

When syncing local public-safe files to AutoDL or another remote run directory,
preserve the repository top-level directory layout exactly.

Do not copy mixed file categories into one remote destination. In particular,
never run a single `scp` command that sends files from `reproduce/`,
`scripts_autodl/`, and `tools/` to the same remote directory.

Use separate sync commands by destination:

- `reproduce/*` -> remote `reproduce/`;
- `scripts_autodl/*` -> remote `scripts_autodl/`;
- `tools/*` -> remote `tools/`.

After any remote sync, run the placement checker locally and, when feasible, on
the remote copy:

```bash
python tools/check_public_file_placement.py
```

If a sync mistake puts helper scripts under `reproduce/`, remove only the
misplaced files that were just copied, then re-sync by top-level directory.

## Core Code Boundary

Treat upstream RobustVLM implementation directories as read-only for reproduction
bookkeeping:

- `train/`
- `CLIP_eval/`
- `vlm_eval/`
- `autoattack/`
- `CLIP_benchmark/`
- `llava/`
- `open_flamingo/`

Inspection is allowed. Reproduction scaffolding should be added outside these
directories unless the user explicitly approves a core-code change.

## Private Control Boundary

Private reproduction decisions, thesis-route analysis, engineering-base
classification, module planning, and future method strategy belong only in the
private control repository, not in this public fork.

When finished with a public-fork reproduction task, report:

1. files created or modified;
2. exact commands used;
3. metrics, if any were produced;
4. output or log paths;
5. what should be recorded privately in the control repository.
