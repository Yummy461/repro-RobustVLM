# E0a Mismatch Diagnosis

Status: `e0a_reference_version_issue_likely`.

This public-safe note diagnoses why the current OpenCLIP/HF E0a clean calibration does not match the RobustVLM README Additional Models clean references. It does not contain private thesis strategy or method design.

## Preserved Mismatch Record

- Vanilla LAION2B ViT-B/16 full ImageNet clean passed almost exactly: `70.2280%` vs reference `70.23%`, gap `-0.0020 pp`.
- FARE4 ViT-B/16 LAION2B full ImageNet clean mismatched: `60.4680%` vs README reference `56.6%`, gap `+3.8680 pp`.
- TeCoA4 ViT-B/16 LAION2B full ImageNet clean also mismatched: `64.4580%` vs README reference `51.5%`, gap `+12.9580 pp`.
- E0a remains blocked because not all required clean numbers are within `±1.0 pp` of their intended references.
- E0b cannot start because the clean-reference gate has not passed.

Main command:

```bash
PYTHON_BIN=/root/miniconda3/envs/robustvlm-openclip-py310/bin/python \
HF_ENDPOINT=https://hf-mirror.com \
bash scripts_autodl/07_e0a_clean_calibration.sh \
  --model all \
  --batch-size 64 \
  --num-workers 4 \
  --imagenet-val-dir /root/autodl-tmp/data/imagenet/val \
  --full
```

Logs:

- `/root/autodl-tmp/robustvlm_outputs/e0a_full_clean_20260612_200057.log`
- `/root/autodl-tmp/robustvlm_outputs/e0a_full_clean_tecoa4_20260612_201933.log`

## Official RobustVLM Evaluation Path

Relevant public repo paths:

- `CLIP_eval/clip_robustbench.py`
- `CLIP_eval/eval_utils.py`
- `CLIP_benchmark/clip_benchmark/cli.py`
- `CLIP_benchmark/clip_benchmark/models/open_clip.py`
- `CLIP_benchmark/clip_benchmark/metrics/zeroshot_classification.py`

Observed facts:

- The README says the provided checkpoints correspond to the CLIP vision encoder; the example loads a full CLIP model and then calls `model.visual.load_state_dict(checkpoint)`.
- `CLIP_eval/eval_utils.py` first creates `open_clip.create_model_and_transforms(clip_model_name, pretrained='openai')`, then tries to load the provided checkpoint into `model.visual`.
- If visual-only loading fails, `eval_utils.py` falls back to loading a whole model through `open_clip.create_model_and_transforms(clip_model_name, pretrained=pretrained, force_quick_gelu=True)`.
- `CLIP_eval/clip_robustbench.py` normalizes image features with `model.encode_image(..., normalize=True)`.
- `CLIP_eval/clip_robustbench.py` normalizes text features for both `std` and `ensemble` templates.
- `CLIP_eval/clip_robustbench.py` multiplies logits by `model.logit_scale.exp()` when `--logit_scale True`, which is the default.
- `CLIP_eval/clip_robustbench.py` defaults to `--template ensemble`.
- The ensemble templates come from `CLIP_eval/zeroshot-templates.json`, key `imagenet1k`.
- `CLIP_benchmark/clip_benchmark/datasets/en_zeroshot_classification_templates.json` has the same 80 ImageNet templates as `CLIP_eval/zeroshot-templates.json`.
- `CLIP_eval/clip_robustbench.py` defaults to `--n_samples_imagenet 5000`, not full `50000` ImageNet val.

Official clean-only follow-up:

- Minimal official-eval dependencies were installed in the isolated AutoDL env only.
- `CLIP_eval/clip_robustbench.py` still was not used for clean-only reporting because the normal entrypoint proceeds into AutoAttack and the `full_benchmark` path calls RobustBench benchmark.
- `CLIP_benchmark` zero-shot classification was used with `--attack none` as the official clean-only path that can run unchanged on the prepared ImageFolder ImageNet val directory.
- `CLIP_benchmark` subset dry-run with `--n_samples 11` failed because `ImageFolder` does not implement `.shuffle()`.
- Full ImageNet clean-only `CLIP_benchmark` runs completed for FARE4 and TeCoA4.


## Official Clean Eval Path Inspection

Current public-code inspection found two relevant official or official-adjacent clean evaluation paths.

1. `CLIP_eval.clip_robustbench`
   - README CLIP ImageNet evaluation points to `python -m CLIP_eval.clip_robustbench ...`.
   - The script defaults to `--dataset imagenet`, `--template ensemble`, `--n_samples_imagenet 5000`, `--batch_size 64`, and `--logit_scale True`.
   - The script uses `CLIP_eval/zeroshot-templates.json`, key `imagenet1k`, for ensemble prompts.
   - Class names come from `open_flamingo.eval.classification_utils.IMAGENET_1K_CLASS_ID_TO_LABEL`.
   - Images are transformed by the OpenCLIP validation preprocess without the final Normalize; the wrapper applies Normalize internally before `encode_image`.
   - Image features are normalized with `model.encode_image(..., normalize=True)`.
   - Text features are normalized; ensemble text features are averaged and re-normalized per class.
   - Logits are multiplied by `model.logit_scale.exp()` by default.
   - The normal non-`full_benchmark` path computes clean accuracy first but then constructs and runs AutoAttack, so it is not a clean-only command as written.
   - The `full_benchmark` path calls RobustBench `benchmark(...)`, which is also not clean-only.
   - The script explicitly handles text feature shapes only for `ViT-B-32` and `ViT-L-14`; `ViT-B-16` would need a clean-only wrapper or a non-core patch to avoid the shape guard.

2. `CLIP_benchmark` zero-shot classification
   - README CLIP Zero-Shot evaluation points to `CLIP_benchmark` scripts.
   - `clip_benchmark.cli eval --task zeroshot_classification --attack none` is clean-only in its metric path.
   - `imagenet1k-unverified` uses `ImageFolder(root=os.path.join(root, split))`, so it can target the prepared `/root/autodl-tmp/data/imagenet/val` via `--dataset_root /root/autodl-tmp/data/imagenet --split val`.
   - English ImageNet class names come from `CLIP_benchmark/clip_benchmark/datasets/en_classnames.json`.
   - English ImageNet templates come from `CLIP_benchmark/clip_benchmark/datasets/en_zeroshot_classification_templates.json`; this 80-template list matches `CLIP_eval/zeroshot-templates.json`.
   - When `--n_samples > 0`, the dataset is shuffled before truncation. This differs from the earlier custom subset runner.
   - The metric path uses `features = F.normalize(model.encode_image(normalize(images)), dim=-1)` and `logits = 100. * features @ classifier`.
   - This differs slightly from direct OpenCLIP `model.logit_scale.exp()`, though current B/16 models have `logit_scale.exp() ~= 100`.

Current inspection does not prove whether the README Additional Models clean values are full 50k ImageNet, RobustBench 5000-sample subset, CLIP_benchmark subset/full, or another follow-up-project protocol. The strongest actionable difference is that the official RobustBench-oriented script defaults to `5000` ImageNet samples, while our earlier E0a full clean used all `50000` val images.

## Official CLIP_benchmark Clean-Only Execution

Script:

```bash
bash scripts_autodl/10_official_clean_eval_dryrun.sh
```

FARE4 full command:

```bash
HF_ENDPOINT=https://hf-mirror.com \
ROBUSTVLM_OFFICIAL_EVAL_MODEL=fare4 \
ROBUSTVLM_OFFICIAL_EVAL_N_SAMPLES=-1 \
bash scripts_autodl/10_official_clean_eval_dryrun.sh
```

TeCoA4 full command:

```bash
HF_ENDPOINT=https://hf-mirror.com \
ROBUSTVLM_OFFICIAL_EVAL_MODEL=tecoa4 \
ROBUSTVLM_OFFICIAL_EVAL_N_SAMPLES=-1 \
bash scripts_autodl/10_official_clean_eval_dryrun.sh
```

Results:

| Model | Path | Samples | Attack setting | Clean top-1 | README reference | Gap pp | Status |
|---|---|---:|---|---:|---:|---:|---|
| FARE4 ViT-B/16 LAION2B | `CLIP_benchmark` zero-shot | 50000 | `none` | 60.4700 | 56.6 | +3.8700 | mismatch |
| TeCoA4 ViT-B/16 LAION2B | `CLIP_benchmark` zero-shot | 50000 | `none` | 64.4820 | 51.5 | +12.9820 | mismatch |

Output files:

- `reproduce/e0a_official_clean_eval_fare4_-1.json`
- `reproduce/e0a_official_clean_eval_tecoa4_-1.json`

Interpretation:

- The official clean-only `CLIP_benchmark` path reproduces the earlier custom full-clean FARE4 result within `0.002 pp`.
- The official clean-only `CLIP_benchmark` path reproduces the earlier custom full-clean TeCoA4 result within `0.024 pp`.
- This makes a local loading, tokenizer, prompt-template, or preprocessing bug less likely.
- The best current public-safe classification is `e0a_reference_version_issue_likely`.
- Remaining uncertainty: the README Additional Models clean values may come from a 5000-sample RobustBench-oriented protocol, a different checkpoint revision, or another evaluation path not captured by the unchanged clean-only `CLIP_benchmark` command.

## Loaded Model Identity Diagnostics

Diagnostic files:

- `reproduce/openclip_model_identity.md`
- `reproduce/openclip_model_identity.json`

Summary:

| Model | OpenCLIP model name | HF revision | Visual class | ViT-B/16 signature | ConvNeXt signature | Visual params | Total params | Logit scale exp |
|---|---|---|---|---:|---:|---:|---:|---:|
| vanilla | `ViT-B-16`, `pretrained=laion2b_s34b_b88k` | n/a | `VisionTransformer` | true | false | 86192640 | 149620737 | 100.000008 |
| FARE4 | `hf-hub:chs20/FARE4-ViT-B-16-laion2B-s34B-b88K` | `1e941d54ce261e19ee072f1c3d5af8670aa19c02` | `VisionTransformer` | true | false | 86192640 | 149620737 | 100.000008 |
| TeCoA4 | `hf-hub:chs20/TeCoA4-ViT-B-16-laion2B-s34B-b88K` | `ae1409bcaf7d062d37a80457747c14b6bddcbc63` | `VisionTransformer` | true | false | 86192640 | 149620737 | 100.000008 |

Both HF configs report:

- image size: `224`;
- ViT layers: `12`;
- ViT width: `768`;
- patch size: `16`;
- text context length: `77`;
- text width: `512`;
- text layers: `12`;
- preprocess mean/std: CLIP mean/std.

Conclusion: FARE4 is not accidentally loading the ConvNeXt-B checkpoint or another non-ViT-B/16 architecture.

## TeCoA4 Continuation

Command:

```bash
PYTHON_BIN=/root/miniconda3/envs/robustvlm-openclip-py310/bin/python \
HF_ENDPOINT=https://hf-mirror.com \
bash scripts_autodl/07_e0a_clean_calibration.sh \
  --model tecoa4 \
  --batch-size 64 \
  --num-workers 4 \
  --imagenet-val-dir /root/autodl-tmp/data/imagenet/val \
  --full
```

Result:

- TeCoA4 ensemble full ImageNet clean: `64.4580%`.
- README reference: `51.5%`.
- Gap: `+12.9580 pp`.
- Status: mismatch.

Interpretation: the mismatch is not FARE4-only.

## Alternative Loading / Official Eval Comparison

FARE4 visual-only transplant:

```bash
HF_ENDPOINT=https://hf-mirror.com \
/root/miniconda3/envs/robustvlm-openclip-py310/bin/python \
tools/openclip_visual_transplant_clean.py \
  --visual-source fare4 \
  --batch-size 64 \
  --num-workers 4 \
  --imagenet-val /root/autodl-tmp/data/imagenet/val \
  --full \
  --write-json reproduce/e0a_visual_transplant_fare4.json
```

Result:

- Variant: vanilla LAION2B ViT-B/16 text tower and logit scale plus FARE4 visual encoder.
- Clean accuracy: `60.4680%`.
- This exactly matches direct hf-hub FARE4 clean accuracy.

Interpretation: the FARE4 mismatch is unlikely to be caused by the hf-hub model's text tower or logit scale. The result follows the FARE4 visual encoder under the current evaluation protocol.

Official `CLIP_benchmark` clean-only check:

```bash
PYTHONPATH=/root/repro-RobustVLM/CLIP_benchmark:/root/repro-RobustVLM \
/root/miniconda3/envs/robustvlm-openclip-py310/bin/python \
  -m clip_benchmark.cli eval \
  --dataset imagenet1k-unverified \
  --dataset_root /root/autodl-tmp/data/imagenet \
  --split val \
  --task zeroshot_classification \
  --model hf-hub:chs20/FARE4-ViT-B-16-laion2B-s34B-b88K \
  --pretrained "" \
  --attack none \
  --batch_size 64 \
  --num_workers 4 \
  --n_samples -1 \
  --output /root/repro-RobustVLM/reproduce/e0a_official_clean_eval_fare4_-1.json
```

Result:

- Completed on all `50000` ImageNet val images.
- FARE4 top-1: `60.4700%`.
- TeCoA4 top-1: `64.4820%`.
- Both remain outside the `±1.0 pp` gate against the README Additional Models references.

RobustBench clean-only helper:

- Intended to reuse `robustbench.data.load_clean_dataset` with `n_examples=5000` and no attack.
- Not run as an official unchanged command.
- Reason: `CLIP_eval/clip_robustbench.py` is attack-oriented as written, and this task avoided patching core eval code or running AutoAttack.

## Prompt Template Diagnostic

FARE4 prompt comparison:

| Model | Prompt mode | Samples | Clean acc | Reference | Gap pp | Runtime |
|---|---|---:|---:|---:|---:|---:|
| FARE4 | ensemble | 50000 | 60.4680 | 56.6 | +3.8680 | 92.09s |
| FARE4 | std | 50000 | 58.6500 | 56.6 | +2.0500 | 95.93s |

TeCoA4 prompt comparison:

| Model | Prompt mode | Samples | Clean acc | Reference | Gap pp | Runtime |
|---|---|---:|---:|---:|---:|---:|
| TeCoA4 | ensemble | 50000 | 64.4580 | 51.5 | +12.9580 | 91.37s |
| TeCoA4 | std | 50000 | 66.0860 | 51.5 | +14.5860 | 90.66s |

Interpretation:

- Prompt choice partially reduces the FARE4 gap but does not make it pass.
- Prompt choice does not explain the TeCoA4 gap.
- Template mismatch is therefore not a sufficient explanation.

## Answers

1. Is FARE4 definitely loading the intended ViT-B/16 LAION2B checkpoint?
   - It is definitely loading a ViT-B/16 HF model from the intended repo id. It is not ConvNeXt-B. Whether the current HF `main` revision is the exact revision used for the README table remains unverified.

2. What HF revision / checkpoint file was used?
   - FARE4 revision: `1e941d54ce261e19ee072f1c3d5af8670aa19c02`; file: `open_clip_pytorch_model.bin`.
   - TeCoA4 revision: `ae1409bcaf7d062d37a80457747c14b6bddcbc63`; file: `open_clip_pytorch_model.bin`.

3. Does TeCoA4 match its `51.5` clean reference?
   - No. The current full ImageNet ensemble result is `64.4580%`, gap `+12.9580 pp`.

4. Does the official RobustVLM clean eval path reproduce `56.6` for FARE4?
   - No under the unchanged clean-only `CLIP_benchmark` path. It gives `60.4700%` for FARE4 on full ImageNet val with `--attack none`.

5. If official eval was not run, why not?
   - `CLIP_benchmark` clean-only eval was run.
   - `CLIP_eval/clip_robustbench.py` was not run as a formal clean-only result because its normal path proceeds into AutoAttack and its benchmark path calls RobustBench benchmark.

6. Likely cause?
   - wrong checkpoint / revision: possible but not proven; current HF revisions may not correspond to the README table protocol or revision.
   - open_clip hf-hub direct loading behavior: unlikely to be the main cause for FARE4 because visual-only transplant exactly matches direct hf-hub accuracy.
   - visual-only vs full-model loading: unlikely to explain FARE4.
   - prompt template difference: partial for FARE4, not sufficient; not explanatory for TeCoA4.
   - preprocessing difference: no evidence found; all inspected paths use CLIP mean/std and bicubic resize/crop.
   - reference value mismatch / README-version mismatch: likely under the currently available unchanged clean-only path.
   - sample-count / official RobustBench subset mismatch: possible and still unresolved because the official `CLIP_eval.clip_robustbench` entrypoint is not clean-only as written.

7. Is E0a still blocked?
   - Yes. Status is `e0a_reference_version_issue_likely`, not passed.

8. Next recommended action?
   - Before E0b, verify the exact README reference protocol/revision for the LAION2B B/16 HF models, or get an author/source-confirmed clean-only command for the Additional Models table.
