# FARE4 / TeCoA4 ViT-B/16 Loading Notes

Run timestamp: `2026-06-12 18:30:21`

This public-safe note records model loading, preprocessing, tokenizer, and lineage details for E0a calibration. It does not contain private route analysis.

## Environment

- Python: `3.10.8`
- Torch: `2.1.2+cu118`
- CUDA available: `True`
- Torchvision: `0.16.2+cu118`
- OpenCLIP: `unknown`

## Model Loading Summary

| Key | Model | Pretrained | HF Hub | Non-QuickGELU B/16 | Image size | Mean | Std | Tokenizer | logit_scale.exp |
|---|---|---|---|---|---|---|---|---|---:|
| vanilla | `ViT-B-16` | `laion2b_s34b_b88k` | False | True | `[224, 224]` | `[0.48145466, 0.4578275, 0.40821073]` | `[0.26862954, 0.26130258, 0.27577711]` | `function` | 100.000008 |
| fare4 | `hf-hub:chs20/FARE4-ViT-B-16-laion2B-s34B-b88K` | `None` | True | True | `[224, 224]` | `[0.48145466, 0.4578275, 0.40821073]` | `[0.26862954, 0.26130258, 0.27577711]` | `function` | 100.000008 |
| tecoa4 | `hf-hub:chs20/TeCoA4-ViT-B-16-laion2B-s34B-b88K` | `None` | True | True | `[224, 224]` | `[0.48145466, 0.4578275, 0.40821073]` | `[0.26862954, 0.26130258, 0.27577711]` | `function` | 100.000008 |

## Preprocessing Details

### vanilla

- Label: vanilla LAION2B ViT-B/16
- Reference clean accuracy: `70.23` from OpenCLIP official classification CSV, ImageNet 1k top-1
- Matches LIFT CLIP normalization: `True`
- LIFT mean/std: `[0.48145466, 0.4578275, 0.40821073]` / `[0.26862954, 0.26130258, 0.27577711]`
- OpenCLIP val preprocess: `[{"type": "Resize", "size": 224, "interpolation": "InterpolationMode.BICUBIC", "max_size": null, "antialias": "warn"}, {"type": "CenterCrop", "size": [224, 224]}, {"type": "function"}, {"type": "ToTensor"}, {"type": "Normalize", "mean": [0.48145466, 0.4578275, 0.40821073], "std": [0.26862954, 0.26130258, 0.27577711]}]`

### fare4

- Label: FARE4 ViT-B/16 LAION2B
- Reference clean accuracy: `56.6` from RobustVLM README Additional Models table
- Matches LIFT CLIP normalization: `True`
- LIFT mean/std: `[0.48145466, 0.4578275, 0.40821073]` / `[0.26862954, 0.26130258, 0.27577711]`
- OpenCLIP val preprocess: `[{"type": "Resize", "size": 224, "interpolation": "InterpolationMode.BICUBIC", "max_size": null, "antialias": "warn"}, {"type": "CenterCrop", "size": [224, 224]}, {"type": "function"}, {"type": "ToTensor"}, {"type": "Normalize", "mean": [0.48145466, 0.4578275, 0.40821073], "std": [0.26862954, 0.26130258, 0.27577711]}]`

### tecoa4

- Label: TeCoA4 ViT-B/16 LAION2B
- Reference clean accuracy: `51.5` from RobustVLM README Additional Models table
- Matches LIFT CLIP normalization: `True`
- LIFT mean/std: `[0.48145466, 0.4578275, 0.40821073]` / `[0.26862954, 0.26130258, 0.27577711]`
- OpenCLIP val preprocess: `[{"type": "Resize", "size": 224, "interpolation": "InterpolationMode.BICUBIC", "max_size": null, "antialias": "warn"}, {"type": "CenterCrop", "size": [224, 224]}, {"type": "function"}, {"type": "ToTensor"}, {"type": "Normalize", "mean": [0.48145466, 0.4578275, 0.40821073], "std": [0.26862954, 0.26130258, 0.27577711]}]`

## Public-Safe Risk Notes

- Epsilon comparability depends on applying perturbations in pixel space after resize/crop and before normalization.
- If any model reports a different mean/std or resize/crop behavior, clean and robust numbers must be interpreted with that preprocessing difference recorded.
- Full ImageNet calibration remains pending until a valid extracted ImageNet validation directory is available.
