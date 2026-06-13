# OpenCLIP Model Identity Diagnostics

This public-safe diagnostic records model identity and cache metadata for E0a mismatch analysis. It does not contain private route analysis.

| Model | OpenCLIP model name | HF repo | Visual class | ViT-B/16 signature | ConvNeXt signature | Visual params | Total params | Image size | Logit scale exp |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| vanilla | `ViT-B-16` | `` | `VisionTransformer` | True | False | 86192640 | 149620737 | `[224, 224]` | 100.000008 |
| fare4 | `hf-hub:chs20/FARE4-ViT-B-16-laion2B-s34B-b88K` | `chs20/FARE4-ViT-B-16-laion2B-s34B-b88K` | `VisionTransformer` | True | False | 86192640 | 149620737 | `[224, 224]` | 100.000008 |
| tecoa4 | `hf-hub:chs20/TeCoA4-ViT-B-16-laion2B-s34B-b88K` | `chs20/TeCoA4-ViT-B-16-laion2B-s34B-b88K` | `VisionTransformer` | True | False | 86192640 | 149620737 | `[224, 224]` | 100.000008 |

## Cache Revisions

### vanilla
- HF cache status: `None`.
- HF cache dir: `None`.

### fare4
- HF cache status: `found`.
- HF cache dir: `/root/autodl-tmp/model_cache/huggingface/hub`.
- revision: `1e941d54ce261e19ee072f1c3d5af8670aa19c02`, refs: `['main']`
- files: `['open_clip_config.json', 'open_clip_pytorch_model.bin']`

### tecoa4
- HF cache status: `found`.
- HF cache dir: `/root/autodl-tmp/model_cache/huggingface/hub`.
- revision: `ae1409bcaf7d062d37a80457747c14b6bddcbc63`, refs: `['main']`
- files: `['open_clip_config.json', 'open_clip_pytorch_model.bin']`

## Raw JSON

```json
[
  {
    "hf_cache": null,
    "hf_repo_id": null,
    "metadata": {
      "hf_hub": false,
      "image_size": [
        224,
        224
      ],
      "key": "vanilla",
      "label": "vanilla LAION2B ViT-B/16",
      "lift_mean": [
        0.48145466,
        0.4578275,
        0.40821073
      ],
      "lift_std": [
        0.26862954,
        0.26130258,
        0.27577711
      ],
      "logit_scale_exp": 100.00000762939453,
      "maps_to_non_quickgelu_vit_b16": true,
      "matches_lift_normalization": true,
      "mean": [
        0.48145466,
        0.4578275,
        0.40821073
      ],
      "model_name": "ViT-B-16",
      "preprocess_train": [
        {
          "antialias": "warn",
          "interpolation": "InterpolationMode.BICUBIC",
          "size": [
            224,
            224
          ],
          "type": "RandomResizedCrop"
        },
        {
          "type": "function"
        },
        {
          "type": "ToTensor"
        },
        {
          "mean": [
            0.48145466,
            0.4578275,
            0.40821073
          ],
          "std": [
            0.26862954,
            0.26130258,
            0.27577711
          ],
          "type": "Normalize"
        }
      ],
      "preprocess_val": [
        {
          "antialias": "warn",
          "interpolation": "InterpolationMode.BICUBIC",
          "max_size": null,
          "size": 224,
          "type": "Resize"
        },
        {
          "size": [
            224,
            224
          ],
          "type": "CenterCrop"
        },
        {
          "type": "function"
        },
        {
          "type": "ToTensor"
        },
        {
          "mean": [
            0.48145466,
            0.4578275,
            0.40821073
          ],
          "std": [
            0.26862954,
            0.26130258,
            0.27577711
          ],
          "type": "Normalize"
        }
      ],
      "pretrained": "laion2b_s34b_b88k",
      "quickgelu_modules": [],
      "reference_clean": 70.23,
      "reference_source": "OpenCLIP official classification CSV, ImageNet 1k top-1",
      "std": [
        0.26862954,
        0.26130258,
        0.27577711
      ],
      "tokenizer_type": "function"
    },
    "model_key": "vanilla",
    "model_label": "vanilla LAION2B ViT-B/16",
    "open_clip_model_name": "ViT-B-16",
    "parameter_counts": {
      "total": 149620737,
      "visual": 86192640
    },
    "pretrained": "laion2b_s34b_b88k",
    "text_signature": {
      "text_projection_shape": [
        512,
        512
      ],
      "text_resblock_count": 12,
      "text_tower_class": "Transformer",
      "token_embedding_shape": [
        49408,
        512
      ]
    },
    "visual_signature": {
      "conv1_weight_shape": [
        768,
        3,
        16,
        16
      ],
      "image_size": [
        224,
        224
      ],
      "ln_post_weight_shape": [
        768
      ],
      "looks_like_convnext": false,
      "looks_like_vit_b16": true,
      "projection_shape": [
        768,
        512
      ],
      "resblock_count": 12,
      "state_key_examples": [
        "class_embedding",
        "positional_embedding",
        "proj",
        "conv1.weight",
        "ln_pre.weight",
        "ln_pre.bias",
        "transformer.resblocks.0.ln_1.weight",
        "transformer.resblocks.0.ln_1.bias",
        "transformer.resblocks.0.attn.in_proj_weight",
        "transformer.resblocks.0.attn.in_proj_bias",
        "transformer.resblocks.0.attn.out_proj.weight",
        "transformer.resblocks.0.attn.out_proj.bias"
      ],
      "visual_class": "VisionTransformer"
    }
  },
  {
    "hf_cache": {
      "cache_dir": "/root/autodl-tmp/model_cache/huggingface/hub",
      "repos": [
        {
          "repo_id": "chs20/FARE4-ViT-B-16-laion2B-s34B-b88K",
          "repo_path": "/root/autodl-tmp/model_cache/huggingface/hub/models--chs20--FARE4-ViT-B-16-laion2B-s34B-b88K",
          "repo_type": "model",
          "revisions": [
            {
              "commit_hash": "1e941d54ce261e19ee072f1c3d5af8670aa19c02",
              "files": [
                {
                  "file_name": "open_clip_config.json",
                  "size_on_disk": 532
                },
                {
                  "file_name": "open_clip_pytorch_model.bin",
                  "size_on_disk": 598599478
                }
              ],
              "refs": [
                "main"
              ],
              "snapshot_path": "/root/autodl-tmp/model_cache/huggingface/hub/models--chs20--FARE4-ViT-B-16-laion2B-s34B-b88K/snapshots/1e941d54ce261e19ee072f1c3d5af8670aa19c02"
            }
          ]
        }
      ],
      "status": "found"
    },
    "hf_repo_id": "chs20/FARE4-ViT-B-16-laion2B-s34B-b88K",
    "metadata": {
      "hf_hub": true,
      "image_size": [
        224,
        224
      ],
      "key": "fare4",
      "label": "FARE4 ViT-B/16 LAION2B",
      "lift_mean": [
        0.48145466,
        0.4578275,
        0.40821073
      ],
      "lift_std": [
        0.26862954,
        0.26130258,
        0.27577711
      ],
      "logit_scale_exp": 100.00000762939453,
      "maps_to_non_quickgelu_vit_b16": true,
      "matches_lift_normalization": true,
      "mean": [
        0.48145466,
        0.4578275,
        0.40821073
      ],
      "model_name": "hf-hub:chs20/FARE4-ViT-B-16-laion2B-s34B-b88K",
      "preprocess_train": [
        {
          "antialias": "warn",
          "interpolation": "InterpolationMode.BICUBIC",
          "size": [
            224,
            224
          ],
          "type": "RandomResizedCrop"
        },
        {
          "type": "function"
        },
        {
          "type": "ToTensor"
        },
        {
          "mean": [
            0.48145466,
            0.4578275,
            0.40821073
          ],
          "std": [
            0.26862954,
            0.26130258,
            0.27577711
          ],
          "type": "Normalize"
        }
      ],
      "preprocess_val": [
        {
          "antialias": "warn",
          "interpolation": "InterpolationMode.BICUBIC",
          "max_size": null,
          "size": 224,
          "type": "Resize"
        },
        {
          "size": [
            224,
            224
          ],
          "type": "CenterCrop"
        },
        {
          "type": "function"
        },
        {
          "type": "ToTensor"
        },
        {
          "mean": [
            0.48145466,
            0.4578275,
            0.40821073
          ],
          "std": [
            0.26862954,
            0.26130258,
            0.27577711
          ],
          "type": "Normalize"
        }
      ],
      "pretrained": null,
      "quickgelu_modules": [],
      "reference_clean": 56.6,
      "reference_source": "RobustVLM README Additional Models table",
      "std": [
        0.26862954,
        0.26130258,
        0.27577711
      ],
      "tokenizer_type": "function"
    },
    "model_key": "fare4",
    "model_label": "FARE4 ViT-B/16 LAION2B",
    "open_clip_model_name": "hf-hub:chs20/FARE4-ViT-B-16-laion2B-s34B-b88K",
    "parameter_counts": {
      "total": 149620737,
      "visual": 86192640
    },
    "pretrained": null,
    "text_signature": {
      "text_projection_shape": [
        512,
        512
      ],
      "text_resblock_count": 12,
      "text_tower_class": "Transformer",
      "token_embedding_shape": [
        49408,
        512
      ]
    },
    "visual_signature": {
      "conv1_weight_shape": [
        768,
        3,
        16,
        16
      ],
      "image_size": [
        224,
        224
      ],
      "ln_post_weight_shape": [
        768
      ],
      "looks_like_convnext": false,
      "looks_like_vit_b16": true,
      "projection_shape": [
        768,
        512
      ],
      "resblock_count": 12,
      "state_key_examples": [
        "class_embedding",
        "positional_embedding",
        "proj",
        "conv1.weight",
        "ln_pre.weight",
        "ln_pre.bias",
        "transformer.resblocks.0.ln_1.weight",
        "transformer.resblocks.0.ln_1.bias",
        "transformer.resblocks.0.attn.in_proj_weight",
        "transformer.resblocks.0.attn.in_proj_bias",
        "transformer.resblocks.0.attn.out_proj.weight",
        "transformer.resblocks.0.attn.out_proj.bias"
      ],
      "visual_class": "VisionTransformer"
    }
  },
  {
    "hf_cache": {
      "cache_dir": "/root/autodl-tmp/model_cache/huggingface/hub",
      "repos": [
        {
          "repo_id": "chs20/TeCoA4-ViT-B-16-laion2B-s34B-b88K",
          "repo_path": "/root/autodl-tmp/model_cache/huggingface/hub/models--chs20--TeCoA4-ViT-B-16-laion2B-s34B-b88K",
          "repo_type": "model",
          "revisions": [
            {
              "commit_hash": "ae1409bcaf7d062d37a80457747c14b6bddcbc63",
              "files": [
                {
                  "file_name": "open_clip_config.json",
                  "size_on_disk": 532
                },
                {
                  "file_name": "open_clip_pytorch_model.bin",
                  "size_on_disk": 598599478
                }
              ],
              "refs": [
                "main"
              ],
              "snapshot_path": "/root/autodl-tmp/model_cache/huggingface/hub/models--chs20--TeCoA4-ViT-B-16-laion2B-s34B-b88K/snapshots/ae1409bcaf7d062d37a80457747c14b6bddcbc63"
            }
          ]
        }
      ],
      "status": "found"
    },
    "hf_repo_id": "chs20/TeCoA4-ViT-B-16-laion2B-s34B-b88K",
    "metadata": {
      "hf_hub": true,
      "image_size": [
        224,
        224
      ],
      "key": "tecoa4",
      "label": "TeCoA4 ViT-B/16 LAION2B",
      "lift_mean": [
        0.48145466,
        0.4578275,
        0.40821073
      ],
      "lift_std": [
        0.26862954,
        0.26130258,
        0.27577711
      ],
      "logit_scale_exp": 100.00000762939453,
      "maps_to_non_quickgelu_vit_b16": true,
      "matches_lift_normalization": true,
      "mean": [
        0.48145466,
        0.4578275,
        0.40821073
      ],
      "model_name": "hf-hub:chs20/TeCoA4-ViT-B-16-laion2B-s34B-b88K",
      "preprocess_train": [
        {
          "antialias": "warn",
          "interpolation": "InterpolationMode.BICUBIC",
          "size": [
            224,
            224
          ],
          "type": "RandomResizedCrop"
        },
        {
          "type": "function"
        },
        {
          "type": "ToTensor"
        },
        {
          "mean": [
            0.48145466,
            0.4578275,
            0.40821073
          ],
          "std": [
            0.26862954,
            0.26130258,
            0.27577711
          ],
          "type": "Normalize"
        }
      ],
      "preprocess_val": [
        {
          "antialias": "warn",
          "interpolation": "InterpolationMode.BICUBIC",
          "max_size": null,
          "size": 224,
          "type": "Resize"
        },
        {
          "size": [
            224,
            224
          ],
          "type": "CenterCrop"
        },
        {
          "type": "function"
        },
        {
          "type": "ToTensor"
        },
        {
          "mean": [
            0.48145466,
            0.4578275,
            0.40821073
          ],
          "std": [
            0.26862954,
            0.26130258,
            0.27577711
          ],
          "type": "Normalize"
        }
      ],
      "pretrained": null,
      "quickgelu_modules": [],
      "reference_clean": 51.5,
      "reference_source": "RobustVLM README Additional Models table",
      "std": [
        0.26862954,
        0.26130258,
        0.27577711
      ],
      "tokenizer_type": "function"
    },
    "model_key": "tecoa4",
    "model_label": "TeCoA4 ViT-B/16 LAION2B",
    "open_clip_model_name": "hf-hub:chs20/TeCoA4-ViT-B-16-laion2B-s34B-b88K",
    "parameter_counts": {
      "total": 149620737,
      "visual": 86192640
    },
    "pretrained": null,
    "text_signature": {
      "text_projection_shape": [
        512,
        512
      ],
      "text_resblock_count": 12,
      "text_tower_class": "Transformer",
      "token_embedding_shape": [
        49408,
        512
      ]
    },
    "visual_signature": {
      "conv1_weight_shape": [
        768,
        3,
        16,
        16
      ],
      "image_size": [
        224,
        224
      ],
      "ln_post_weight_shape": [
        768
      ],
      "looks_like_convnext": false,
      "looks_like_vit_b16": true,
      "projection_shape": [
        768,
        512
      ],
      "resblock_count": 12,
      "state_key_examples": [
        "class_embedding",
        "positional_embedding",
        "proj",
        "conv1.weight",
        "ln_pre.weight",
        "ln_pre.bias",
        "transformer.resblocks.0.ln_1.weight",
        "transformer.resblocks.0.ln_1.bias",
        "transformer.resblocks.0.attn.in_proj_weight",
        "transformer.resblocks.0.attn.in_proj_bias",
        "transformer.resblocks.0.attn.out_proj.weight",
        "transformer.resblocks.0.attn.out_proj.bias"
      ],
      "visual_class": "VisionTransformer"
    }
  }
]
```
