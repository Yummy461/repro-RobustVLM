# ImageNet Val Data Check

Status: ImageNet val ImageFolder is prepared and verified.

## Remote Environment

- Host: AutoDL RTX 4090 instance.
- GPU: NVIDIA GeForce RTX 4090, driver `550.120`, memory `24564 MiB`.
- Data disk: `/root/autodl-tmp`, `50G` total.
- Available before val prep: about `49G`.
- Available after val prep: about `42G`.
- System disk is not used for datasets.

## Searched Paths

- `/root/autodl-pub/`
- `/root/autodl-pub/ImageNet/`
- `/root/autodl-pub/imagenet/`
- `/root/autodl-pub/ILSVRC2012/`
- `/root/autodl-tmp/data/`
- `/root/autodl-tmp/data/ImageNet/`
- `/root/autodl-tmp/data/imagenet/`
- `/root/autodl-tmp/data/ILSVRC2012/`
- `/root/autodl-tmp/data/imagenet-val/`
- `/root/autodl-tmp/data/imagenet/val/`

## Source Files

- Public source directory: `/root/autodl-pub/ImageNet/ILSVRC2012`.
- Validation tar: `/root/autodl-pub/ImageNet/ILSVRC2012/ILSVRC2012_img_val.tar`.
- Validation tar size: `6744924160` bytes, about `6.3 GiB`.
- Devkit: `/root/autodl-pub/ImageNet/ILSVRC2012/ILSVRC2012_devkit_t12.tar.gz`.
- Devkit size: `2568145` bytes.
- The val tar is flat, e.g. `ILSVRC2012_val_00046108.JPEG`.
- Devkit contains `data/meta.mat` and `data/ILSVRC2012_validation_ground_truth.txt`.

## Preparation Command

```bash
bash scripts_autodl/09_prepare_imagenet_val.sh --extract --prepare
```

Notes:

- Only ImageNet val was extracted.
- ImageNet train tar was not touched.
- No ImageNet download was run.
- Source tar files were not deleted.
- Flat staging path: `/root/autodl-tmp/data/imagenet/val_raw`.
- Target ImageFolder path: `/root/autodl-tmp/data/imagenet/val`.

The first run exposed a public-safe script issue: the remote image has no default `python` on `PATH`. The script now uses `PYTHON_BIN`, defaulting to `/root/miniconda3/envs/robustvlm-openclip-py310/bin/python`.

## Verification

Re-check command:

```bash
bash scripts_autodl/09_prepare_imagenet_val.sh --check
```

Observed:

```text
STATUS=ready_imagefolder
```

Independent structure checks:

- Class folders: `1000`.
- Image files under class folders: `50000`.
- Root-level JPEG files under `val/`: `0`.
- Example class folders:
  - `/root/autodl-tmp/data/imagenet/val/n01440764`
  - `/root/autodl-tmp/data/imagenet/val/n01443537`
  - `/root/autodl-tmp/data/imagenet/val/n01484850`
- `torchvision.datasets.ImageFolder` can load the directory.
- ImageFolder length: `50000`.
- ImageFolder classes: `1000`.
- First classes: `['n01440764', 'n01443537', 'n01484850', 'n01491361', 'n01494475']`.

## Space Estimate

- Source val tar remains in `/root/autodl-pub` and does not consume `/root/autodl-tmp`.
- Prepared val data under `/root/autodl-tmp/data/imagenet` uses about `6.4G`.
- Hard links were used where possible, so prepared ImageFolder did not duplicate the full image payload.
- Current 50GB data disk is sufficient for val-only E0a clean calibration.
- Full ImageNet train tar is about `147.9 GB`; current 50GB data disk is not enough for full ImageNet train extraction.

## Time Estimate

- Observed val extraction/preparation completed in a few minutes on this instance.
- Future reruns should be idempotent and check-ready if `/root/autodl-tmp/data/imagenet/val` remains present.
- E0a full clean calibration after val is ready, all three B/16 models: estimated `15-35 minutes` on RTX 4090 with batch size `64`.

## Current Data Path

Use this path for ImageNet clean calibration:

```bash
/root/autodl-tmp/data/imagenet/val
```
