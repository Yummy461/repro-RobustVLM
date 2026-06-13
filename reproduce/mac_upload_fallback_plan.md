# Mac Upload Fallback Plan

This fallback is currently not needed. AutoDL public storage was available, and ImageNet val was prepared successfully from the public tar/devkit.

Keep this fallback for future instances where AutoDL public storage becomes unavailable, extraction stalls for more than 10-15 minutes without progress, or the public val tar cannot be read reliably.

## Missing Or Blocked Resource

- Required final resource: extracted ImageNet validation ImageFolder.
- Expected source file: `ILSVRC2012_img_val.tar`.
- Expected devkit file: `ILSVRC2012_devkit_t12.tar.gz`.
- Current AutoDL public source exists, so Mac upload is not the first choice.

## Expected Server Destination

- Val tar destination if uploaded: `/root/autodl-tmp/data/ILSVRC2012_img_val.tar`
- Devkit destination if uploaded: `/root/autodl-tmp/data/ILSVRC2012_devkit_t12.tar.gz`
- Prepared ImageFolder: `/root/autodl-tmp/data/imagenet/val`

## Checksum Commands

Mac side:

```bash
shasum -a 256 /path/on/mac/ILSVRC2012_img_val.tar
shasum -a 256 /path/on/mac/ILSVRC2012_devkit_t12.tar.gz
```

Server side:

```bash
sha256sum /root/autodl-tmp/data/ILSVRC2012_img_val.tar
sha256sum /root/autodl-tmp/data/ILSVRC2012_devkit_t12.tar.gz
```

## Upload Commands

Use placeholders for host and port; do not commit machine credentials.

```bash
rsync -avP -e "ssh -p PORT" /path/on/mac/ILSVRC2012_img_val.tar root@SERVER:/root/autodl-tmp/data/
rsync -avP -e "ssh -p PORT" /path/on/mac/ILSVRC2012_devkit_t12.tar.gz root@SERVER:/root/autodl-tmp/data/
```

Alternative:

```bash
scp -P PORT /path/on/mac/ILSVRC2012_img_val.tar root@SERVER:/root/autodl-tmp/data/
scp -P PORT /path/on/mac/ILSVRC2012_devkit_t12.tar.gz root@SERVER:/root/autodl-tmp/data/
```

## Server-Side Resume

After upload:

```bash
IMAGENET_TAR_PATH=/root/autodl-tmp/data/ILSVRC2012_img_val.tar \
IMAGENET_DEVKIT_PATH=/root/autodl-tmp/data/ILSVRC2012_devkit_t12.tar.gz \
bash scripts_autodl/09_prepare_imagenet_val.sh --extract --prepare
```

Then run subset clean calibration:

```bash
bash scripts_autodl/07_e0a_clean_calibration.sh \
  --model all \
  --max-samples 512 \
  --batch-size 64 \
  --num-workers 4 \
  --imagenet-val-dir /root/autodl-tmp/data/imagenet/val
```

Full clean calibration still requires explicit `--full`.
