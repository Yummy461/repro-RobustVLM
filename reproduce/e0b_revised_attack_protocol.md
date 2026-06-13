# Revised E0b Attack Protocol

Status: `e0b_revised_passed`.

Scope boundary: this is internal attack calibration under the local CIFAR-100
protocol, not RobustVLM README reproduction. APGD-lite is not AutoAttack. Full
ImageNet attack was not run. Gate 2 remains pending because E1.5 is not part of
this run.

- Task: closed-set zero-shot CIFAR-100 image classification.
- Models: vanilla LAION2B ViT-B/16, FARE4 ViT-B/16 LAION2B, TeCoA4 ViT-B/16 LAION2B.
- Model identifiers:
  - vanilla: `ViT-B-16`, pretrained `laion2b_s34b_b88k`;
  - FARE4: `hf-hub:chs20/FARE4-ViT-B-16-laion2B-s34B-b88K`, observed HF revision `1e941d54ce261e19ee072f1c3d5af8670aa19c02`;
  - TeCoA4: `hf-hub:chs20/TeCoA4-ViT-B-16-laion2B-s34B-b88K`, observed HF revision `ae1409bcaf7d062d37a80457747c14b6bddcbc63`.
- Dataset: canonical CIFAR-100 Python test split, 10000 images.
- Class order: CIFAR-100 `fine_label_names` from `meta`.
- Text template: `a photo of a {c}.`
- Text/classifier side: fixed precomputed class features.
- Image preprocessing: resize 224, center crop 224, convert to `[0,1]` pixel tensor.
- Normalization: OpenCLIP CLIP mean/std inside the model call, mean `[0.48145466, 0.4578275, 0.40821073]`, std `[0.26862954, 0.26130258, 0.27577711]`.
- Epsilon convention: `pixel_after_resize_center_crop_before_normalization`.
- Norm: `L_inf`.
- PGD-20: untargeted CE loss, random start for eps > 0, step size `eps / 4`.
- APGD-lite: internal CE adaptive-step attack only; not AutoAttack.
- No TTE, no eval-time augmentation, no training, no thesis module implementation.
- Full ImageNet attack is out of scope unless separately approved.
