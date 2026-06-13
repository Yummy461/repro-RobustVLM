# OpenCLIP Attack Sanity Report

This public-safe report records a small CIFAR-100 attack-pipeline sanity check. It is not a formal robustness result.

- Model key: `vanilla`
- CIFAR-100 status: `{"path": "/root/autodl-tmp/data/cifar-100-python", "status": "exists"}`
- Dataset: CIFAR-100 test subset
- Text side: fixed precomputed class features
- Epsilon domain: pixel space after resize/crop and before normalization
- Attack: untargeted image-only L_inf PGD

| Epsilon | Steps | Step size | Clean acc | Robust acc | Grad ok | Runtime |
|---:|---:|---:|---:|---:|---|---:|
| 0.00000000 | 5 | 0.00000000 | 77.5391 | 77.5391 | True | 9.50s |
| 0.00392157 | 5 | 0.00196078 | 77.5391 | 0.0000 | True | 9.29s |
| 0.00784314 | 5 | 0.00392157 | 77.5391 | 0.0000 | True | 9.30s |
| 0.01568627 | 5 | 0.00784314 | 77.5391 | 0.0000 | True | 9.39s |

- Monotonic non-increasing robust accuracy: `True`
- If epsilon 0 does not exactly match clean accuracy, treat the attack pipeline as invalid.
- If eps 4/255 does not substantially reduce vanilla accuracy, inspect preprocessing, gradients, class order, and text features before running larger attacks.
- Observed eps 1/255 robust accuracy is `0.0`; keep this as an attack-interface stress signal only. Verify loss, labels, preprocessing, and official ImageNet protocol before treating it as a formal robustness number.
