# Revised E0b ImageNet Attack Subset Plan

Status: command plan only; not executed in this run.

Final revised E0b status is `e0b_revised_passed` from CIFAR-100 internal
calibration. Gate 2 remains pending because E1.5 is not part of this run.

Do not run full ImageNet attack in the revised E0b public fork workflow.

Optional future subset estimate:

```bash
cd /root/repro-RobustVLM
HF_ENDPOINT=https://hf-mirror.com \
python tools/openclip_e0b_attack_calibration.py \
  --dataset cifar100 \
  --model fare4 \
  --attack pgd20 \
  --eps-list 1 \
  --max-samples 1000 \
  --batch-size 32
```

The command above intentionally remains CIFAR-only because the current runner is scoped to CIFAR-100. A real ImageNet subset runner should be added separately and must keep outputs outside git.
