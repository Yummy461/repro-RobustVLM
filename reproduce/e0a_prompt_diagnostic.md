# E0a Clean Calibration Report

This public-safe report records E0a clean calibration status for OpenCLIP LAION2B / FARE4 / TeCoA4. Full ImageNet calibration must use an extracted ImageNet validation directory and `--full`.

## Reference Targets

| Model | Reference clean top-1 | Source |
|---|---:|---|
| vanilla LAION2B ViT-B/16 | 70.23 | OpenCLIP official classification CSV |
| FARE4 ViT-B/16 LAION2B | 56.6 | RobustVLM README Additional Models table |
| TeCoA4 ViT-B/16 LAION2B | 51.5 | RobustVLM README Additional Models table |

## Measured Results

| Model | Mode | Samples | Clean acc | Reference | Gap pp | Runtime | Status |
|---|---|---:|---:|---:|---:|---:|---|
| fare4 | full | 50000 | 58.6500 | 56.6 | 2.0500 | 95.93s | mismatch |

## Stop Condition

- If any full clean calibration differs from its reference by more than 1.0 percentage point, stop before full attacks and investigate prompt templates, preprocessing, weight loading, tokenizer, and class ordering.

## Mismatch Report

At least one full clean calibration result differs from its reference by more than 1.0 percentage point.

Suspect order:

1. prompt templates;
2. preprocessing;
3. weight loading;
4. tokenizer;
5. class ordering.
