# ADI Table Narrative For Paper

## Recommended Main Evidence

Use CIFAR-100 clean interpolation as the central table: ADI-LoRA and ADI-DoRA are both evaluated across seeds 42/43/44 with same-checkpoint, final-checkpoint-only evaluation. Alpha is selected only from Bicubic/Bilinear validation, while Nearest is held out for testing.

Use CIFAR-100-C ADI-DoRA subset as the main corruption robustness evidence because it covers 3 seeds x 4 corruptions x 3 severities. The full19 severity=3 seed42 table should be a detailed supplement, not a three-seed claim.

Use Tiny-ImageNet LoRA three-seed results as Stage 4 dataset-extension evidence. Tiny-ImageNet DoRA seed42 should be described as supplementary single-seed evidence only.

ImageNet-100 LoRA/DoRA seed42 should be reported only as pilot evidence due to the generated class-stratified symlink split.

## Aggregates Recomputed From CSV

| Group | Mean gain pp | Positive / Total | Alpha distribution |
|---|---:|---:|---|
| CIFAR-100 ADI-LoRA clean three-seed | 4.0733 | 3/3 | 0.8: 3 |
| CIFAR-100 ADI-DoRA clean three-seed | 8.6967 | 3/3 | 0.6: 1; 0.8: 2 |
| CIFAR-100-C ADI-DoRA subset_3seed_4corruptions_s1s3s5 | 2.7322 | 34/36 | 0.6: 12; 0.8: 24 |
| CIFAR-100-C ADI-DoRA full19_severity3_seed42 | 2.2153 | 19/19 | 0.8: 19 |
| Tiny-ImageNet ADI-LoRA clean three-seed | 0.7733 | 3/3 | 0.8: 3 |
| Tiny-ImageNet ADI-DoRA single_seed_supplement | 1.52 | 1/1 | 0.8: 1 |
| ImageNet-100 ADI-LoRA pilot_single_seed | 0.6394 | 1/1 | 0.6: 1 |
| ImageNet-100 ADI-DoRA pilot_single_seed | 0.7183 | 1/1 | 0.6: 1 |

## Claims To Avoid

- Do not state that Nearest or corruption data were used for alpha selection.
- Do not write ImageNet-100 as a formal standard benchmark result.
- Do not write Tiny-ImageNet DoRA as three-seed evidence.
- Do not use LoRA CIFAR-100-C narrative numbers as formal table entries until the raw CSV/package is supplied.