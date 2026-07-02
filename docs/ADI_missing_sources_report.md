# ADI Missing Sources Report

This report is generated from the uploaded 17-file handoff package. Narrative anchors were used only for cross-checking, not as substitutes for raw CSV evidence.

| Item | Status | Impact | Suggested file |
|---|---|---|---|
| LoRA CIFAR-100-C full19 severity=3 seed42 raw CSV/package | missing raw source | Do not use the +1.46 pp / 19 of 19 narrative anchor as a formal table until raw CSV is supplied. | lora_cifar100c_full19_s3_seed42_no_ckpt*.tar.gz or lora_cifar100c_comparison.csv |
| LoRA CIFAR-100-C subset / 3-seed raw CSV/package | missing raw source | Cannot claim LoRA corruption subset as formal 3-seed evidence from this upload. | lora_cifar100c_subset*.tar.gz / lora_cifar100c_3seed*.tar.gz |
| ImageNet-100 LoRA/DoRA seed42 | source present but pilot only | May be reported only as pilot; generated class-stratified symlink split is not a standard official ImageNet-100 benchmark. | standard official split manifest and class list if upgrading beyond pilot. |
| Tiny-ImageNet DoRA | single seed only | Do not write as Tiny-ImageNet DoRA 3-seed conclusion. | tiny_imagenet_dora_seed43/44 same-checkpoint packages if needed. |

## Source Boundary Notes

- `ADI_final_table_sources_20260629_213223` contains MISSING markers for CIFAR-100-C packages; the uploaded DoRA CIFAR-100-C subset/full19 packages supplement those missing entries.
- No raw LoRA CIFAR-100-C corruption CSV/package was found in the uploaded files, although narrative documents mention subset/full19 LoRA gains.
- ImageNet-100 rows are retained as pilot only because the available evidence uses a generated class-stratified symlink split.
- VPT/Adapter results are boundary diagnostics and are not formal ADI-LoRA evidence.