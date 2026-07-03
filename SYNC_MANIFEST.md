# ADI-LoRA GitHub sync delta 20260703

Copy these files into the root of the public `ADI-LoRA` repository, then commit and push. This delta combines the original Q2 edge protocol patch with the later Fig. 4/Fig. 5 and merged-checkpoint audit updates. No checkpoints, merged dense weights, raw datasets, or AutoDL logs are included.

## Files

- `README.md`
- `CITATION.cff`
- `src/adi_lora/engine/run_fr_peft.py`
- `data/figure_data/fig4_alpha_response_cifar100.csv`
- `data/figure_data/fig5_mechanism_diagnostics_cifar100.csv`
- `data/figure_data/fig5_mechanism_diagnostics_summary.csv`
- `data/processed/Merge_Equivalence.csv`
- `data/processed/Merge_Equivalence_QA.json`
- `docs/ADI_Q2_EDGE_STAGE_COMMANDS.md`
- `docs/MERGE_EQUIVALENCE_QA.md`
- `figures/fig4_alpha_response_cifar100.pdf`
- `figures/fig5_mechanism_diagnostics_cifar100.pdf`
- `figures/generated/fig4_alpha_response_cifar100.pdf`
- `figures/generated/fig4_alpha_response_cifar100.png`
- `figures/generated/fig4_alpha_response_cifar100.svg`
- `figures/generated/fig5_mechanism_diagnostics_cifar100.pdf`
- `figures/generated/fig5_mechanism_diagnostics_cifar100.png`
- `figures/generated/fig5_mechanism_diagnostics_cifar100.svg`
- `tools/eval_dora_same_checkpoint_alpha_sweep.py`
- `tools/eval_lora_full_alpha_nearest_sweep.py`
- `tools/eval_mechanism_diagnostics.py`
- `tools/export_merged_adi_checkpoint.py`
- `tools/plot_fig4_alpha_response.py`
- `tools/plot_fig5_mechanism_diagnostics.py`
- `tools/run_adi_protocol.py`
- `tools/summarize_merge_audits.py`

## Recommended commit message

`Add ADI protocol audits, diagnostic figures, and merged-checkpoint export`
