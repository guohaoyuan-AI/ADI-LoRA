# Merged Checkpoint Equivalence QA

Status: PASS

Checked package: `merged_checkpoint_equivalence_audit_only_20260703.tar.gz`

Summary:
- Expected runs: 6
- Found runs: 6
- All expected found: True
- All merge equivalence pass: True
- Max output error observed: 1.16229057312e-05
- Pass threshold: 1.0e-04
- LoRA max error: 1.16229057312e-05
- DoRA max error: 0

Protocol checks:
- `merged=true` for all six runs.
- `extra_inference_modules=0` for all six runs.
- `load_missing_keys=0` and `load_unexpected_keys=0` for all six runs.
- `alpha_selection_source=val_bicubic_and_val_bilinear` for all six runs.
- `nearest_used_for_alpha_selection=0` and `corruption_used_for_alpha_selection=0`.
- `checkpoint_selection_rule=final_checkpoint_only`.

Interpretation boundary:
This is deployment equivalence evidence only. It should support the claim that ADI can be merged into a dense checkpoint with no extra inference-time module. It should not be described as a new accuracy result or alpha-selection experiment.
