# ADI-LoRA / ADI-DoRA Q2-Edge Stage Commands

This note records the protocol-valid commands for the next optimization stage.
It keeps Nearest and corruption results outside alpha selection.

## Protocol Boundaries

- Alpha selection source: validation Bicubic and validation Bilinear only.
- Held-out Nearest full-alpha sweeps are diagnostic only.
- Corruption tests are never used for alpha selection.
- Checkpoint selection rule: `final_checkpoint_only`.
- ADI adds no trainable parameters and no inference module.
- Tied validation scores choose the largest alpha under the minimal-intervention principle.

## A. DoRA seed43/44 Full-Alpha Nearest Diagnostic

Run on AutoDL from the project root:

```bash
cd /path/to/FR-PEFT-dialogue3b
export PYTHONPATH=$PWD/src:$PWD

python tools/eval_dora_same_checkpoint_alpha_sweep.py \
  --run-dir outputs/fr_peft_cifar100/dora_seed43_20260627_142535 \
  --checkpoint outputs/fr_peft_cifar100/dora_seed43_20260627_142535/checkpoints/final_dora_seed43.pth \
  --out-dir outputs/adi_q2_edge/dora_seed43_full_alpha_nearest_diagnostic \
  --seed 43 \
  --alphas 0.2,0.4,0.6,0.8,1.0 \
  --eval-all-test-alphas

python tools/eval_dora_same_checkpoint_alpha_sweep.py \
  --run-dir outputs/fr_peft_cifar100/dora_seed44_20260627_161856 \
  --checkpoint outputs/fr_peft_cifar100/dora_seed44_20260627_161856/checkpoints/final_dora_seed44.pth \
  --out-dir outputs/adi_q2_edge/dora_seed44_full_alpha_nearest_diagnostic \
  --seed 44 \
  --alphas 0.2,0.4,0.6,0.8,1.0 \
  --eval-all-test-alphas
```

Expected outputs per seed:

```text
alpha_selection_same_checkpoint.csv
test_eval_same_checkpoint.csv
dora_full_alpha_response.csv
same_checkpoint_comparison.csv
protocol_audit.json
README_same_checkpoint_dora_alpha_eval.md
```

Use `dora_full_alpha_response.csv` for Fig. 4 only as a diagnostic curve.

## B. Fig. 4 Alpha-Response Curve

After LoRA and DoRA full-alpha response CSVs are available:

```bash
cd /path/to/FR-PEFT-dialogue3b
export PYTHONPATH=$PWD/src:$PWD

python tools/plot_fig4_alpha_response.py \
  --inputs \
    "outputs/adi_q2_edge/lora_seed*/lora_full_alpha_nearest_sweep.csv" \
    "outputs/adi_q2_edge/dora_seed*_full_alpha_nearest_diagnostic/dora_full_alpha_response.csv" \
  --out figures/generated/fig4_alpha_response.pdf
```

Caption boundary:

```text
Alpha is selected from Bicubic/Bilinear validation only. The Nearest curve is a
held-out diagnostic evaluation on the same final checkpoint. No retraining or
extra inference module is introduced.
```

## C. Fig. 5 Mechanism Diagnostic

First generate diagnostic CSV rows:

```bash
cd /path/to/FR-PEFT-dialogue3b
export PYTHONPATH=$PWD/src:$PWD

python tools/eval_mechanism_diagnostics.py \
  --run-dir outputs/fr_peft_cifar100/delta_lora_seed42_20260624_164413 \
  --out-csv outputs/adi_q2_edge/mechanism/lora_seed42_mechanism.csv \
  --method-label LoRA

python tools/eval_mechanism_diagnostics.py \
  --run-dir outputs/fr_peft_cifar100/delta_lora_seed43_20260624_182356 \
  --out-csv outputs/adi_q2_edge/mechanism/lora_seed43_mechanism.csv \
  --method-label LoRA

python tools/eval_mechanism_diagnostics.py \
  --run-dir outputs/fr_peft_cifar100/delta_lora_seed44_20260624_194722 \
  --out-csv outputs/adi_q2_edge/mechanism/lora_seed44_mechanism.csv \
  --method-label LoRA

python tools/eval_mechanism_diagnostics.py \
  --run-dir "$(find outputs/fr_peft_cifar100 -maxdepth 1 -type d -name 'dora_seed42_*' | sort | tail -n 1)" \
  --out-csv outputs/adi_q2_edge/mechanism/dora_seed42_mechanism.csv \
  --method-label DoRA

python tools/eval_mechanism_diagnostics.py \
  --run-dir outputs/fr_peft_cifar100/dora_seed43_20260627_142535 \
  --out-csv outputs/adi_q2_edge/mechanism/dora_seed43_mechanism.csv \
  --method-label DoRA

python tools/eval_mechanism_diagnostics.py \
  --run-dir outputs/fr_peft_cifar100/dora_seed44_20260627_161856 \
  --out-csv outputs/adi_q2_edge/mechanism/dora_seed44_mechanism.csv \
  --method-label DoRA
```

If the automatic seed42 path is empty, locate it with:

```bash
find outputs/fr_peft_cifar100 -maxdepth 1 -type d -name "dora_seed42_*" | sort
```

Then plot:

```bash
python tools/plot_fig5_mechanism_diagnostics.py \
  --inputs "outputs/adi_q2_edge/mechanism/*_mechanism.csv" \
  --out figures/generated/fig5_mechanism_diagnostics.pdf
```

Paper wording:

```text
These diagnostics suggest, rather than prove, that ADI improves robustness partly
by reducing excessive representation and spectral drift from the pretrained
visual anchor.
```

## D. Unified ADI Protocol Runner

For a new formal run:

```bash
cd /path/to/FR-PEFT-dialogue3b
export PYTHONPATH=$PWD/src:$PWD

python tools/run_adi_protocol.py \
  --config configs/fr_peft_cifar100_delta_lora.yaml \
  --seed 42
```

Expected outputs inside the new run directory:

```text
train_log.csv
alpha_selection.csv
test_eval_alpha1.csv
test_eval_selected.csv
same_checkpoint_comparison.csv
summary.csv
protocol_audit.json
split_manifest.json
```

## E. Merged Checkpoint Export

Export a deployable dense checkpoint after alpha is selected:

```bash
python tools/export_merged_adi_checkpoint.py \
  --config outputs/fr_peft_cifar100/delta_lora_seed42_20260624_164413/resolved_config.json \
  --checkpoint outputs/fr_peft_cifar100/delta_lora_seed42_20260624_164413/checkpoints/final_delta_lora_seed42.pth \
  --alpha 0.8 \
  --out outputs/adi_q2_edge/merged/merged_adi_lora_seed42_alpha08.pth \
  --audit-out outputs/adi_q2_edge/merged/merge_audit_lora_seed42_alpha08.json
```

The audit JSON contains:

```text
alpha
merged=true
extra_inference_modules=0
max_output_error_before_after_merge
```

Use this only to support mergeable deployment, not as a new training result.
