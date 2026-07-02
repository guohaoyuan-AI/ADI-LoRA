# ADI-LoRA / ADI-DoRA

This repository contains the core code, processed experimental tables, figure
data, and analysis scripts for the study:

**Adapter Delta Interpolation for Robust Visual Adaptation in Vision Transformers**

The repository is intended to support paper review and reproducibility. It
includes the cleaned experiment code subset used in this study, but does not
include model checkpoints, pretrained weights, or raw datasets.

## Scope

The released evidence supports ADI for LoRA/DoRA-style weight-delta PEFT:

```text
LoRA: W_ADI(alpha) = W0 + alpha * DeltaW
DoRA: W_ADI(alpha) = W0 + alpha * (W_DoRA - W0)
```

The main supported claims are limited to:

- CIFAR-100 clean interpolation-shift results for ADI-LoRA and ADI-DoRA.
- CIFAR-100-C ADI-DoRA corruption results.
- Processed tables, figure data, and scripts used to generate the manuscript
  tables and figures.
- Boundary diagnostics showing why non-weight-delta PEFT families such as
  Adapter/VPT should not be treated as main ADI claims.

This repository should not be used to claim universal ADI effectiveness across
all PEFT methods.

## Repository Structure

```text
configs/              Experiment configuration files used for CIFAR-100 runs
src/                  Cleaned experiment code subset
tools/                Evaluation and table-building utilities
scripts/              Figure-generation scripts
data/processed/       Processed paper tables in CSV format
data/figure_data/     Figure source data in CSV format
figures/              Generated figure PDFs
docs/                 Evidence notes, narrative boundaries, and release checks
patches/              Patches documenting DoRA implementation corrections
```

## Key Files

- `src/adi_lora/models/peft/lora_dora.py`  
  Core LoRA/DoRA modules and delta-scale control.

- `src/adi_lora/engine/run_fr_peft.py`  
  Main experiment runner for ADI-style CIFAR experiments.

- `src/adi_lora/engine/evaluate.py`  
  Evaluation loop used by the same-checkpoint alpha-sweep tools.

- `tools/eval_dora_same_checkpoint_alpha_sweep.py`  
  Same-checkpoint alpha evaluation for DoRA.

- `tools/eval_dora_cifar100c_same_checkpoint.py`  
  CIFAR-100-C same-checkpoint corruption evaluation for DoRA.

- `tools/build_adi_tables.py`  
  Aggregation script documenting how the processed ADI result tables were built
  from raw experiment packages. The raw packages themselves are not redistributed.

- `data/processed/Main_CIFAR100_Clean.csv`  
  Main CIFAR-100 clean interpolation-shift table.

- `data/processed/Main_CIFAR100C.csv`  
  Main/supplementary CIFAR-100-C corruption table.

- `data/figure_data/fig2_cifar100_paired_clean.csv`  
  Source data for the same-checkpoint clean-shift figure.

- `data/figure_data/fig3_alpha_selection_validation.csv`  
  Validation alpha-selection curve data.

- `data/figure_data/fig3_full_alpha_nearest_diagnostic.csv`  
  Diagnostic full-alpha Nearest curve data, when available.

## Environment

The experiments were run in a PyTorch/timm environment. A minimal Python
environment is:

```bash
pip install -r requirements.txt
```

The code expects datasets and pretrained ViT weights to be available locally.
Raw datasets, model checkpoints, and pretrained weights are not redistributed
here. Set `dataset.root` and, if needed, `backbone.checkpoint_path` in the YAML
configuration files before running experiments.

## Reproducing Training and Evaluation

Example CIFAR-100 ADI-LoRA training run:

```bash
export PYTHONPATH=$PWD/src
python -m adi_lora.engine.run_fr_peft \
  --config configs/fr_peft_cifar100_delta_lora.yaml \
  --seed 42
```

Example CIFAR-100 DoRA training run:

```bash
export PYTHONPATH=$PWD/src
python -m adi_lora.engine.run_fr_peft \
  --config configs/fr_peft_cifar100_dora.yaml \
  --seed 42
```

Dataset roots and local pretrained-checkpoint paths may need to be adjusted in
the configuration files.

## Reproducing Tables and Figures

The processed tables used by the manuscript are released directly under
`data/processed/`. To inspect them:

```bash
python - <<'PY'
import pandas as pd
print(pd.read_csv("data/processed/Main_CIFAR100_Clean.csv"))
print(pd.read_csv("data/processed/Main_CIFAR100C.csv").head())
PY
```

To regenerate manuscript figures from the released CSV files:

```bash
python scripts/plot_figures.py
python scripts/plot_fig3_nearest_diagnostic.py
```

The released plotting scripts operate on the CSV files in `data/figure_data/`.
The table aggregation script records the original aggregation logic, but
requires raw experiment packages that are not included in this public release.
The evaluation scripts expect locally available datasets and trained
checkpoints.

## Reproducing Same-checkpoint Evaluation

Example DoRA same-checkpoint alpha evaluation:

```bash
python tools/eval_dora_same_checkpoint_alpha_sweep.py \
  --run-dir outputs/fr_peft_cifar100/dora_seed42_YYYYMMDD_HHMMSS \
  --out-dir /tmp/cifar100_dora_seed42_same_checkpoint_alpha_eval \
  --alphas 0.2,0.4,0.6,0.8,1.0 \
  --seed 42 \
  --eval-all-test-alphas
```

This evaluates final checkpoints only. Nearest interpolation is not used for
alpha selection.

## Data Availability Statement

Suggested manuscript wording after the GitHub repository is created:

```text
The processed experimental tables, figure data, cleaned implementation code,
and analysis scripts supporting the findings of this study are publicly
available at: https://github.com/USER_OR_ORG/ADI-LoRA.
```

Replace the URL after creating the actual GitHub repository.

## What Is Not Included

- Model checkpoints (`*.pth`, `*.pt`, `*.safetensors`)
- Raw CIFAR-100/CIFAR-100-C/Tiny-ImageNet/ImageNet-100 datasets
- AutoDL temporary logs unrelated to the processed evidence
- LaTeX build artifacts

## Citation

If you use this repository, please cite the accompanying manuscript. A
`CITATION.cff` template is included and should be updated with the final title,
authors, DOI, and repository URL once available.
