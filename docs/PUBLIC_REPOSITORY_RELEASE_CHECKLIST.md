# Public Repository Release Checklist

This checklist defines what should be released for the IVC submission data availability statement.
The goal is to support the manuscript's reported tables and figures without overstating missing or pilot evidence.

## Repository URL Placeholder

Current manuscript placeholder:

```text
https://github.com/USER_OR_ORG/ADI-LoRA
```

Before submission, replace it in:

- `sections/08_declarations_ivc.tex`
- `DECLARATIONS_IVC_DRAFT.md`

## Recommended Repository Contents

Include:

- `README.md` explaining ADI-LoRA / ADI-DoRA evidence boundaries.
- `data/figure_data/fig2_cifar100_paired_clean.csv`
- `data/figure_data/fig_cifar100c_summary.csv`
- final manuscript tables in CSV/XLSX form if available.
- checked summary tables used for CIFAR-100 clean interpolation.
- checked summary tables used for CIFAR-100-C ADI-DoRA subset.
- Tiny-ImageNet LoRA three-seed summary.
- Tiny-ImageNet DoRA seed42 supplementary summary.
- ImageNet-100 seed42 pilot summary with clear pilot-only label.
- plotting scripts used to regenerate final manuscript figures.
- table-generation scripts or notebooks if available.
- protocol checklist documenting:
  - alpha selected only from Bicubic/Bilinear validation;
  - Nearest not used for alpha selection;
  - corruption not used for alpha selection;
  - final checkpoint only;
  - same-checkpoint comparison.

## Do Not Release As Formal Evidence

Do not present the following as formal main evidence:

- LoRA CIFAR-100-C formal results unless the missing raw CSV is supplied.
- ImageNet-100 as a standard benchmark.
- Tiny-ImageNet DoRA as a three-seed result.
- VPT / Adapter diagnostic alpha sweeps as main ADI evidence.
- best-Nearest checkpoint results.
- any target-selected alpha results.

## Dataset And Weight Boundaries

- Do not upload third-party dataset images unless license and redistribution rights are clear.
- Do not upload ImageNet-derived image files unless redistribution is permitted.
- If using only generated split manifests, upload the manifest rather than image data.
- Model checkpoints are not required for the manuscript tables unless the repository explicitly claims full checkpoint reproducibility.
- If checkpoints are omitted, state that the repository supports processed-result and figure/table reproducibility.

## Recommended README Wording

```text
This repository contains processed experimental tables, figure data, plotting scripts,
and protocol audit files for the manuscript "Adapter Delta Interpolation for Robust
Visual Adaptation in Vision Transformers".

The repository supports reproduction of the reported tables and figures from checked
processed records. It does not claim to provide a complete raw dataset mirror or full
model checkpoint archive.

ImageNet-100 entries are pilot-only and use a generated class-stratified symlink split.
LoRA CIFAR-100-C is not included as a formal claim unless raw source tables are supplied.
```

## Final Pre-Submission Check

- No placeholder URL remains in the manuscript.
- Public repository is accessible without login.
- Repository README states evidence boundaries.
- Repository files match the manuscript tables.
- Missing-source limitations match `tables/tableA2_missing_sources.tex`.
- Figure data match the final plotted figures.
