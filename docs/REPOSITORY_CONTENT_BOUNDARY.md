# Repository Content Boundary

This public repository is intended to release the evidence needed for paper
review and reproducibility without overclaiming the method scope.

## Included

- Cleaned experiment code subset used for LoRA/DoRA delta interpolation.
- DoRA repair patches.
- Experiment configurations used for CIFAR-100 LoRA/DoRA runs.
- Processed result tables used in the manuscript.
- Figure source data and plotting scripts.
- Same-checkpoint alpha evaluation scripts.

## Excluded

- Checkpoints and pretrained weights.
- Raw datasets.
- AutoDL temporary folders.
- Uncurated historical failed experiments.
- Adapter/VPT exploratory code as a main method implementation.

## Claim Boundary

The released evidence supports ADI for LoRA-style weight-delta PEFT. Adapter
and VPT diagnostics should be interpreted as boundary checks only, not as main
positive evidence.
