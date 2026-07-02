#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
mkdir -p outputs/dora_audit

python tools/test_dora_alpha_equivalence.py | tee outputs/dora_audit/alpha_equivalence.log
python tools/audit_dora_single_batch.py \
  --config configs/fr_peft_cifar10_dora.yaml \
  --seed 42 \
  --out outputs/dora_audit/single_batch_dora_seed42.json | tee outputs/dora_audit/single_batch_dora_seed42.log

python -m adi_lora.engine.run_fr_peft \
  --config configs/fr_peft_cifar10_dora.yaml \
  --seed 42 \
  --max-train-batches 10 \
  --max-eval-batches 10 | tee outputs/dora_audit/dora_seed42_10batch_sanity.log

python -m adi_lora.engine.run_fr_peft \
  --config configs/fr_peft_cifar10_delta_dora.yaml \
  --seed 42 \
  --max-train-batches 10 \
  --max-eval-batches 10 | tee outputs/dora_audit/adi_dora_seed42_10batch_sanity.log
