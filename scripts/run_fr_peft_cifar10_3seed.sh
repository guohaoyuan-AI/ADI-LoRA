#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
for seed in 42 43 44; do
  for cfg in \
    configs/fr_peft_cifar10_lora.yaml \
    configs/fr_peft_cifar10_dora.yaml \
    configs/fr_peft_cifar10_lp_lora.yaml \
    configs/fr_peft_cifar10_lp_dora.yaml \
    configs/fr_peft_cifar10_delta_lora.yaml \
    configs/fr_peft_cifar10_delta_dora.yaml \
    configs/fr_peft_cifar10_lp_delta_lora.yaml \
    configs/fr_peft_cifar10_lp_delta_dora.yaml; do
    echo "===== RUN ${cfg} seed=${seed} ====="
    python -m adi_lora.engine.run_fr_peft --config "${cfg}" --seed "${seed}"
  done
done
