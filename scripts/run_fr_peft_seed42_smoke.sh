#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
python -m adi_lora.engine.run_fr_peft --config configs/fr_peft_cifar10_lora.yaml --seed 42 --max-train-batches 2 --max-eval-batches 2
python -m adi_lora.engine.run_fr_peft --config configs/fr_peft_cifar10_dora.yaml --seed 42 --max-train-batches 2 --max-eval-batches 2
python -m adi_lora.engine.run_fr_peft --config configs/fr_peft_cifar10_lp_delta_lora.yaml --seed 42 --max-train-batches 2 --max-eval-batches 2
python -m adi_lora.engine.run_fr_peft --config configs/fr_peft_cifar10_lp_delta_dora.yaml --seed 42 --max-train-batches 2 --max-eval-batches 2
