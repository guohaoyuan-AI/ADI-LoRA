#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
python tests/test_fr_peft_forward_backward.py
python tests/test_delta_interpolation.py
python tests/test_lp_stage_checkpoint.py
python tests/test_fr_metrics.py
python tests/test_trainable_params.py
