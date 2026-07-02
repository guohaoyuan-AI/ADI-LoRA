#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${1:-dora_audit_results_no_ckpt.tar.gz}
mkdir -p outputs/dora_audit_pack
find outputs/dora_audit -type f \
  ! -name '*.pth' ! -name '*.pt' ! -name '*.safetensors' \
  > outputs/dora_audit_pack/files.txt

tar -czf "$OUT" \
  scripts/run_dora_audit_seed42.sh \
  scripts/pack_dora_audit_results.sh \
  tools/test_dora_alpha_equivalence.py \
  tools/audit_dora_single_batch.py \
  tools/audit_dora_checkpoint_state.py \
  outputs/dora_audit_pack/files.txt \
  $(cat outputs/dora_audit_pack/files.txt)

echo "DONE: $OUT"
