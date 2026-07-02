#!/usr/bin/env bash
set -euo pipefail
OUT=${1:-fr_peft_cifar10_results_no_ckpt.tar.gz}
TMP=/tmp/fr_peft_pack_no_ckpt_$$
mkdir -p "$TMP"
rsync -av \
  --include='*/' \
  --include='*.csv' \
  --include='*.yaml' \
  --include='*.yml' \
  --include='*.json' \
  --include='*.txt' \
  --include='*.log' \
  --exclude='*.pth' \
  --exclude='*.pt' \
  --exclude='*.ckpt' \
  --exclude='*.safetensors' \
  --exclude='*' \
  outputs/fr_peft_cifar10/ "$TMP/outputs/" || true
cp -r configs "$TMP/"
cp -r logs "$TMP/" 2>/dev/null || true
tar -czf "$OUT" -C "$TMP" .
rm -rf "$TMP"
echo "Saved $OUT"
