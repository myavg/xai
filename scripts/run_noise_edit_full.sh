#!/usr/bin/env sh
set -eu

CONFIG="${1:-configs/noise_edit_full.yaml}"
INPUT_DIR="${2:-data/faces}"
OUTPUT_DIR="${3:-outputs/noise_edit_full}"

mkdir -p "$OUTPUT_DIR"

python src/run_noise_edit_full.py \
  --config "$CONFIG" \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR"
