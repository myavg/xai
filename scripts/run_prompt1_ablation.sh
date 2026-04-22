#!/usr/bin/env sh
set -eu

CONFIG="${1:-configs/baseline.yaml}"
INPUT_DIR="${2:-data/faces}"
OUTPUT_DIR="${3:-outputs/prompt1_ablation}"

mkdir -p "$OUTPUT_DIR"

python src/run_prompt1_ablation.py \
  --config "$CONFIG" \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR"
