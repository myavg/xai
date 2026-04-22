#!/usr/bin/env sh
set -eu

CONFIG="${1:-configs/multi_prompt.yaml}"
INPUT_DIR="${2:-data/faces}"
OUTPUT_DIR="${3:-outputs/multi_prompt}"

mkdir -p "$OUTPUT_DIR"

python src/run_multi_prompt_ablation.py \
  --config "$CONFIG" \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR"

python src/summarize_multi_prompt_results.py \
  --input_dir "$OUTPUT_DIR" \
  --output_dir "$OUTPUT_DIR"

python src/prepare_report_assets.py \
  --results_dir "$OUTPUT_DIR" \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR/for_report"
