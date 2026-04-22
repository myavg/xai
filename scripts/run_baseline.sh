#!/usr/bin/env sh
set -eu

CONFIG="${1:-configs/baseline.yaml}"
INPUT_DIR="${2:-data/faces}"
OUTPUT_DIR="${3:-outputs/baseline}"

mkdir -p "$OUTPUT_DIR"

python src/run_batch_baseline.py \
  --config "$CONFIG" \
  --input_dir "$INPUT_DIR" \
  --output_dir "$OUTPUT_DIR"

python src/make_grid.py \
  --input_dir "$OUTPUT_DIR/inputs" \
  --edit_dir "$OUTPUT_DIR/edits" \
  --output_path "$OUTPUT_DIR/grids/baseline_input_vs_edit.jpg"

python src/eval_clip.py \
  --config "$CONFIG" \
  --edits_dir "$OUTPUT_DIR/edits" \
  --output_path "$OUTPUT_DIR/metrics_clip.json"

python src/run_recon_sanity.py \
  --config "$CONFIG" \
  --input_dir "$OUTPUT_DIR/inputs" \
  --output_dir "$OUTPUT_DIR/recon"

python src/make_grid_recon.py \
  --input_dir "$OUTPUT_DIR/inputs" \
  --recon_dir "$OUTPUT_DIR/recon" \
  --output_path "$OUTPUT_DIR/grids/sanity_input_vs_recon.jpg"
