#!/usr/bin/env bash

set -e

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_ROOT"

INPUT_DIR="data/faces"

if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: data/faces directory does not exist"
    exit 1
fi

NUM_IMAGES=$(find "$INPUT_DIR" -type f \( -name "*.jpg" -o -name "*.png" \) | wc -l)

if [ "$NUM_IMAGES" -eq 0 ]; then
    echo "ERROR: data/faces is empty"
    exit 1
fi

echo "Using existing images from $INPUT_DIR"
echo "Number of images: $NUM_IMAGES"

chmod -R a-w "$INPUT_DIR"

RUN_NAME=$(date +"run_%Y-%m-%d_%H-%M-%S")
OUTPUT_ROOT="outputs/final_submission/${RUN_NAME}"

echo "Run name: ${RUN_NAME}"
echo "Output root: ${OUTPUT_ROOT}"

mkdir -p "${OUTPUT_ROOT}"
mkdir -p "${OUTPUT_ROOT}/baseline"
mkdir -p "${OUTPUT_ROOT}/baseline/grids"
mkdir -p "${OUTPUT_ROOT}/prompt1_ablation"
mkdir -p "${OUTPUT_ROOT}/multi_prompt"
mkdir -p "${OUTPUT_ROOT}/noise_edit_full"
mkdir -p "${OUTPUT_ROOT}/report_assets"
mkdir -p logs

echo "Starting baseline..."

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python src/run_batch_baseline.py \
    --config configs/baseline.yaml \
    --input_dir "$INPUT_DIR" \
    --output_dir "${OUTPUT_ROOT}/baseline"

echo "Evaluating CLIP..."

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python src/eval_clip.py \
    --edits_dir "${OUTPUT_ROOT}/baseline/edits" \
    --output_path "${OUTPUT_ROOT}/baseline/metrics_clip.json" \
    --prompt "a smiling person, portrait, high quality"

echo "Running reconstruction sanity..."

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python src/run_recon_sanity.py \
    --config configs/baseline.yaml \
    --input_dir "${OUTPUT_ROOT}/baseline/inputs" \
    --output_dir "${OUTPUT_ROOT}/baseline/recon"

echo "Creating grids..."

python src/make_grid.py \
    --input_dir "${OUTPUT_ROOT}/baseline/inputs" \
    --edit_dir "${OUTPUT_ROOT}/baseline/edits" \
    --output_path "${OUTPUT_ROOT}/baseline/grids/baseline_input_vs_edit.jpg"

python src/make_grid_recon.py \
    --input_dir "${OUTPUT_ROOT}/baseline/inputs" \
    --recon_dir "${OUTPUT_ROOT}/baseline/recon" \
    --output_path "${OUTPUT_ROOT}/baseline/grids/sanity_input_vs_recon.jpg"

echo "Running prompt1 ablation..."

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python src/run_prompt1_ablation.py \
    --config configs/baseline.yaml \
    --input_dir "$INPUT_DIR" \
    --output_dir "${OUTPUT_ROOT}/prompt1_ablation" \
    --strengths 0.3,0.5,0.6,0.7 \
    --guidance_scales 5.0,7.5,10.0

echo "Running multi-prompt ablation..."

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python src/run_multi_prompt_ablation.py \
    --config configs/multi_prompt.yaml \
    --input_dir "$INPUT_DIR" \
    --output_dir "${OUTPUT_ROOT}/multi_prompt"

echo "Summarizing multi-prompt results..."

python src/summarize_multi_prompt_results.py \
    --input_dir "${OUTPUT_ROOT}/multi_prompt" \
    --output_dir "${OUTPUT_ROOT}/multi_prompt"

echo "Preparing report assets..."

python src/prepare_report_assets.py \
    --results_dir "${OUTPUT_ROOT}/multi_prompt" \
    --input_dir "$INPUT_DIR" \
    --output_dir "${OUTPUT_ROOT}/report_assets" \
    --top_k 3

echo "Running noise-first branch..."

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} python src/run_noise_edit_full.py \
    --config configs/noise_edit_full.yaml \
    --input_dir "$INPUT_DIR" \
    --output_dir "${OUTPUT_ROOT}/noise_edit_full"

echo "Creating manifest..."

cat <<EOF > "${OUTPUT_ROOT}/manifest.json"
{
  "run_name": "${RUN_NAME}",
  "num_images": ${NUM_IMAGES},
  "timestamp": "$(date)"
}
EOF

chmod -R u+w "$INPUT_DIR"

echo "Final submission pipeline completed"
echo "Results saved to: ${OUTPUT_ROOT}"