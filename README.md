# XAI Baseline Implementation

Baseline implementation for explainable real face editing with Stable Diffusion img2img.

The project studies diffusion-based face editing as both a generative modeling task and an explainability problem. We build a simple, reproducible editing pipeline for real portraits, evaluate the trade-off between edit strength and preservation, and use the results as a baseline for later attention-based analysis and control.

Our work is inspired by *Prompt-to-Prompt Image Editing with Cross Attention Control* (Hertz et al., 2022), which motivates the use of cross-attention as an interpretability signal in text-guided image editing.

---

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Login to HuggingFace to access Stable Diffusion weights:

```bash
hf auth login
```

## Data

Place input face images in:

`data/faces/`

We use a small subset of aligned portrait images for reproducible experiments.

## Project Structure

Main components:

```text
configs/                    # configuration files
scripts/                    # shell entry points
src/                        # experiment runners and evaluation code
data/faces/                 # input face images
results/                    # generated outputs, metrics, plots, report artifacts
```

## Running the Project

### 1. Run the baseline pipeline

```bash
bash scripts/run_baseline.sh
```

This pipeline:

- generates img2img edits for the target attribute,
- builds qualitative grids,
- computes CLIP similarity,
- runs a reconstruction sanity check.

### 2. Run the parameter sweep

```bash
bash scripts/run_prompt1_ablation.sh
```

This evaluates multiple combinations of `strength` and `guidance_scale` for the smile edit and saves per-run metrics together with a summary of the best trade-off configuration.

### 3. Run the full noise-edit experiment

```bash
bash scripts/run_noise_edit_full.sh
```

This runs the latent-noise editing pipeline across multiple prompts such as smile, glasses, and bangs, and produces aggregated outputs for analysis and reporting.

## Main Outputs

Results are saved under:

`results/`

Typical artifacts include:

```text
results/
├── edits/                    # generated edits
├── inputs/                   # copied input images
├── recon/                    # reconstruction sanity check outputs
├── grids/                    # qualitative visualization grids
├── prompt1_ablation/         # parameter sweep results
├── noise_edit_full/          # multi-prompt noise-edit results
└── metrics_clip.json         # baseline CLIP metric
```

Important report-ready files include:

- `grids/baseline_input_vs_edit.jpg`
- `grids/sanity_input_vs_recon.jpg`
- `prompt1_ablation/summary_prompt1.json`
- `noise_edit_full/all_runs.csv`
- `noise_edit_full/best_per_prompt.csv`
- `noise_edit_full/clip_vs_lpips.png`
- `noise_edit_full/overall_summary_table.json`

## Evaluation

We evaluate the system with both generation and preservation metrics:

- CLIP — prompt alignment
- LPIPS — perceptual drift from the source image
- SSIM — structural preservation

For parameter comparison we also use:

`tradeoff_score = clip_mean - 0.25 * lpips_mean`

This helps identify a balanced operating point rather than simply maximizing edit intensity.

## Experimental Setting

Default setup:

- model: Stable Diffusion v1.5
- task: real face img2img editing
- data: 20 face images
- inference steps: 30
- sweep parameters:
  - strength ∈ {0.3, 0.5, 0.6, 0.7}
  - guidance_scale ∈ {5.0, 7.5, 10.0}

This gives:

- 12 parameter configurations
- 20 images per configuration
- 240 generated edited samples in the ablation study

## What the Project Shows

The baseline confirms a consistent trade-off in diffusion-based real-face editing:

- stronger edits improve prompt alignment,
- but also increase drift from the original face,
- so the best practical setting is usually a balanced one rather than the strongest one.

This makes the project useful both as a generative baseline and as an XAI baseline for studying how edits propagate through the model.

## Configuration

Main settings are stored in:

`configs/baseline.yaml`

These include:

- prompts
- edit strength
- inference steps
- sample count

## Compute

The project was tested with Stable Diffusion v1.5 and can run on CPU, although inference is slower than on GPU.

## Notes

This repository provides the baseline implementation for:

- reproducible img2img face editing,
- controlled parameter sweeps,
- metric-based operating point selection,
- qualitative and quantitative analysis for XAI-oriented reporting.

The codebase is intended as a clean reference system for explainable diffusion editing experiments.
