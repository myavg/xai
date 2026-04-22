# Diffusion Face Editing

Real face editing with diffusion models, focusing on the trade-off between editability and preservation.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## One-command final run

```bash
bash scripts/run_final_submission.sh
```

## Main output

```text
outputs/final_submission/<run_name>/
```

## Experiments

- baseline
- prompt1 ablation
- multi-prompt ablation
- noise edit full

## Notes

- CelebA images are prepared automatically into `data/faces/`
- the project uses Stable Diffusion v1.5 weights from Hugging Face
