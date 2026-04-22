import argparse
import json
import os
from glob import glob
from typing import Dict, List

import open_clip
import torch
import yaml
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--edits_dir", type=str, default="outputs/baseline/edits")
    parser.add_argument("--output_path", type=str, default="outputs/baseline/metrics_clip.json")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--max_images", type=int, default=None)
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def collect_images(edits_dir: str) -> List[str]:
    patterns = ["edit_*.jpg", "edit_*.jpeg", "edit_*.png", "*.jpg", "*.jpeg", "*.png"]
    seen = set()
    paths = []
    for pattern in patterns:
        for p in sorted(glob(os.path.join(edits_dir, pattern))):
            if p not in seen:
                paths.append(p)
                seen.add(p)
    return paths


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    prompt = args.prompt or cfg["edit_prompt"]

    paths = collect_images(args.edits_dir)
    if args.max_images is not None:
        paths = paths[: args.max_images]
    if not paths:
        raise RuntimeError(f"No edited images found in {args.edits_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device).eval()

    text = tokenizer([prompt]).to(device)
    scores = []

    with torch.no_grad():
        text_feat = model.encode_text(text)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        for path in paths:
            img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            img_feat = model.encode_image(img)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            sim = float((img_feat @ text_feat.T).item())
            scores.append(sim)

    out = {
        "prompt": prompt,
        "n": len(scores),
        "clip_mean": sum(scores) / len(scores),
        "clip_scores": scores,
    }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(args.output_path)


if __name__ == "__main__":
    main()
