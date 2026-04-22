import argparse
import json
import os
from glob import glob
from typing import Dict, List

import torch
import yaml
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--input_dir", type=str, default="data/faces")
    parser.add_argument("--output_dir", type=str, default="outputs/baseline")
    parser.add_argument("--max_images", type=int, default=None)
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def collect_images(input_dir: str) -> List[str]:
    patterns = ["*.jpg", "*.jpeg", "*.png"]
    paths = []
    for pattern in patterns:
        paths.extend(glob(os.path.join(input_dir, pattern)))
    return sorted(paths)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    model_id = cfg["model_id"]
    resolution = int(cfg["resolution"])
    num_steps = int(cfg["num_inference_steps"])
    guidance_scale = float(cfg["guidance_scale"])
    strength = float(cfg["edit_strength"])
    prompt = cfg["edit_prompt"]
    seed = int(cfg.get("seed", 42))
    num_images = int(args.max_images or cfg["num_images"])

    image_paths = collect_images(args.input_dir)[:num_images]
    if not image_paths:
        raise RuntimeError(f"No input images found in {args.input_dir}")

    inputs_dir = os.path.join(args.output_dir, "inputs")
    edits_dir = os.path.join(args.output_dir, "edits")
    os.makedirs(inputs_dir, exist_ok=True)
    os.makedirs(edits_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)
    pipe.safety_checker = None
    pipe.set_progress_bar_config(disable=True)

    run_info = {
        "model_id": model_id,
        "resolution": resolution,
        "num_inference_steps": num_steps,
        "guidance_scale": guidance_scale,
        "edit_strength": strength,
        "edit_prompt": prompt,
        "seed": seed,
        "num_images": len(image_paths),
    }

    for i, path in enumerate(image_paths):
        image = Image.open(path).convert("RGB").resize((resolution, resolution))
        image.save(os.path.join(inputs_dir, f"in_{i:03d}.jpg"))

        generator = torch.Generator(device=device).manual_seed(seed + i)
        result = pipe(
            prompt=prompt,
            image=image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            generator=generator,
        ).images[0]

        result.save(os.path.join(edits_dir, f"edit_{i:03d}.jpg"))

    with open(os.path.join(args.output_dir, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)

    print(args.output_dir)


if __name__ == "__main__":
    main()
