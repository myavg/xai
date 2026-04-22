import argparse
import math
import os
from glob import glob
from typing import List

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="outputs/baseline/inputs")
    parser.add_argument("--edit_dir", type=str, default="outputs/baseline/edits")
    parser.add_argument("--output_path", type=str, default="outputs/baseline/grids/baseline_input_vs_edit.jpg")
    parser.add_argument("--max_images", type=int, default=None)
    return parser.parse_args()


def collect_images(path: str, prefix: str) -> List[str]:
    patterns = [f"{prefix}_*.jpg", f"{prefix}_*.jpeg", f"{prefix}_*.png"]
    files = []
    for pattern in patterns:
        files.extend(glob(os.path.join(path, pattern)))
    return sorted(files)


def main() -> None:
    args = parse_args()
    inputs = collect_images(args.input_dir, "in")
    edits = collect_images(args.edit_dir, "edit")

    n = min(len(inputs), len(edits))
    if args.max_images is not None:
        n = min(n, args.max_images)
    if n == 0:
        raise RuntimeError("No paired input/edit images found")

    sample_inp = Image.open(inputs[0]).convert("RGB")
    w, h = sample_inp.size

    grid = Image.new("RGB", (2 * w, n * h))
    for i in range(n):
        inp = Image.open(inputs[i]).convert("RGB").resize((w, h))
        out = Image.open(edits[i]).convert("RGB").resize((w, h))
        grid.paste(inp, (0, i * h))
        grid.paste(out, (w, i * h))

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    grid.save(args.output_path)
    print(args.output_path)


if __name__ == "__main__":
    main()
