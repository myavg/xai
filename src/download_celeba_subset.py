import argparse
import json
import os
import random
import shutil
from glob import glob
from typing import List

from torchvision.datasets import CelebA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--root", type=str, default="data/celeba_download")
    parser.add_argument("--output_dir", type=str, default="data/faces")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num_images", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strategy", type=str, default="evenly_spaced", choices=["evenly_spaced", "random"])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def supported_files(path: str) -> List[str]:
    exts = ("*.jpg", "*.jpeg", "*.png")
    files = []
    for ext in exts:
        files.extend(glob(os.path.join(path, ext)))
    return sorted(files)


def build_indices(total: int, count: int, strategy: str, seed: int) -> List[int]:
    count = min(count, total)
    if strategy == "random":
        rng = random.Random(seed)
        return sorted(rng.sample(range(total), count))
    if count == 1:
        return [0]
    step = max(1, total // count)
    return list(range(0, total, step))[:count]


def main() -> None:
    args = parse_args()

    os.makedirs(args.root, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.overwrite:
        for path in supported_files(args.output_dir):
            os.remove(path)

    existing = supported_files(args.output_dir)
    if len(existing) >= args.num_images:
        metadata = {
            "root": args.root,
            "output_dir": args.output_dir,
            "split": args.split,
            "num_images": args.num_images,
            "seed": args.seed,
            "strategy": args.strategy,
            "status": "reused_existing_images",
        }
        with open(os.path.join(args.output_dir, "subset_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(args.output_dir)
        return

    ds = CelebA(root=args.root, split=args.split, download=True)
    src_dir = os.path.join(args.root, "celeba", "img_align_celeba")
    if not os.path.isdir(src_dir):
        raise RuntimeError(f"Missing image directory: {src_dir}")

    indices = build_indices(len(ds), args.num_images, args.strategy, args.seed)
    copied = []
    for out_idx, dataset_idx in enumerate(indices):
        filename = ds.filename[dataset_idx]
        src = os.path.join(src_dir, filename)
        dst = os.path.join(args.output_dir, f"face_{out_idx:03d}.jpg")
        if not os.path.isfile(src):
            raise FileNotFoundError(src)
        shutil.copy2(src, dst)
        copied.append({"dataset_index": dataset_idx, "source_filename": filename, "output_filename": os.path.basename(dst)})

    metadata = {
        "root": args.root,
        "output_dir": args.output_dir,
        "split": args.split,
        "num_images": len(copied),
        "seed": args.seed,
        "strategy": args.strategy,
        "copied": copied,
    }
    with open(os.path.join(args.output_dir, "subset_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(args.output_dir)


if __name__ == "__main__":
    main()
