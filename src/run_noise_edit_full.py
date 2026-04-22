import argparse
import inspect
import json
import os
import shutil
from glob import glob
from typing import Dict, List, Tuple

import lpips
import matplotlib.pyplot as plt
import numpy as np
import open_clip
import pandas as pd
import torch
import yaml
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/noise_edit_full.yaml")
    parser.add_argument("--input_dir", type=str, default="data/faces")
    parser.add_argument("--output_dir", type=str, default="outputs/noise_edit_full")
    parser.add_argument("--max_images", type=int, default=None)
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def pil_to_numpy_uint8(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"), dtype=np.uint8)


def compute_ssim(inp: Image.Image, out: Image.Image) -> float:
    return float(ssim(pil_to_numpy_uint8(inp), pil_to_numpy_uint8(out), channel_axis=2, data_range=255))


def pil_to_lpips_tensor(image: Image.Image) -> torch.Tensor:
    tensor = transforms.ToTensor()(image).unsqueeze(0)
    return tensor * 2.0 - 1.0


def pil_to_vae_tensor(image: Image.Image, resolution: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    image = image.convert("RGB").resize((resolution, resolution))
    arr = np.array(image).astype(np.float32) / 255.0
    arr = arr[None].transpose(0, 3, 1, 2)
    tensor = torch.from_numpy(arr).to(device=device, dtype=dtype)
    return tensor * 2.0 - 1.0


def encode_prompt(pipe: StableDiffusionPipeline, prompt: str, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    text_inputs = tokenizer([prompt], padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    uncond_inputs = tokenizer([""], padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt")

    with torch.no_grad():
        text_embeds = text_encoder(text_inputs.input_ids.to(device))[0]
        uncond_embeds = text_encoder(uncond_inputs.input_ids.to(device))[0]
    return text_embeds, uncond_embeds


def latent_noise_edit(
    pipe: StableDiffusionPipeline,
    image: Image.Image,
    text_embeds: torch.Tensor,
    uncond_embeds: torch.Tensor,
    strength: float,
    guidance_scale: float,
    num_inference_steps: int,
    generator: torch.Generator,
    resolution: int,
    device: str,
    dtype: torch.dtype,
) -> Image.Image:
    with torch.no_grad():
        image_tensor = pil_to_vae_tensor(image, resolution, device, dtype)
        init_latents = pipe.vae.encode(image_tensor).latent_dist.sample(generator=generator)
        init_latents = init_latents * pipe.vae.config.scaling_factor

        pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = pipe.scheduler.timesteps[t_start:]
        if len(timesteps) == 0:
            timesteps = pipe.scheduler.timesteps[-1:]

        noise = torch.randn(init_latents.shape, generator=generator, device=device, dtype=init_latents.dtype)
        latents = pipe.scheduler.add_noise(init_latents, noise, timesteps[0])

        encoder_hidden_states = torch.cat([uncond_embeds, text_embeds], dim=0)

        for t in timesteps:
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
            noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=encoder_hidden_states).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            step_kwargs = {}
            if "generator" in inspect.signature(pipe.scheduler.step).parameters:
                step_kwargs["generator"] = generator

            latents = pipe.scheduler.step(noise_pred, t, latents, **step_kwargs).prev_sample

        latents = latents / pipe.vae.config.scaling_factor
        image_out = pipe.vae.decode(latents).sample
        image_out = (image_out / 2 + 0.5).clamp(0, 1)
        image_out = image_out.detach().cpu().permute(0, 2, 3, 1).numpy()[0]
        image_out = (image_out * 255).round().astype(np.uint8)
        return Image.fromarray(image_out)


def get_font(size: int = 18):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def make_tile(image: Image.Image, title: str, subtitle: str = "", width: int = 320) -> Image.Image:
    image = image.convert("RGB")
    w, h = image.size
    new_h = int(h * width / w)
    image = image.resize((width, new_h))

    top_h = 40 if subtitle else 28
    canvas = Image.new("RGB", (width, new_h + top_h), (24, 24, 24))
    canvas.paste(image, (0, top_h))

    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), title, font=get_font(16), fill=(240, 240, 240))
    if subtitle:
        draw.text((8, 22), subtitle, font=get_font(13), fill=(200, 200, 200))
    return canvas


def hstack(images: List[Image.Image], gap: int = 10) -> Image.Image:
    out_w = sum(im.width for im in images) + gap * (len(images) - 1)
    out_h = max(im.height for im in images)
    out = Image.new("RGB", (out_w, out_h), (255, 255, 255))
    x = 0
    for im in images:
        out.paste(im, (x, 0))
        x += im.width + gap
    return out


def vstack(images: List[Image.Image], gap: int = 14) -> Image.Image:
    out_w = max(im.width for im in images)
    out_h = sum(im.height for im in images) + gap * (len(images) - 1)
    out = Image.new("RGB", (out_w, out_h), (255, 255, 255))
    y = 0
    for im in images:
        out.paste(im, (0, y))
        y += im.height + gap
    return out


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
    strengths = [float(x) for x in cfg["strengths"]]
    prompts = cfg["prompts"]
    seed = int(cfg.get("seed", 42))
    num_images = int(args.max_images or cfg["num_images"])

    image_paths = collect_images(args.input_dir)[:num_images]
    if not image_paths:
        raise RuntimeError(f"No input images found in {args.input_dir}")

    os.makedirs(args.output_dir, exist_ok=True)
    for prompt_spec in prompts:
        os.makedirs(os.path.join(args.output_dir, prompt_spec["name"]), exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype).to(device)
    pipe.safety_checker = None
    pipe.set_progress_bar_config(disable=True)

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
    clip_model = clip_model.to(device).eval()
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()

    all_rows: List[Dict] = []
    run_rows: List[Dict] = []

    for prompt_idx, prompt_spec in enumerate(prompts):
        prompt_name = prompt_spec["name"]
        edit_prompt = prompt_spec["edit_prompt"]
        text_embeds, uncond_embeds = encode_prompt(pipe, edit_prompt, device)
        text_tokens = clip_tokenizer([edit_prompt]).to(device)
        with torch.no_grad():
            text_feat = clip_model.encode_text(text_tokens)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        for strength in strengths:
            run_name = f"s{strength:.2f}_g{guidance_scale:.1f}"
            run_dir = os.path.join(args.output_dir, prompt_name, run_name)
            os.makedirs(run_dir, exist_ok=True)

            clip_scores: List[float] = []
            lpips_scores: List[float] = []
            ssim_scores: List[float] = []

            for i, path in enumerate(image_paths):
                image_id = os.path.splitext(os.path.basename(path))[0]
                inp = Image.open(path).convert("RGB").resize((resolution, resolution))
                sample_seed = seed + prompt_idx * 10000 + int(strength * 1000) + i
                generator = torch.Generator(device=device).manual_seed(sample_seed)

                out = latent_noise_edit(
                    pipe=pipe,
                    image=inp,
                    text_embeds=text_embeds,
                    uncond_embeds=uncond_embeds,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_steps,
                    generator=generator,
                    resolution=resolution,
                    device=device,
                    dtype=dtype,
                )

                out_path = os.path.join(run_dir, f"edit_{i:03d}.jpg")
                out.save(out_path)

                with torch.no_grad():
                    out_clip = clip_preprocess(out).unsqueeze(0).to(device)
                    img_feat = clip_model.encode_image(out_clip)
                    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                    clip_val = float((img_feat @ text_feat.T).item())

                    inp_lp = pil_to_lpips_tensor(inp).to(device)
                    out_lp = pil_to_lpips_tensor(out).to(device)
                    lpips_val = float(lpips_model(inp_lp, out_lp).item())

                ssim_val = compute_ssim(inp, out)
                tradeoff = clip_val - 0.25 * lpips_val
                rel_path = os.path.relpath(out_path, args.output_dir)

                clip_scores.append(clip_val)
                lpips_scores.append(lpips_val)
                ssim_scores.append(ssim_val)

                all_rows.append(
                    {
                        "image_id": image_id,
                        "prompt_name": prompt_name,
                        "run_name": run_name,
                        "strength": strength,
                        "guidance_scale": guidance_scale,
                        "clip": clip_val,
                        "lpips": lpips_val,
                        "ssim": ssim_val,
                        "tradeoff": tradeoff,
                        "path": rel_path,
                    }
                )

            run_metrics = {
                "prompt_name": prompt_name,
                "edit_prompt": edit_prompt,
                "run_name": run_name,
                "strength": strength,
                "guidance_scale": guidance_scale,
                "n_images": len(image_paths),
                "clip_mean": float(np.mean(clip_scores)),
                "lpips_mean": float(np.mean(lpips_scores)),
                "ssim_mean": float(np.mean(ssim_scores)),
                "tradeoff_score": float(np.mean(clip_scores) - 0.25 * np.mean(lpips_scores)),
                "clip_scores": clip_scores,
                "lpips_scores": lpips_scores,
                "ssim_scores": ssim_scores,
            }

            with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(run_metrics, f, indent=2)

            run_rows.append(run_metrics)

    all_df = pd.DataFrame(all_rows).sort_values(["prompt_name", "run_name", "image_id"]).reset_index(drop=True)
    all_runs_path = os.path.join(args.output_dir, "all_runs.csv")
    all_df.to_csv(all_runs_path, index=False)

    run_df = pd.DataFrame(run_rows).sort_values(["prompt_name", "tradeoff_score"], ascending=[True, False]).reset_index(drop=True)
    best_df = run_df.groupby("prompt_name", as_index=False).first()

    best_records = []
    for _, row in best_df.iterrows():
        subset = all_df[(all_df["prompt_name"] == row["prompt_name"]) & (all_df["run_name"] == row["run_name"])].copy()
        subset = subset.sort_values("tradeoff", ascending=False)
        rep = subset.iloc[0]
        best_records.append(
            {
                "prompt_name": row["prompt_name"],
                "run_name": row["run_name"],
                "strength": float(row["strength"]),
                "guidance_scale": float(row["guidance_scale"]),
                "clip_mean": float(row["clip_mean"]),
                "lpips_mean": float(row["lpips_mean"]),
                "ssim_mean": float(row["ssim_mean"]),
                "tradeoff_score": float(row["tradeoff_score"]),
                "best_image_id": rep["image_id"],
                "best_image_path": rep["path"],
            }
        )

    best_per_prompt_df = pd.DataFrame(best_records).sort_values("prompt_name").reset_index(drop=True)
    best_per_prompt_path = os.path.join(args.output_dir, "best_per_prompt.csv")
    best_per_prompt_df.to_csv(best_per_prompt_path, index=False)

    overall_summary = {
        "model_id": model_id,
        "num_inference_steps": num_steps,
        "guidance_scale": guidance_scale,
        "strengths": strengths,
        "images_used": len(image_paths),
        "prompts": best_per_prompt_df.to_dict(orient="records"),
    }
    overall_summary_path = os.path.join(args.output_dir, "overall_summary_table.json")
    with open(overall_summary_path, "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, indent=2)

    plt.figure(figsize=(8, 6))
    for prompt_name, group in all_df.groupby("prompt_name"):
        plt.scatter(group["lpips"], group["clip"], alpha=0.7, label=prompt_name)
    plt.xlabel("LPIPS")
    plt.ylabel("CLIP")
    plt.title("CLIP vs LPIPS (noise edit)")
    plt.legend()
    plt.tight_layout()
    clip_plot_path = os.path.join(args.output_dir, "clip_vs_lpips.png")
    plt.savefig(clip_plot_path, dpi=200)
    plt.close()

    input_map = {os.path.splitext(os.path.basename(path))[0]: path for path in image_paths}
    overview_rows = []
    for _, row in best_per_prompt_df.iterrows():
        input_path = input_map[row["best_image_id"]]
        output_path = os.path.join(args.output_dir, row["best_image_path"])

        inp = Image.open(input_path).convert("RGB")
        out = Image.open(output_path).convert("RGB")
        left = make_tile(inp, f"{row['prompt_name']} | input #{row['best_image_id']}")
        right = make_tile(out, f"{row['prompt_name']} | {row['run_name']}", f"CLIP={row['clip_mean']:.3f} LPIPS={row['lpips_mean']:.3f} SSIM={row['ssim_mean']:.3f}")
        overview_rows.append(hstack([left, right]))
    best_overview = vstack(overview_rows)
    best_overview_path = os.path.join(args.output_dir, "best_overview.png")
    best_overview.save(best_overview_path)

    for_report_dir = os.path.join(args.output_dir, "for_report")
    os.makedirs(for_report_dir, exist_ok=True)
    shutil.copy2(best_per_prompt_path, os.path.join(for_report_dir, "best_per_prompt.csv"))
    shutil.copy2(clip_plot_path, os.path.join(for_report_dir, "clip_vs_lpips.png"))
    shutil.copy2(best_overview_path, os.path.join(for_report_dir, "best_overview.png"))
    shutil.copy2(overall_summary_path, os.path.join(for_report_dir, "overall_summary_table.json"))

    print(args.output_dir)


if __name__ == "__main__":
    main()
