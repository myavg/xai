"""Run a single Stable Diffusion img2img baseline edit on one face image."""

from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image
import os

os.makedirs("results", exist_ok=True)

# Use fp16 on GPU for speed/memory, fp32 on CPU for compatibility.
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

print("Device:", device)

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=dtype
).to(device)

pipe.safety_checker = None

input_path = "data/sample_face.jpg"
image = Image.open(input_path).convert("RGB").resize((512, 512))

prompt = "a smiling person, portrait, high quality"

# A mid-level strength keeps identity while applying a visible smile edit.
result = pipe(
    prompt=prompt,
    image=image,
    strength=0.6,
    guidance_scale=7.5,
    num_inference_steps=30,
).images[0]

out_path = "results/test_edit.jpg"
result.save(out_path)
print("Saved to", out_path)
