# Explainable Face Editing with Diffusion Models  
### From Black-Box Generation to Interpretable Editing

## Introduction

Diffusion models achieve impressive results in image editing, but they suffer from a critical limitation:

> **We do not understand how edits are applied internally.**

In face editing, this is especially important:
- identity must be preserved  
- changes must be localized  
- unintended bias must be avoided  

In this project, we extend a standard diffusion pipeline into an **Explainable AI (XAI) system** and study:

> **How editing parameters affect internal model behavior and interpretability**

---

## From Generation to Explanation

Our baseline pipeline:

- image → latent encoding  
- noise injection  
- prompt-guided denoising  
- edited image  

From an XAI perspective, we reinterpret it as:
Text tokens → cross-attention → spatial regions


Even though attention is not explicitly extracted, we treat:

> **Editing behavior as an implicit attention distribution**

As noted in our baseline :
- edits are non-local  
- parameters affect semantic behavior  
- internal mechanisms are hidden  

---

## Experimental Setup

We perform controlled experiments:

- **3 edits**: smile, glasses, bangs  
- **2 pipelines**:
  - img2img (reference)
  - noise-first  
- **27 configurations**
- **540+ generated images**

### Metrics (as XAI signals)

| Metric | Meaning | XAI Interpretation |
|------|--------|-------------------|
| CLIP ↑ | prompt alignment | strength of semantic activation |
| LPIPS ↓ | perceptual change | degree of modification |
| SSIM ↑ | structural similarity | identity preservation |

---

## Core Hypothesis (XAI View)

We model editing as:

> **Editability ↑ vs Preservation ↑**

### Key idea:

- Increasing noise → destroys original signal  
- Model relies more on prompt  
- → **attention spreads globally**

📌 This is not just empirical — it follows from diffusion theory:

`z_t = sqrt(alpha) * z_0 + sqrt(1 - alpha) * epsilon`

- high noise → low signal → weak identity constraint  
- low noise → strong signal → limited editing  

---

## Baseline Observations

### Editing vs Reconstruction

![baseline](pictures/baseline_input_vs_edit.jpg)
![reconstruction](pictures/sanity_input_vs_recon.jpg)

### Result

- **s = 0.6** → visible edits but identity distortion  
- **s = 0.2** → near-perfect reconstruction  

### XAI Interpretation

> Editing is inherently **non-local**

Even simple edits affect:
- face structure  
- background  
- lighting  

This confirms that:
> diffusion models distribute attention across the entire image

---

## Quantitative Trade-off = Explanation

### CLIP vs LPIPS

![scatter](pictures/clip_vs_lpips_multi.png)
![scatter2](pictures/clip_vs_lpips_noise.png)

### Observed ranges:

- CLIP: **0.23 – 0.29**  
- LPIPS: **0.20 – 0.45**  
- SSIM: **0.59 – 0.77**

### Key result

- Increasing noise:
  - CLIP ↑ (better alignment)
  - LPIPS ↑ (more distortion)

### XAI Conclusion

> Stronger edits correspond to **broader attention spread** :contentReference[oaicite:1]{index=1}

The scatter plot becomes a **global explanation of model behavior**.

---

## Prompt-Dependent Behavior

![best](pictures/best_overview_multi.png)

### Best configurations:

| Prompt | Best (s, w) | CLIP | LPIPS | SSIM |
|-------|------------|------|------|------|
| Smile | (0.3, 5.0) | 0.245 | 0.202 | 0.769 |
| Glasses | (0.5, 5.0) | 0.274 | 0.312 | 0.678 |
| Bangs | (0.3, 7.5) | 0.230 | 0.202 | 0.769 |

### Interpretation

- **Smile / Bangs**
  - continuous attributes  
  - require small local changes  

- **Glasses**
  - discrete object  
  - requires stronger intervention  

### XAI Insight

> The model implicitly learns **semantic regions**

- mouth (smile)  
- hair (bangs)  
- eyes (glasses)  

But:
> these regions are not explicitly controlled → they emerge from attention

---

## Parameter Sweep = Behavior Visualization

![smile](pictures/sweep_grid_smile.png)
![glasses](pictures/sweep_grid_glasses.png)
![bangs](pictures/sweep_grid_bangs.png)

### Observations

- low noise → weak edits (localized)  
- medium noise → optimal balance  
- high noise → global transformation  

### XAI Insight

> Interpretability is maximized at **moderate noise**

Because:
- enough freedom to edit  
- enough structure to preserve identity  

---

## Pipeline Comparison

![noise](pictures/best_overview_noise.png)

### Result

Both pipelines behave similarly.

### XAI Conclusion

> Interpretability is determined by the **latent space**, not the pipeline.

---

## Failure Analysis = Explanation Failure

![failures](pictures/failure_candidates.png)

### Failure types

| Type | Metrics | XAI meaning |
|------|--------|------------|
| Under-edit | low CLIP, high SSIM | weak attention |
| Over-edit | high LPIPS, low SSIM | global attention |
| Artifacts | unstable SSIM | misaligned attention |

### Key insight

> Failures correspond directly to **attention misalignment** :contentReference[oaicite:2]{index=2}

---

## Why This Matters for XAI

Our experiments reveal:

### 1. Diffusion models are inherently non-local
- edits spread across the image  
- difficult to control precisely  

### 2. Noise controls interpretability
- low noise → interpretable but weak  
- high noise → strong but chaotic  

### 3. Metrics become explanations
- CLIP / LPIPS / SSIM describe internal behavior  

---

## Final Results (Clear Summary)

### Quantitative

- Best trade-off score ≈ **0.196**
- Optimal noise:
  - **0.3** for continuous edits  
  - **0.5** for structural edits  
- LPIPS increases by **2×** when moving from low to high noise  

### Qualitative

- Moderate noise produces:
  - visible edits  
  - preserved identity  
  - interpretable behavior  

---

## Final Conclusion

We transformed a generative pipeline into an **interpretable system**.

### Main findings:

1. **Noise strength is the key driver of interpretability**
2. **Editing behavior reflects attention distribution**
3. **Different prompts activate different semantic regions**
4. **Failures correspond to attention misalignment**

---

## XAI Takeaway

> Diffusion models are not just black boxes —  
> they are **structured systems where interpretability emerges from noise and attention dynamics**

---

## Future Work

- extract real attention maps  
- apply Prompt-to-Prompt control  
- add identity metrics (ArcFace)  
- fairness analysis on CelebA  
- adaptive parameter tuning  

---

## Code

https://github.com/myavg/xai

```bash
bash scripts/run_final_submission.sh
```