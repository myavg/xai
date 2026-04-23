# Explainable Face Editing with Diffusion Models  
### From Black-Box Generation to Interpretable Editing

## Introduction

Diffusion models have recently emerged as one of the most powerful approaches for image generation and editing, producing highly realistic and semantically consistent outputs. However, despite their impressive performance, they suffer from a fundamental limitation:

> **We do not understand how edits are applied internally.**

In face editing, this limitation becomes especially critical. Unlike general image generation, face editing requires strict control over identity and structure. Any unintended modification may distort identity, alter facial geometry, or introduce biases. Therefore, interpretability is not optional — it is essential.

In this project, we extend a standard diffusion-based editing pipeline into an **Explainable AI (XAI) system** and investigate the following research question:

> **How do editing parameters affect internal model behavior and interpretability?**

Our work is inspired by the idea that attention mechanisms inside diffusion models encode meaningful relationships between text and image regions, as demonstrated in *Prompt-to-Prompt Image Editing with Cross Attention Control* (Hertz et al., 2023). Instead of directly modifying attention, we use this insight to interpret model behavior indirectly.

---

## From Generation to Explanation

Our baseline pipeline follows the standard diffusion editing procedure introduced in works such as *Denoising Diffusion Probabilistic Models* (Ho et al., 2020) and *Latent Diffusion Models* (Rombach et al., 2022):

- image → latent encoding  
- noise injection  
- prompt-guided denoising  
- edited image  

In our implementation, we use **Stable Diffusion v1.5** (Rombach et al., 2022), a latent diffusion model that operates in a compressed latent space and is conditioned on text embeddings derived from CLIP (Radford et al., 2021).

From an XAI perspective, we reinterpret the pipeline as:

Text tokens → cross-attention → spatial regions

This abstraction allows us to treat the model not just as a generator, but as a system that distributes influence across the image.

Even though attention maps are not explicitly extracted in our baseline, we interpret the final editing behavior as an **implicit attention distribution**.

As observed in our baseline analysis:
- edits are inherently non-local  
- parameters influence semantic behavior  
- internal mechanisms remain hidden  

Our goal is to uncover and explain these hidden dynamics.

---

## Experimental Setup

To systematically analyze the behavior of the model, we perform controlled experiments.

We consider three types of semantic edits:
- **smile** — modifying facial expression  
- **glasses** — adding a new object  
- **bangs** — modifying hairstyle  

These edits were chosen because they represent different types of transformations: continuous attribute changes versus discrete structural additions.

We evaluate two pipelines:
- **img2img (reference pipeline)** — the standard SDEdit-style approach (Meng et al., 2022)  
- **noise-first pipeline** — a variant where noise is manually injected into the latent representation before denoising  

We perform experiments across **27 parameter configurations**, generating more than **540 images**.

### Metrics (as XAI signals)

| Metric | Meaning | XAI Interpretation |
|------|--------|-------------------|
| CLIP ↑ | prompt alignment | strength of semantic activation |
| LPIPS ↓ | perceptual change | degree of modification |
| SSIM ↑ | structural similarity | identity preservation |

CLIP (Radford et al., 2021) measures how well the generated image matches the text prompt.  
LPIPS (Zhang et al., 2018) captures perceptual differences between images using deep features.  
SSIM (Wang et al., 2004) measures structural similarity at the pixel level.

From an XAI perspective, these metrics act as **proxies for internal model behavior**, allowing us to infer how the model balances semantic alignment and preservation.

---

## Core Hypothesis (XAI View)

We model diffusion editing as a trade-off:

> **Editability ↑ vs Preservation ↑**

This trade-off is not merely empirical — it is grounded in diffusion theory (Ho et al., 2020).

The forward diffusion process can be written as:

`z_t = sqrt(alpha) * z_0 + sqrt(1 - alpha) * epsilon`

This equation shows that the noisy latent representation is a combination of:
- the original signal  
- Gaussian noise  

As noise increases:
- the contribution of the original image decreases  
- the model relies more on the text prompt  

This leads to a fundamental consequence:

- high noise → weak identity constraint → strong edits  
- low noise → strong identity constraint → limited edits  

From an interpretability perspective, this implies:

> **increasing noise leads to a broader and less localized distribution of influence across the image**

We interpret this phenomenon as **attention spreading globally**, even without explicitly observing attention maps.

---

## Baseline Observations

### Editing vs Reconstruction

![baseline](pictures/baseline_input_vs_edit.jpg)
![reconstruction](pictures/sanity_input_vs_recon.jpg)

### Result

- **s = 0.6** → visible edits but identity distortion  
- **s = 0.2** → near-perfect reconstruction  

These results confirm that noise strength directly controls how much the model modifies the image.

### XAI Interpretation

Even simple edits affect multiple aspects of the image:
- facial geometry  
- background  
- illumination  

This demonstrates that editing is inherently **non-local**.

Rather than modifying a single region, the model redistributes information across the entire image.

This behavior aligns with findings from diffusion-based editing literature (Meng et al., 2022; Hertz et al., 2023), where attention and latent interactions operate globally.

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

As noise increases:
- CLIP increases, indicating stronger alignment with the prompt  
- LPIPS increases, indicating stronger deviation from the input  

This confirms the existence of a fundamental trade-off.

### XAI Conclusion

> Stronger edits correspond to **broader attention spread**

The scatter plots provide a **global view of the model's behavior**, summarizing how different parameter configurations affect the balance between alignment and preservation.

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

Different edits require different levels of intervention:

- **Smile and bangs** correspond to continuous changes and require only small perturbations  
- **Glasses** require the introduction of a new object and therefore need stronger noise  

### XAI Insight

This suggests that the model learns **implicit semantic regions** within the latent space:
- mouth region for expressions  
- hair region for hairstyle  
- eye region for accessories  

These regions are not explicitly defined but emerge from the interaction between text conditioning and latent structure.

---

## Parameter Sweep = Behavior Visualization

![smile](pictures/sweep_grid_smile.png)
![glasses](pictures/sweep_grid_glasses.png)
![bangs](pictures/sweep_grid_bangs.png)

### Observations

- low noise → weak and localized edits  
- medium noise → optimal balance  
- high noise → global transformation  

### XAI Insight

Interpretability is maximized at **moderate noise levels**, where:
- the model retains sufficient information from the input  
- while still being flexible enough to apply meaningful edits  

This aligns with the theoretical understanding of diffusion processes and their signal-to-noise dynamics.

---

## Pipeline Comparison

![noise](pictures/best_overview_noise.png)

### Result

Both pipelines produce similar qualitative and quantitative behavior.

### XAI Conclusion

> Interpretability is determined by the **latent space and noise level**, not by the specific pipeline implementation.

This reinforces the idea that the observed trade-offs are intrinsic to the diffusion model itself.

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

Failures are not random.

They correspond to **systematic misalignment between the model’s internal attention distribution and the intended edit**.

This perspective connects failure modes directly to interpretability.

---

## Why This Matters for XAI

Our experiments demonstrate that:

1. Diffusion models operate in a **non-local manner**, making precise control challenging  
2. Noise level acts as a **control parameter for interpretability**  
3. Standard metrics can be reinterpreted as **explanatory signals**  

This shifts the perspective from evaluation to understanding.

---

## Final Results (Clear Summary)

### Quantitative

- Best trade-off score ≈ **0.196**
- Optimal noise:
  - **0.3** for continuous edits  
  - **0.5** for structural edits  
- LPIPS increases by approximately **2×** between low and high noise  

### Qualitative

- Moderate noise produces visible and meaningful edits  
- Identity is largely preserved  
- Model behavior remains interpretable  

---

## Final Conclusion

We transformed a generative diffusion pipeline into an interpretable system.

### Main findings:

1. Noise strength is the primary factor controlling interpretability  
2. Editing behavior reflects an implicit attention distribution  
3. Different prompts activate different semantic regions  
4. Failure modes correspond to attention misalignment  

---

## XAI Takeaway

> Diffusion models are not purely black-box systems.  
> Their behavior follows structured patterns that can be understood through noise and attention dynamics.

---

## References

- Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*.  
- Rombach, R. et al. (2022). *High-Resolution Image Synthesis with Latent Diffusion Models*.  
- Meng, C. et al. (2022). *SDEdit: Guided Image Synthesis and Editing*.  
- Hertz, A. et al. (2023). *Prompt-to-Prompt Image Editing with Cross Attention Control*.  
- Mokady, R. et al. (2023). *Null-text Inversion for Editing Real Images*.  
- Brooks, T. et al. (2023). *InstructPix2Pix*.  
- Radford, A. et al. (2021). *CLIP*.  
- Zhang, R. et al. (2018). *LPIPS*.  
- Wang, Z. et al. (2004). *SSIM*.  

---

## Code

https://github.com/myavg/xai

```bash
bash scripts/run_final_submission.sh
