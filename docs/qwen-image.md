# Qwen Image Integration Guide — Mix Studio

Qwen Image is a 20B-parameter MMDiT text-to-image and image-editing foundation model developed by QwenLM (Alibaba). It is available in Mix Studio as a native generation engine alongside Krea 2, with full local execution on compatible hardware including Intel Arc B70 (32 GB).

This guide covers:

1. Overview of the Qwen Image T2I integration in Mix Studio
2. Usage guide — engine selection and the FP8 Lightning / bf16 toggle
3. VRAM requirements table (FP8 ~19 GB fits 32 GB Arc B70; bf16 ~38 GB needs offloading)
4. Canvas size cheat sheet (256–2560 px, step 16)
5. Troubleshooting — auth errors, VRAM OOM, model loading delays, FP8 quality issues

Qwen Image is released under the Apache 2.0 license. Model weights are on Hugging Face (Qwen/Qwen-Image, Qwen/Qwen-Image-Edit-2511) and ModelScope.

---

## 1. Overview of Qwen Image T2I Integration in Mix Studio

Mix Studio exposes Qwen Image as a first-class generation engine. The integration wraps the Qwen Image diffusers pipeline (and the Qwen-Image-Edit pipeline for image-to-image tasks) and presents it through the same generation panel used for other engines, so the user can switch between Qwen Image and Krea 2 without changing workflow.

Key capabilities exposed by the integration:

- **Text-to-image generation** — Qwen Image generates images directly from a text prompt. It is particularly strong at complex text rendering (especially Chinese characters, multi-line paragraphs, and mixed-script typography) and realistic detail.
- **Image editing** — when Mix Studio supports image-to-image editing, Qwen-Image-Edit (single-image and multi-image inputs) is wired as a separate pipeline. Edit mode requires an input image upload plus a prompt; it is not a text-only generation.
- **Bilingual prompts** — Qwen Image accepts English and Chinese prompts natively. No engine switching or model change is needed when the user writes in either language.
- **LoRA support** — Qwen Image uses a Qwen-native LoRA format. Lightning LoRAs (4-step and 8-step) are the primary examples and reduce the denoising step count from ~50 to 4 or 8. Community style/character LoRAs follow the same format and must live in Mix Studio's Qwen LoRA directory, not the generic ComfyUI LoRA directory.
- **Precision modes** — Mix Studio exposes bf16 (reference quality, ~38 GB VRAM) and FP8 Lightning (4-step and 8-step, ~19 GB VRAM). The precision toggle changes both the model precision and, when a Lightning LoRA is active, the sampler step count.

### Integration topology

- Base model: loaded in bf16 from the Hugging Face repo configured in Mix Studio's model registry.
- Lightning LoRA: applied at runtime on top of the bf16 base model; not a standalone checkpoint.
- Edit pipeline: separate from T2I; requires an input image + prompt.
- Model registry: Mix Studio stores the repo tag for qwen-image-3.0-pro and qwen-image-3.0. Verify the tag matches the latest Qwen release at integration time (e.g. Qwen-Image-2512 for T2I, Qwen-Image-Edit-2511 for editing).

### Licensing

Qwen Image is Apache 2.0. Generated images are owned by the user. This differs from Krea 2, which is a proprietary cloud/API product governed by Krea's terms. The Apache 2.0 badge in Mix Studio's model selector signals this to users who need permissive licensing for commercial work.

---

## 2. Usage Guide

### 2.1 Selecting the Qwen Image engine

In the Mix Studio generation panel, the engine selector presents the available Qwen tiers:

- **Qwen Image 3.0 Pro** — highest quality. Use for final renders, print-ready output, complex text scenes, and Chinese typography. Runs at full step count (~50 steps) at bf16 unless a Lightning LoRA is attached.
- **Qwen Image 3.0** — standard / faster tier. Use for iteration, concept exploration, batch generation, and real-time preview. Slightly less fine detail than pro.

Default recommendations:

- On Intel Arc B70 (32 GB): default to **Qwen Image 3.0 + FP8 Lightning 8-step** — good balance of speed and quality, fits in VRAM without offloading.
- On GPUs with >= 38 GB VRAM: default to **Qwen Image 3.0 Pro + bf16** — reference quality, no offloading.

When comparing with Krea 2, keep in mind the architectural difference: Qwen Image is a native open-weight MMDiT model that runs locally; Krea 2 is a proprietary cloud pipeline. The choice is not just about quality — it is also about local vs cloud, LoRA support, and licensing.

### 2.2 Switching between FP8 Lightning and bf16

The precision toggle in the generation panel controls two things at once:

1. The base model precision (bf16 vs FP8).
2. When a Lightning LoRA is active, the KSampler step count.

| Precision | VRAM | Quality | Speed | When to use |
|---|---|---|---|---|
| bf16 (BFloat16) | ~38 GB | Reference / full quality — no quantization loss | Standard, full denoising schedule (~50 steps) | Final-quality output; GPUs with >= 38 GB VRAM, or when quality is non-negotiable |
| FP8 Lightning 8-step | ~19 GB | Better than 4-step; closer to bf16 | Fast, 8 NFEs | Batch work where quality matters more than raw speed; default on 32 GB cards |
| FP8 Lightning 4-step | ~19 GB | Good, some fine-detail loss vs bf16 | Very fast, 4 NFEs | Previews, iteration, real-time preview, tight iteration budgets |

Switching behavior details:

- **FP8 Lightning is a LoRA on the bf16 base.** Mix Studio loads the base model in bf16, then attaches the Lightning LoRA (4-step or 8-step) and sets KSampler steps to 4 or 8 respectively. It is not a standalone checkpoint.
- **Switching from FP8 back to bf16:** Mix Studio unloads the Lightning LoRA and restores KSampler steps to the base value (around 50).
- **Switching precision on a 32 GB card:** selecting bf16 on Intel Arc B70 triggers blocks_to_swap offloading. Mix Studio should show a hardware notice and expect 2–4x slower inference.
- **Lightning LoRA files:** published by LightX2V / ModelTC under the Qwen-Image-Lightning repository. The LoRA file version must match the base model version.

### 2.3 Prompt language and prompt enhancement

Qwen Image accepts English and Chinese prompts natively. If Mix Studio exposes a prompt enhancement tool, it should work for both languages. A small badge in the prompt box indicating bilingual support helps users know they can write in either language. For editing tasks, applying Qwen's official prompt rewriting / enhancement improves stability.

### 2.4 Edit mode vs T2I mode

- **T2I mode:** text prompt only. Generate from scratch.
- **Edit mode:** input image upload + prompt. Used for object replacement, style transfer, identity-preserving edits, and layout changes. If Mix Studio supports editing, wire Qwen-Image-Edit separately from the T2I pipeline.

---

## 3. VRAM Requirements

The Intel Arc B70 ships with 32 GB of VRAM. Qwen Image's VRAM footprint differs significantly by precision mode.

| Precision | VRAM needed | Fits on B70 32 GB? | Configuration |
|---|---|---|---|
| bf16 (base model, no LoRA) | ~38 GB | **No — exceeds 32 GB** | Requires blocks_to_swap / layer offloading. Slower inference; potential thrash with large canvas. Use only when quality is non-negotiable. |
| FP8 Lightning 4-step | ~19 GB | **Yes — fits comfortably** | No offloading. Full-speed inference on B70. Recommended for interactive / preview workflows. |
| FP8 Lightning 8-step | ~19 GB | **Yes — fits comfortably** | No offloading. Good balance of speed and quality on B70. |

### Canvas size interacts with VRAM

- On B70 with bf16 and offloading active: a safe maximum is around **2048 × 2048**.
- On B70 with FP8 Lightning: up to **2560 × 2560** is feasible without offloading.

### Offloading notice

When the user selects bf16 on a 32 GB GPU (e.g. Intel Arc B70), Mix Studio should show a tooltip that offloading is active and generation will be slower (roughly 2–4x vs a GPU with sufficient native VRAM).

---

## 4. Canvas Size Cheat Sheet

Qwen Image canvas constraints:

- **Min:** 256 × 256 px
- **Max:** 2560 × 2560 px
- **Step:** 16 px — width and height must be multiples of 16
- **Soft ceiling:** ~6.5M total pixels (hard limit is the 2560 × 2560 max, but very large canvases near the edge may hit memory limits depending on precision mode)

### Common presets (step-16 valid, within bounds)

| Use case | Width × Height | Aspect ratio | Notes |
|---|---|---|---|
| Square (standard) | 1024 × 1024 | 1:1 | Common default |
| Square (native Qwen) | 1328 × 1328 | 1:1 | Native aspect used in Qwen examples |
| Wide (16:9) | 1664 × 928 | 16:9 | Step-16 valid |
| Tall (9:16) | 928 × 1664 | 9:16 | Step-16 valid |
| 4:3 | 1472 × 1104 | 4:3 | Step-16 valid |
| 3:4 | 1104 × 1472 | 3:4 | Step-16 valid |
| 3:2 | 1584 × 1056 | 3:2 | Step-16 valid |
| 2:3 | 1056 × 1584 | 2:3 | Step-16 valid |
| Max square | 2560 × 2560 | 1:1 | Hard maximum; use FP8 on 32 GB cards |

### Mix Studio UI guidance

- When a user types a custom width/height, round to the nearest multiple of 16 within [256, 2560].
- Show a warning if the chosen canvas approaches the memory limit for the selected precision (e.g. above ~2048 × 2048 for bf16 on a 32 GB card, or above ~6.5M total pixels as a soft guideline).

---

## 5. Troubleshooting

### 5.1 Authentication / access errors

Symptom: generation fails with a Hugging Face access error, model download failure, or 401/403 response.

Common causes:

- Missing or invalid Hugging Face token in Mix Studio settings.
- The selected model repo requires access approval (rare for Qwen Image — it is public, but verify the exact repo and tag).
- Rate limiting on Hugging Face Hub downloads.

Resolution steps:

1. Verify the Hugging Face token is configured in Mix Studio settings and is valid (test with `huggingface-cli whoami` if possible).
2. Confirm the model repo is public: Qwen/Qwen-Image, Qwen/Qwen-Image-Edit-2511, or the equivalent tag used in Mix Studio's model registry.
3. If the model cache is corrupt, clear the local cache for the affected repo and re-download.
4. For Lightning LoRA files, confirm the correct repo path (LightX2V / ModelTC Qwen-Image-Lightning) and that the LoRA file version matches the base model version.

### 5.2 VRAM OOM (out of memory)

Symptom: generation fails during model load or first inference step with an OOM error.

Common causes:

- bf16 selected on a GPU with less than ~38 GB VRAM (e.g. Intel Arc B70 at 32 GB) without offloading configured.
- Canvas too large for the selected precision and available VRAM.
- Lightning LoRA loaded incorrectly, or alongside other memory-heavy nodes.

Resolution steps:

1. Switch from bf16 to FP8 Lightning (8-step recommended, 4-step for the tightest memory budget). This drops VRAM from ~38 GB to ~19 GB.
2. Reduce canvas size to a safe preset (e.g. 1024 × 1024 or smaller).
3. On B70 with bf16, ensure blocks_to_swap offloading is enabled and the swap factor is set appropriately for the Arc B70's memory bandwidth.
4. If using other ComfyUI nodes alongside Qwen, reduce their memory footprint or run Qwen Image in isolation.
5. Restart the generation session to clear any residual CUDA/SYCL memory state.

### 5.3 Model loading delays

Symptom: generation starts slowly, or the UI appears stuck during model load.

Common causes:

- First run after install: model weights and LoRA files are being downloaded and cached.
- bf16 on a 32 GB card: offloading adds overhead to model load and each inference step.
- Hugging Face Hub download speed or rate limiting.
- Multi-GPU setup with queue contention.

Resolution steps:

1. Pre-cache the base model and Lightning LoRA files before the first generation. Download with `huggingface-cli download` into the Mix Studio model directory.
2. For bf16 on B70, expect slower load and inference due to offloading; show a progress indicator so the user knows loading is intentional.
3. If the delay is a one-time download, let it complete — subsequent runs will be faster from cache.
4. On multi-GPU systems, verify NUM_GPUS_TO_USE and TASK_QUEUE_SIZE are appropriate for the hardware.
5. If the model fails to load entirely, check that the model registry entry points to the correct HF repo and tag, and that the local files (UNet, CLIP, VAE) are present in the expected ComfyUI subfolders.

### 5.4 FP8 Lightning quality too low

Symptom: output quality is noticeably worse than bf16.

Common causes:

- Using the 4-step Lightning LoRA where 8-step would be more appropriate.
- Canvas at the maximum size (2560 × 2560) with 4-step; the reduced NFE budget is stretched thin.
- Lightning LoRA version mismatch with the base model.

Resolution steps:

1. Switch from 4-step to 8-step Lightning LoRA.
2. Reduce canvas size to a moderate preset (e.g. 1024 × 1024 or 1328 × 1328).
3. Confirm the Lightning LoRA file version matches the base model version.
4. If quality is still insufficient, switch to bf16 (requires sufficient VRAM or offloading on 32 GB cards).

### 5.5 Edit mode not behaving as expected

Symptom: edit output does not preserve identity, or the instruction is only partially followed.

Common causes:

- Prompt rewriting not applied (Qwen's official prompt enhancement for editing improves stability).
- Using the older Qwen-Image-Edit (single-image-only) when Qwen-Image-Edit-2511 (multi-image, better consistency) is available.
- LoRA or base model version mismatch.

Resolution steps:

1. Apply prompt rewriting / prompt enhancement for edit tasks (Qwen provides official prompt utilities for editing).
2. Use Qwen-Image-Edit-2511 for multi-image and better identity consistency.
3. Confirm the base model and any LoRA versions align.

---

## References

- Qwen Image Technical Report — arXiv:2508.02324 (https://arxiv.org/abs/2508.02324)
- Qwen Image GitHub — https://github.com/QwenLM/Qwen-Image
- Qwen Image Lightning (LightX2V / ModelTC) — https://github.com/ModelTC/Qwen-Image-Lightning
- Qwen Image Hugging Face (T2I) — https://huggingface.co/Qwen/Qwen-Image
- Qwen Image Edit Hugging Face — https://huggingface.co/Qwen/Qwen-Image-Edit-2511
- ComfyUI Qwen Image tutorial — https://docs.comfy.org/tutorials/image/qwen/qwen-image
- DiffSynth-Studio (low-VRAM offload, FP8, LoRA) — https://github.com/modelscope/DiffSynth-Studio
