# Intel Arc / XPU compatibility (fork notes)

This fork (`jluna0413/Mix-Studio`, branch `intel-arc-xpu`) tracks Intel Arc / Intel XPU compatibility work on top of Mix Studio upstream (`BlackMixture/Mix-Studio`). It is **experimental**: behavior depends on your ComfyUI build, Intel GPU drivers, PyTorch/XPU stack, installed custom nodes, and model variants. Validate on your own hardware before relying on it for production work.

**Status at this commit:** `node --check server.js` clean; `node --test` **1584/1584 passed** (Windows). These checks validate app/server logic and graph/tests, not end-to-end GPU generation quality on every Arc system.

## What this fork focuses on

- **MiniMax H3 on Intel Arc/XPU (ComfyUI workflow path):** graph/wiring fixes so H3 text/frame/reference flows can run on XPU-oriented ComfyUI setups where NVIDIA-only assumptions break (device/loader choices, capability probing, attention backend selection, and CLIP/UNet compatibility).
- **H3 CLIP safety:** prevents the lightweight-CLIP fallback from loading a 4B encoder without its required projection/adapter path. A mismatched 4B/non-32B encoder produces hidden states with the wrong dimension for the H3 DiT and fails at sampling with an `addmm`/matmul shape mismatch (e.g. `mat1 and mat2 shapes cannot be multiplied ... 5120x5376`). When a projection/adapter loader is unavailable, the graph must use a compatible 32B H3 text encoder.
- **Splat/orbit graph correctness:** probes `UNETLoader.weight_dtype` from live `/object_info` for the H3 INT8/orbit path and removes invalid/undefined RTX-upscaler dependencies from the splat orbit graph so it can be built/queued consistently with the installed ComfyUI.
- **Stability/recovery:** detects Intel Level Zero GPU device-lost errors (`UR_RESULT_ERROR_DEVICE_LOST` / device lost), surfaces a clear failure, and reaps stranded jobs when ComfyUI crashes/restarts without history so setup/model/dependency operations are not permanently blocked by a dead queue entry.
- **Intel Arc-oriented workflow fixes carried in this line:** XPU device pinning/resident handling for H3 text-encoder paths where applicable, SeedVR2 batch sizing fixes for 32 GB Arc cards, SDPA/Triton attention path (SageAttention gated off where not usable on Arc), and related UI/workflow/test updates merged/restored on top of upstream v1.3.0-era work.

See also: [`docs/comfyui-h3-arc-patch.md`](comfyui-h3-arc-patch.md) for the ComfyUI-side quantized CLIP/vision loader patch required for reliable H3 (especially image-to-video/vision tower) on Arc in affected ComfyUI builds.

## Setup notes (Intel Arc / XPU)

1. **Use a ComfyUI environment that actually supports Intel XPU** (Intel Extension for PyTorch / XPU-enabled PyTorch and a ComfyUI build known to run on Arc). Mix Studio builds and submits graphs; it does not replace your ComfyUI/GPU runtime.
2. **Point Mix Studio at your running ComfyUI** (correct URL/port, model paths). Verify node/model availability in ComfyUI (`/object_info`) and in Mix Studio’s setup/dependency checks.
3. **Apply the ComfyUI H3 Arc patch if needed:** [`docs/comfyui-h3-arc-patch.md`](comfyui-h3-arc-patch.md) documents the `sd1_clip.py` / loader fixes for mixed-precision quantized Qwen3-VL text encoders on Arc (text tower quant bypass + vision tower lazy-weight handling). **Re-apply after ComfyUI upgrades.** Confirm with a fresh ComfyUI process (no cached execution) and check for `visual.*`/`model.*` missing-weight warnings and `NoneType ... device` errors.
4. **Model/encoder choices:** use H3 model/CLIP variants that are compatible with your ComfyUI XPU stack. The curated upstream NVIDIA-specific quant/kernel routes (e.g. NVFP4/AWQ/INT8 ConvRot assumptions tied to CUDA kernels in some paths) may not all be available or optimal on Arc; prefer variants your installed nodes actually support and that Mix Studio reports as compatible. If a lightweight/distilled encoder path requires `ClipProjLoader`/`MiniMaxH3AdapterLoader` and the corresponding projection/adapter/student files, install them explicitly; otherwise use a compatible 32B H3 CLIP.
5. **Attention:** on Arc/XPU, prefer the attention backends Mix Studio/ComfyUI report as actually working (often PyTorch SDPA / native `ModelAttentionBackend`; Triton only if your stack supports it). CUDA-only SageAttention/SLA/Comfy Kitchen paths may be unavailable or unprobed — trust runtime capability checks, not hardware marketing claims.
6. **Sanitize ComfyUI launch environment when needed:** stray venv/Python env vars from other tools can break imports in ComfyUI’s environment (see notes in `comfyui-h3-arc-patch.md`). Use ComfyUI’s own Python/venv for its process.

## Known bugs / limitations (current)

- **End-to-end H3/3D generation can be very slow on Intel Arc.** Long runtimes (including multi-hour orbit/splat runs) have been observed in practice; performance depends heavily on model choice, resolution/frames/steps, attention backend, drivers, and ComfyUI XPU optimizations.
- **GPU device loss / driver instability:** Intel Level Zero may report `UR_RESULT_ERROR_DEVICE_LOST` and crash the ComfyUI worker mid-generation. Mix Studio now detects this and cleans up stranded jobs, but you may need to restart ComfyUI and retry. System TDR/driver settings and workload size affect stability.
- **ComfyUI patch dependency:** affected ComfyUI builds need the `sd1_clip.py` quantized/mixed-precision CLIP loading fix for H3 on Arc; upgrades can overwrite it. Symptoms include `NoneType object has no attribute 'device'` in CLIP/vision forward passes and many `Missing weight for layer ...` warnings.
- **Lightweight/distilled H3 text encoders are conditional:** `ClipProjLoader` / adapter-based distilled paths only work when the required nodes and model files exist and match the expected interface. Without them, use a compatible 32B H3 CLIP (this fork guards against the 4B-without-projection fallback that caused shape mismatches).
- **Attention feature parity is not guaranteed on XPU:** SageAttention, SLA, and Comfy Kitchen acceleration paths are not universally available on Intel Arc. The app falls back to native/SDPA paths where appropriate, but speed/quality characteristics differ from CUDA setups.
- **3D Gaussian splat quality is experimental:** orbit-video-to-splat reconstruction can produce floaters, background distortion, or inconsistent geometry (especially for unseen regions). This is inherent to the pipeline; see [`docs/gaussian-splats.md`](gaussian-splats.md).
- **Upstream NVIDIA-specific curated routes remain hardware-dependent:** some workflows/weights (e.g. certain LTX 2.5 / H3 quantized routes) are designed around NVIDIA kernels in upstream guidance. Intel Arc support in this fork is best-effort and may require alternative compatible weights/nodes.
- **Not integrated:** vLLM-Omni serving features (e.g. SPARGE_ATTN, USP, tensor-parallel XPU serving) target a different runtime than this ComfyUI graph workflow and are outside the current Mix Studio/ComfyUI integration.

## Reporting issues (for this fork)

When reporting an Intel Arc issue, include: GPU/driver version, ComfyUI commit/version and Python/XPU stack, exact Mix Studio commit/branch, the workflow/model/encoder/attention settings, whether `docs/comfyui-h3-arc-patch.md` is applied, and the full ComfyUI/Mix Studio error text (especially `addmm` shape mismatches, `UR_RESULT_ERROR_DEVICE_LOST`, or missing-node/model messages). Reproduce with a fresh ComfyUI process when investigating loader/cache issues.

## Upstream alignment / planning

This fork is based on upstream Mix Studio work through the merged v1.3.0-era line plus local Intel fixes. Upstream planning/status changes over time; check [`ROADMAP.md`](../ROADMAP.md), [`CHANGELOG.md`](../CHANGELOG.md), releases, and open PRs/issues in [`BlackMixture/Mix-Studio`](https://github.com/BlackMixture/Mix-Studio) before proposing or rebasing large changes.
