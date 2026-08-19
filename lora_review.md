# H3 LoRA Feasibility Review — 2026-08-19

## Methods
- HuggingFace API metadata (downloads, likes, file listings)
- HTTP Range requests for actual `.safetensors`/`.gguf` file sizes
- Cross-referenced against installed ComfyUI custom nodes
  (`ComfyUI-H3-Motion-Context`, `ComfyUI-MiniMax-H3-Turbo`)
- Cross-referenced against Arc B70 VRAM budget (31.16 GB)
- Cross-referenced against working encoder paths (NVFP4, lightweight, distilled)

## VRAM Context

| Encoder | VRAM | Status |
|---|---|---|
| NVFP4 AWQ (32B) | 14.61 GB | ✅ Working (your default) |
| Lightweight 4B + ClipProj | 2.65 GB | ✅ Installed, available in dropdown |
| Distilled CondBridge | 3.8 GB | ✅ Installed, available in dropdown |

With the NVFP4 encoder (14.61 GB) + DiT (~13 GB) + VAE (~0.5 GB) = ~28.1 GB, leaving ~3 GB headroom. A small LoRA adds 0.5-1 GB. A large LoRA (1.2 GB+ loaded onto the DiT) eats into that headroom.

---

## LoRA-by-LoRA Analysis

### 1. `fal/MiniMax-H3-Realism-People-LoRA`
- **File**: `h3-realism-people-t2v-i2v-r2v.safetensors` — **125 MB**
- **Downloads**: 22,358 | **Likes**: 261
- **Type**: T2V/I2V/R2V LoRA for human realism
- **Verdict**: ✅ **Feasible, low risk.**
- 125 MB adds ~0.5 GB VRAM (at rank ~32). Standard `LoraLoader` on the H3 UNet. No node changes needed. Most proven option here (22K downloads).

### 2. `Jojocodex/minimax-h3-spatial-physics-lora`
- **Key files**:
  - `wushu_spatial_physics_v2_3000.safetensors` — 296 MB
  - `wushu_spatial_physics_v2_1000_pruned.safetensors` — **148 MB**
  - `wushu_spatial_physics_v2_1000_full.safetensors` — 296 MB
- **Downloads**: 1,351 | **Likes**: 55
- **Type**: Movement/physics LoRA (wushu spatial reasoning)
- **Verdict**: ✅ **Feasible.**
- Use the `_pruned` variant (148 MB, ~0.5 GB) to stay conservative. Standard LoraLoader workflow.

### 3. `Jojocodex/minimax-h3-wushu-action-lora`
- **Key files**:
  - `wushu_action_h3_lora_v4_2000_pruned.safetensors` — **148 MB**
  - `KungFu_MiniMaxH3_LoRA_step4500.safetensors` — 148 MB
  - `wushu_action_h3_lora_v4_2000_full.safetensors` — 296 MB
- **Downloads**: 684 | **Likes**: 27
- **Type**: Action/movement LoRA
- **Verdict**: ✅ **Feasible.**
- Same creator as #2 (Jojocodex), sibling LoRA. Use the 148 MB pruned variant.

### 4. `lightx2v/MiniMax-H3-Prompt-Rewriter-LoRA`
- **File**: `adapter_model.safetensors` — **3.56 GB**
- **Downloads**: 911 | **Likes**: 168
- **Type**: Text-encoder LoRA (Qwen3-based)
- **Verdict**: ⚠️ **Conditional.**
- 3.56 GB adapter on the text encoder. With NVFP4 encoder (14.61 GB), total = ~18 GB — within your 31.16 GB budget but eats into DiT/VAE headroom. **Safer with the lightweight encoder**: 2.65 + 3.56 = ~6.2 GB, leaving ~25 GB for DiT/VAE/LoRA.
- Requires loading onto the CLIP text encoder, not the DiT. Use a `LoraLoader` targeting the H3 CLIPLoader node.
- The `pytraveler` variant (GGUF, Q8_0, 1.89 GB) is lighter — would be 2.65 + 1.89 = ~4.5 GB with lightweight.

### 5. `DiffSynth-Studio/MiniMax-H3-LoRA-LineartAnime`
- **File**: `model.safetensors` — **1.2 GB**
- **Downloads**: 0 | **Likes**: 12
- **Type**: Stylized LoRA (lineart anime)
- **Verdict**: ⚠️ **Use with lightweight encoder.**
- 1.2 GB onto the DiT. With NVFP4: 14.61 + 13 + 1.2 + 0.5 = ~29.3 GB — too close to your 31.16 GB ceiling for comfort with batch > 1.
- **Better approach**: Switch the dropdown to **Lightweight** (2.65 GB encoder), then: 2.65 + 13 + 1.2 + 0.5 = ~17.4 GB. Comfortably within budget.
- ⚠️ 0 downloads — untested in the wild. Structurally sound (DiffSynth-Studio is reputable) but no community validation.

### 6. `Abiray/MiniMax-H3-Turbo-Lora-Pruned-ComfyUI`
- **Files**: All 4 checkpoints are **592 MB** (`.safetensors`)
- **Downloads**: 16,844 | **Likes**: 44
- **Type**: Speed LoRA (4-step checkpoints, turbo sampling)
- **Verdict**: ✅ **Feasible.**
- This is a pruned checkpoint — loads as a LoRA but replaces ~592 MB of DiT weights. VRAM impact is minimal (~0.3-0.5 GB because it's a weight merge, not additive params).
- **Trade-off**: 4-8 sampling steps instead of 20+. Use for quick iterations, not final renders.
- Any of the 4 variants works — `minimax_h3_turbo_4step_ckpt600_ema_V4.safetensors` is recommended (EMA = exponentially averaged, generally cleaner).

### 7. `StellarVoyager/H3-IR-Qwen3.6-27B-LoRA`
- **File**: `adapter_model.safetensors` — 223 MB
- **Downloads**: 19 | **Likes**: 2
- **Type**: LoRA for `H3-IR-Qwen3.6-27B` (Image Regeneration pipeline)
- **Verdict**: ❌ **Not compatible.**
- This targets the **IR (Image Regeneration)** variant of H3, which uses Qwen3.6-27B as its image encoder. Your workflow uses MiniMax-H3 (the 32B NVFP4 encoder at ~14.6 GB). The architectures don't match — you cannot load an IR-Qwen LoRA onto the 32B text encoder and expect it to work.
- It will either error on load or silently produce garbage.

---

## Summary Table

| LoRA | Size | VRAM impact | Feasible? | How to use |
|---|---|---|---|---|
| fal — Realism People | 125 MB | ~0.5 GB | ✅ Yes | Directly on NVFP4 path |
| Jojocodex — Spatial Physics | 148 MB (pruned) | ~0.5 GB | ✅ Yes | Load pruned variant |
| Jojocodex — Wushu Action | 148 MB (pruned) | ~0.5 GB | ✅ Yes | Load pruned variant |
| lightx2v — Prompt Rewriter | 3.56 GB | ~3.6 GB | ⚠️ Conditional | Lightweight encoder preferred |
| DiffSynth — Lineart Anime | 1.2 GB | ~1.2 GB | ⚠️ Tight on NVFP4 | **Switch to lightweight encoder** |
| Abiray — Turbo (4-step) | 592 MB | ~0.3-0.5 GB | ✅ Yes | Any variant, 4-step sampling |
| StellarVoyager — IR Qwen3.6 | 223 MB | N/A | ❌ No | Wrong architecture |

## Recommendation
Start with **fal — Realism People** (125 MB, 22K downloads). It's the safest, proven LoRA that adds negligible VRAM. The **Wushu LoRAs** are also safe at 148 MB each. The **Turbo LoRA** is worth trying for fast iteration (4-6 steps). For the **Lineart Anime** LoRA, switch the encoder dropdown to **Lightweight** first. Skip the **Prompt Rewriter** unless you specifically want text-encoder tweaks. Avoid the **IR Qwen3.6** LoRA — it targets the wrong model variant entirely.
