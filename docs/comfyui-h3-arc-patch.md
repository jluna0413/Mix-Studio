# ComfyUI Patch — MiniMax H3 quantized text-encoder loading on Intel Arc

## Status
**FIXED & VERIFIED** — H3 video generation works on Intel Arc Pro B70 32GB.

## Files patched
| File | Purpose |
|---|---|
| `comfy/sd1_clip.py` (`load_sd`, ~line 308) | **The real fix.** Correct prefix derivation so quantized weight keys survive the shape filter. |
| `comfy/ops.py` (`cast_bias_weight`, ~line 358) | Defensive `if s.weight is None` guard — turns a hard crash into a recoverable path. |
| `comfy/ops.py` (`forward_comfy_cast_weights`, ~line 1353) | `device=self.weight.device` instead of `device=input.device` in the `weight_only_quant` branch. |

Patched copy kept at `patches/sd1_clip.py.patched`. Re-apply after any ComfyUI upgrade.

## Symptom
```
AttributeError: 'NoneType' object has no attribute 'device'
  comfy/text_encoders/llama.py:554   xq = self.q_proj(hidden_states)
  comfy/ops.py:1414                  forward_comfy_cast_weights(...)
  comfy/ops.py:393                   if offloadable and (device != s.weight.device or ...
```
Preceded by ~466 `Missing weight for layer model.layers.N...` warnings.

## Root cause
`SDClipModel.load_sd()` filters the checkpoint against `self.transformer.state_dict()`:

```python
sd = {k: v for k, v in sd.items() if k in model_sd and v.shape == model_sd[k].shape}
```

A quantized `ops.Linear` is **lazy** — it registers no `weight` parameter until
`_load_quantized_module` builds the `QuantizedTensor`. So `model_sd` contains **none**
of the quantized projection keys, and the filter deleted every single one
(`q_proj`/`k_proj`/`v_proj`/`o_proj`/`gate_proj`/`up_proj`/`down_proj` × 50 layers).
With no weights loaded, `module.weight` stayed `None` and the first forward pass
dereferenced `None.device`.

An earlier attempt at the bypass had an **off-by-one-dot bug**, which is why it
appeared not to work:

```python
# WRONG — prefixes never compared equal
quant_prefixes.add(k[:-len(".comfy_quant")])      # "...q_proj"    (no dot)
prefix = k[:-len("weight")] if k.endswith("weight")  # "...q_proj." (trailing dot!)
```

## The fix
Derive the prefix by stripping `".weight"` / `".bias"` **including the dot**, so it
matches the prefix stored from `".comfy_quant"`:

```python
quant_prefixes = set()
for k in sd.keys():
    if k.endswith(".comfy_quant"):
        quant_prefixes.add(k[:-len(".comfy_quant")])   # no trailing dot

quant_suffixes = (".comfy_quant", ".weight_scale", ".weight_scale_2",
                  ".input_scale", ".pre_quant_scale", "_rotary")

new_sd = {}
for k, v in sd.items():
    if k.endswith(".weight"):
        prefix = k[:-len(".weight")]
    elif k.endswith(".bias"):
        prefix = k[:-len(".bias")]
    else:
        prefix = k
    if k.endswith(quant_suffixes) or prefix in quant_prefixes:
        new_sd[k] = v          # quantized: bypass the shape filter
        continue
    m = model_sd.get(k)
    if m is not None and v.shape == m.shape:
        new_sd[k] = v          # plain: shape-filter as before
```

Quantized keys bypass the filter and are consumed by
`_load_from_state_dict` → `_load_quantized_module`, which builds the `QuantizedTensor`.
Non-quantized keys keep the original shape-filter behaviour (klein T5 compatibility).

## Verification
Offline filter check against the real checkpoint (2054 keys):
```
quantized layer prefixes            : 351
keys bypassing shape filter         : 1503   (previously 0)
projection weights hitting filter   : 0
RESULT: PASS
```

Live run — fresh server, empty queue (`number: 0`, nothing cached):
```
status_str : success
completed  : True
execution_error : none
outputs    : save.images = 124 frames
Missing weight for layer model.layers.* : 0
NoneType ... 'device' errors            : 0
```

## The `visual.*` warnings were NOT benign — second bug, image path

> **Correction.** An earlier version of this document claimed the ~116
> `Missing weight for layer visual.*` warnings were cosmetic. **That was wrong**,
> and the wrong call cost real debugging time. Those warnings were the bug
> announcing itself.

### Symptom
Text-to-video worked. Image-to-video crashed with the *same* error as before:

```
AttributeError: 'NoneType' object has no attribute 'device'   (MiniMaxH3ImageToVideo)
```

### Why the first fix missed it
The first patch bypassed the shape filter for **quantized** layers, keyed off a
`<prefix>.comfy_quant` sibling. But qwen3vl's checkpoint is **mixed precision**:

| Tower | Quantized? | `.comfy_quant` sibling | dtype |
|---|---|---|---|
| Text (`model.layers.*`) | yes (NVFP4) | present | U8 + scales |
| **Vision (`visual.*`)** | **no** | **absent** | **BF16** |

The vision `qkv` is *also* a lazy `ops.Linear` — it registers no `weight` in
`state_dict()` until load populates it — but it has no `comfy_quant` key, so it
got **neither** the quant bypass **nor** a `state_dict` entry:

```python
m = model_sd.get(k)          # None: lazy layer, not enumerated
if m is not None and ...:    # False -> key silently DROPPED
```

`weight` stayed `None`, and the first image forward pass died:

```
qwen35.py:490   self.qkv(x)            # vision attention
ops.py:1338     forward_comfy_cast_weights
ops.py:393      s.weight.device        # s.weight is None
```

**Only the image path runs the vision tower.** That is precisely why t2v passed
and i2v failed on the identical checkpoint — and why "swap the text encoder"
looked like a fix when it never was.

### The fix
Treat *absence* from `state_dict()` as "not enumerated", not as "unwanted".
Shape-filter only the keys the model actually enumerates:

```python
m = model_sd.get(k)
if m is None:
    new_sd[k] = v          # lazy layer (quantized OR plain BF16): keep
elif v.shape == m.shape:
    new_sd[k] = v          # enumerated: original klein T5 mismatch filter
```

`load_state_dict(..., strict=False)` makes genuinely unexpected keys harmless.

### Verification (Arc B70, fresh server, patched loader)

Differential probe — two graphs identical except `first_frame`:

| Case | `first_frame` | Before fix | After fix |
|---|---|---|---|
| A (t2v) | absent | success | success |
| B (i2v) | connected | **`NoneType` crash** | **success** |

Direct load assertion through `comfy.sd.load_clip`:

```
vision blocks                 : 27
blocks with real qkv weight   : 27
blocks still broken           : 0
block 0 qkv.weight            : shape=(3456, 1152) dtype=torch.bfloat16
Missing weight for layer visual.* : 0   (was ~116)
Missing weight for layer model.*  : 0
NoneType ... 'device' errors      : 0
```

The ~116 warnings disappearing is the direct confirmation: those weights are now
loaded instead of dropped.

### Lesson
A warning that only fires on a code path you haven't exercised is not benign —
it is untested. t2v success was used as evidence about a tower t2v never runs.
Regression coverage: `test/comfyui-quantized-clip-filter.test.js` (test 4 fails
against the old logic — verified by reverting).

## Working configuration
```
h3Clip        = qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors   (14.6 GB)
h3ClipVariant = nvfp4-awq
h3Unet        = minimax_h3_fl2va_pruned_int8_convrot.safetensors
h3RefUnet     = minimax_h3_ref2va_pruned_int8_convrot.safetensors
h3VideoVae    = minimax_h3_video_vae_fp16.safetensors
h3AudioVae    = minimax_h3_audio_vae_fp32.safetensors
```

**Note:** `int8-convrot` was previously offered as a clip/text-encoder variant
but has been removed — it required CUDA-only ConvRot kernels and failed on
Intel Arc. The `int8_convrot` **UNet** models (above) are unaffected: ComfyUI
exposes native ConvRot ops on XPU (`int8_tensorwise, convrot_w4a4`).

Launch (env must be sanitized — Mix Studio's venv vars break ComfyUI's PIL):
```bash
env -u PYTHONPATH -u PYTHONHOME -u VIRTUAL_ENV \
  .venv/Scripts/python.exe main.py --port 49000 --default-device 0 \
  --disable-cuda-malloc --force-fp16 --preview-method auto
```

## Pitfalls hit while debugging
1. **Stale server masks the fix.** ComfyUI loads `sd1_clip.py` at startup. A pre-existing
   process on port 49000 kept serving old code; the new instance exited with
   `Port 49000 is already in use` while I read its log and saw misleading zeros.
   Always confirm the PID owning the port, and check `number: 0` + absence of
   `execution_cached` to be sure nothing is cached.
2. **`taskkill` needs MSYS quoting.** `taskkill //F //PID n` fails under git-bash.
   Use `MSYS2_ARG_CONV_EXCL='*' taskkill /F /PID n`.
3. **Match the repro to your settings.** The repro initially used the `int8_convrot`
   clip encoder while settings specified `nvfp4_awq` — different code paths, wrong
   conclusion. *(Since `int8_convrot` clip variant removal, this class of mismatch
   is no longer possible — the variant simply doesn't exist.)*
4. **`visual.*` warnings are a red herring.** Don't chase them; verify against the
   model's `state_dict()` before assuming a load failure.
