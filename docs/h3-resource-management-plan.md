# H3 Workflow Resource Management — Diagnosis & Plan

**Status:** DIAGNOSIS ONLY. No code changed. Awaiting approval.
**Date:** 2026-08-18
**Trigger:** Machine hard-crashed running H3 image-to-video on Intel Arc Pro B70 (32 GB).

---

## 1. What happened, in order

| # | Event | Evidence |
|---|---|---|
| 1 | H3 i2v failed with `'NoneType' object has no attribute 'device'` | job `20fba4f3` |
| 2 | Root cause found: vision-tower weights dropped by `sd1_clip.py` filter | differential probe, 27/27 blocks |
| 3 | Fixed (`c25f644`); probe case B went crash → success | live verification |
| 4 | User submitted real job → got **past** H3 conditioning, died in sampler | `SamplerCustomAdvanced -> RuntimeError: level_zero ... error 20 (UR_RESULT_ERROR_DEVICE_LOST)` |
| 5 | ComfyUI process died; then the **machine crashed** | port 49000 dead; reboot 15:55:39 |

**The fix worked.** The failure moved from model loading to the sampler — a
different, deeper stage. But it exposed a resource problem.

### Uncomfortable but important: my fix increased VRAM pressure

Before `c25f644`, the loader silently dropped **351 `visual.*` tensors** — they
were never allocated. Now they load correctly. So the job progresses further and
allocates more. The crash is not a regression in correctness; it is the workload
finally running for real against a budget nobody was enforcing.

---

## 2. The core resource problem

Models the H3 i2v graph loads, measured on disk:

| Model | Size |
|---|---|
| `minimax_h3_fl2va_pruned_int8_convrot` (DiT/UNET) | 19.53 GB |
| `qwen3vl_32b_minimax_h3_nvfp4_awq` (text+vision CLIP) | 14.60 GB |
| `minimax_h3_video_vae_fp16` | 4.85 GB |
| `minimax_h3_audio_vae_fp32` | 0.56 GB |
| `minimax_h3_turbo_v4_step600_ema` (LoRA) | 0.72 GB |
| **Total** | **40.28 GB** |

**Against 32 GB of VRAM** — before counting activations, the 124-frame latent,
attention workspace, or fragmentation. Sequential CPU offload is enabled for
Intel, which is why anything runs at all, but nothing checks whether the job
*can* fit before submitting it.

`UR_RESULT_ERROR_DEVICE_LOST` on Arc is the level-zero driver losing the device —
typically VRAM exhaustion or a TDR-style reset, not a graph error.

---

## 3. Confirmed bug: `h3R2V` escalation is dead code — **FIXED (`655b131`)**

`lib/offload-config.js:27`

```js
function getOffloadConfig(gpuVendor, vramGb = 0, options = {}) {
  const h3R2V = options.h3R2V === true;      // line 30
  ...
  if (h3R2V && vram > 0 && vram <= 48) {     // line 42 — NEVER REACHED
```

Every call site passed the wrong third argument, or none:

```js
server.js:4547  getOffloadConfig(p.gpuVendor, p.vramGb, p.vramFreeGb)   // number, not {options}
server.js:4567  getOffloadConfig(p.gpuVendor, p.vramGb, p.vramFreeGb)   // same
server.js:9357  getOffloadConfig(...gpuVendor, ...vramGb)               // omitted
server.js:9584  getOffloadConfig(vGpuVendor, vGpuVramGb)                // omitted
```

`grep h3R2V` across `lib/` + `server.js` found **no caller** that set it.
Verified: the branch had never executed.

- **Impact on Intel:** none directly — Intel returns early at line 37 with
  sequential offload always on.
- **Impact on NVIDIA ≤48 GB:** the intended OOM safety net for H3 R2V had
  never once run.

### Resolution (Phase 2, commit `655b131`)

1. The video/animate path now passes `{ h3R2V: engine === 'h3' }` — derived
   from the engine, so non-H3 engines are unaffected.
2. The two image paths dropped the stray `vramFreeGb` argument. Image
   generation never builds an H3 graph, so no escalation applies. **`vramFreeGb`
   itself was kept** — `lib/generation-capabilities.js` uses it for real
   budgeting; only the misuse in this argument slot was wrong.
3. `getOffloadConfig` now guards the options slot. A latent crash was found
   *by the new test*, not by inspection: `= {}` defaults on `undefined` but
   **not** `null`, so `getOffloadConfig(v, g, null)` threw
   `Cannot read properties of null (reading 'h3R2V')`.

**Verification:** 11 new tests (7 unit against the real export, 4 call-site
guards). Both fixes proven load-bearing by reverting each in turn — removing the
call-site `h3R2V` fails call-site tests 3 & 4; removing the null guard fails unit
test 7. Suite: **1315 pass, 0 fail**.

**Bonus finding:** `test/edit-image-input.test.js` asserted the *exact buggy
string* including `p.vramFreeGb`, so the test suite was actively pinning the bug
in place. Rewritten to assert intent rather than formatting. Worth watching for
this pattern elsewhere — a test that pins formatting can lock in a defect.

**Note for Phase 3:** `lib/generation-capabilities.js:150-160` already contains
VRAM budgeting logic (`vramFreeGb`, `lowVramThreshold`, `requiredGb`). Phase 3
should extend that rather than build a parallel estimator.

---

## 4. Second issue: the DiT quant — **RESOLVED, DiT is NOT the problem**

The failing job loaded a DiT whose filename looks alarming on an Intel GPU:

```
unet_name = minimax_h3_fl2va_pruned_int8_convrot.safetensors
```

`int8_convrot` is the same family label as the text encoder that broke on XPU,
so this was flagged as a suspect. **Phase 4 cleared it.** Evidence:

### 4.1 The DiT's actual quant layout

Read from the checkpoint's `comfy_quant` payload (`blocks.0.attn.out_proj`):

```json
{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}
```

| Property | H3 DiT | H3 CLIP (nvfp4, working) |
|---|---|---|
| tensors | 932 | 2054 |
| dtypes | I8 200, U8 200, F32 210, BF16 220, F16 102 | U8 701, BF16 651, F8_E4M3 350, F32 351 |
| `.comfy_quant` | 200 | 351 |
| `*_rotary` tensors | **0** | 0 |
| `.weight_scale_2` (NVFP4) | 0 | 350 |

So the DiT is plain per-channel **INT8 tensorwise** with a ConvRot rotation flag —
*not* the W4A4 ConvRot variant, and it ships no rotary weight tensors.

### 4.2 It dispatches to comfy_kitchen — but the EAGER backend

`comfy/ops.py:984` routes this layout to `quant_ops.ck.int8_linear`, where `ck`
is `comfy_kitchen`. That is the same extension implicated in the CLIP failure,
and there is **no eager fallback** exposed on `quant_ops`
(`int8_linear`, `INT8_EAGER`, `int8_linear_eager` all absent).

However `comfy_kitchen` resolves internally to
`comfy_kitchen/backends/eager/quantization.py` — a pure-PyTorch path that works
on XPU. Confirmed by direct call on `xpu:0` (tiny 256×128 tensors, a few KB):

```
ck.int8_linear(x, qdata, scale, None, bf16, convrot=True, convrot_groupsize=256)
  -> out shape=(8, 128) dtype=torch.bfloat16 device=xpu:0
  -> finite values: True
```

Also verified: `grep comfy_kitchen comfy/ops.py` → **0 hits**, and no
`is_cuda` / `.cuda()` / `device.type == 'cuda'` gate exists on any dequant path.

**Conclusion: the DiT's int8+convrot path is functional on Intel Arc.** It is
*not* a contributor to the crash, and swapping the DiT is not needed. This
removes Phase 4 from the remaining work.

> Method note: my first smoke test failed with `ConvRot group size 256 does not
> divide input features 64` — my own test-fixture error (I picked 64 input
> features), not a code defect. Corrected to 256 and it passed. Worth recording
> so the earlier failure isn't later misread as evidence against the DiT.

---

## 5. Crash forensics (Phase 0 — COMPLETE, read-only)

### 5.1 The crash left no dump

Windows System log around the crash:

| Time | Event ID | Provider | Meaning |
|---|---|---|---|
| 15:55:42 | 172 | Kernel-Power | power-transition detail |
| 15:55:42 | **41** | Kernel-Power | "rebooted without cleanly shutting down" |
| 15:55:58 | 6008 | EventLog | unexpected shutdown recorded |

```
BugcheckCode : 0
MEMORY.DMP   : absent
Minidump     : none
```

Bugcheck 0 with no dump = Windows never reached the bugcheck writer. A **hard
hang or power-level reset**, not a trappable driver fault. The GPU driver was
fine afterwards (`Status: OK`, `32.0.101.8805`).

### 5.2 TDR recovery is DISABLED on this machine — critical

```
HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers
    TdrLevel = 0        <-- 0 = TDR DETECTION OFF (default is 3 = recover)
    TdrDelay = 120      <-- 120 seconds (default is 2)
```

This is the single most important finding of the investigation.

**What TDR normally does:** if the GPU stops responding for ~2 s, Windows resets
the driver, kills the offending work, and the machine survives with a brief
flicker plus an Event 4101 "display driver recovered".

**With `TdrLevel = 0`:** that safety net is switched off. A wedged GPU is never
reset — the system simply waits, forever. This converts what should be a
*recoverable stutter* into a **full-system hang**.

That is exactly the observed signature:
- `UR_RESULT_ERROR_DEVICE_LOST` from level-zero (GPU stopped responding)
- **zero** Event 4101 TDR-recovery events in 24 h — because detection is off
- no bugcheck, no dump — nothing trapped it
- machine hung hard

**Consequence:** on this configuration *any* GPU hang, from any application,
takes the whole machine down. That is a pre-existing platform setting, not an
H3 or Mix Studio bug — but it converts every VRAM overcommit into a hard crash
rather than a caught error, which is why resource management here matters far
more than on a default system.

### 5.3 This is chronic, not new

14-day shutdown history:

```
hard crashes (Event 41) : 38
clean shutdowns (6006)  :  1
```

**38 hard crashes to 1 clean shutdown.** This long predates today's H3 work and
predates commit `c25f644`. Two of the 38 recorded a bugcheck code:

| Date | Bugcheck | Name |
|---|---|---|
| 8/17 14:23 | 455 = `0x1C7` | `STORE_DATA_STRUCTURE_CORRUPTION` |
| 8/16 22:40 | 299 = `0x12B` | `FAULTY_HARDWARE_CORRUPTED_PAGE` |

`0x12B` is a **hardware-level** indicator (single-bit memory/bus error). No
WHEA-Logger events were recorded, so it is not conclusive on its own.

RAM checked and looks sane: 4 × 32 GB = 128 GB, `F4-2666C18-32GVK`, running at
rated 2667 MT/s with no overclock (`Speed == ConfiguredClockSpeed`).

**Read carefully:** 38 crashes in 14 days with `TdrLevel = 0` is most plausibly
the TDR setting turning ordinary GPU hangs into hard resets — this machine runs
heavy GPU work daily. But `0x12B` means underlying hardware cannot be ruled out
and deserves a memory test independently of anything in this repo.

---

## 6. What is NOT yet known

Being explicit about the limits of this diagnosis:

- **No VRAM telemetry from the crash.** I deleted `_comfy_restart.log` during
  cleanup before the fatal run — a mistake. There is no peak-allocation trace.
  (Fixed going forward: see §6.1.)
- **No crash dump** (see §5.1), so no kernel-side evidence of the hang cause.
- **Whether the crash is VRAM exhaustion specifically**, versus the
  int8_convrot DiT on XPU, or 124 frames at 608×384 simply being too large.
  `DEVICE_LOST` is a symptom, not a cause.
- **Whether `0x12B` indicates real failing hardware** or is a downstream
  artifact of the GPU hangs.
- Whether reducing frames/resolution alone makes it pass.

### 6.1 Phase 0 deliverables (in place, not yet exercised)

| Artifact | Purpose |
|---|---|
| `D:\...\ComfyUI\start-comfy-logged.bat` | launches ComfyUI with a **timestamped, append-only** log under `ComfyUI\logs\`; `-u` unbuffered so a hard crash keeps the tail |
| `D:\...\ComfyUI\vram-monitor.py` | samples `torch.xpu` allocated/reserved/peak to CSV, `flush + fsync` per row so evidence survives a hang |

Both syntax-checked. Neither has been run against the GPU.

---

## 7. Proposed plan (ordered, each step gated)

### Phase 0 — Preserve evidence (no risk)
1. Check Windows Event Log for the bugcheck / driver TDR around 15:5x.
2. Confirm Arc driver version + whether a TDR was recorded.
3. Re-enable persistent ComfyUI logging so the next run leaves a trace.

### Phase 1 — Measure before changing (low risk, no GPU stress)
4. Instrument a *small* H3 i2v run (short length, low res) with
   `torch.xpu.max_memory_allocated()` sampling to get a real peak-VRAM curve.
5. Determine empirically where the ceiling is on this hardware:
   frames × resolution → peak VRAM.

### Phase 2 — Fix the confirmed bug (small, testable)
6. Fix the `getOffloadConfig` call sites so `options` is an object and
   `h3R2V` actually reaches the branch. **TDD:** failing test first proving
   `h3R2V` is ignored today.
7. Decide `vramFreeGb`'s fate: either consume it deliberately or remove it
   from the signature. No silently-ignored parameters.

### Phase 3 — Real resource management (the actual request)
8. Add an H3 VRAM budget estimator: sum the models the graph will load +
   estimated latent/activation cost from frames × W × H.
9. Add a **preflight guard**: when the estimate exceeds the budget, either
   (a) escalate offload automatically, or (b) refuse with an actionable message
   ("124 frames at 608×384 needs ~N GB; you have 32 GB — try 97 frames")
   *before* submitting to ComfyUI. Failing fast beats crashing the machine.
10. Surface the estimate in the UI so the cost is visible before clicking
    generate.

### Phase 4 — Investigate the DiT quant
11. Determine whether `int8_convrot` is safe/performant for the DiT on XPU, and
    whether an nvfp4 or bf16 DiT variant exists.

---

## 8. Recommendation — REVISED after Phase 0

Phase 0 changed the priority order. The most valuable fix is no longer in this
repo at all.

### 8.1 Do this first: re-enable TDR (outside Mix Studio)

`TdrLevel = 0` means **any** GPU hang hard-hangs the machine. Restoring the
default converts today's crash class into a recoverable driver reset:

```
HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers
    TdrLevel = 3     (default: detect + recover)
    TdrDelay = 10    (generous for long AI kernels, but not infinite)
```

`TdrDelay = 10` rather than the 2 s default: heavy diffusion kernels can
legitimately block for several seconds, and 2 s risks false resets mid-inference.
10 s tolerates real work while still catching a genuine wedge.

**This is a system-level registry change requiring a reboot, and it is the
user's call — I have not made it.** Note the caveat: TDR resets kill the running
job, so a long generation that trips it is lost. That is still strictly better
than losing the whole session.

**Why `TdrLevel = 0` may have been set deliberately:** some AI workflows disable
TDR precisely to stop long kernels being killed. If that was intentional, the
tradeoff should be a conscious one — `TdrDelay` high with detection ON, rather
than detection OFF entirely.

### 8.2 Then: rule out hardware

Bugcheck `0x12B FAULTY_HARDWARE_CORRUPTED_PAGE` on 8/16 warrants a memory test
(`mdsched.exe`, or MemTest86 for a thorough pass) independently of any code here.
38 crashes in 14 days should not be normalised.

### 8.3 Then, in this repo

- **Phase 1** — measure peak VRAM on small runs, step up. Tooling in place
  (§6.1). **BLOCKED on TDR:** running real generations while `TdrLevel = 0`
  risks another full-system hang.
- **Phase 2** — ~~fix the `h3R2V` dead-code bug~~ **DONE (`655b131`)**.
- **Phase 3** — VRAM budget estimator + preflight guard that refuses
  over-budget jobs with an actionable message. Extend
  `lib/generation-capabilities.js` (which already has budgeting) rather than
  building a parallel estimator.
- **Phase 4** — ~~investigate the `int8_convrot` DiT on XPU~~ **DONE — the DiT
  is fine** (§4). No DiT swap needed; suspicion removed.

### 8.4 Do NOT do

Re-run the 124-frame 608×384 job to "see if it still fails." With TDR disabled
that risks another hard crash, and it would produce no new information.

### 8.5 Current blocking state

With Phases 0, 2 and 4 complete, the remaining work splits cleanly:

| Work | Needs GPU? | Blocked? |
|---|---|---|
| Phase 3 estimator + preflight guard | no | **no — can proceed** |
| Phase 1 empirical VRAM curve | yes | yes, until TDR is re-enabled |

Phase 3 can be built now using the measured **model sizes** (§2, 40.28 GB from
disk) with a conservative activation margin, then *calibrated* by Phase 1 once
GPU work is safe. That ordering avoids sitting idle while keeping the risky step
gated.

---

## 9. Phase 3 — VRAM preflight guard (**DONE, `1455044`**)

Added to `lib/generation-capabilities.js`, extending the existing
`ltxRefinePreflight` / `ltxVramEstimate` pattern rather than a parallel one:

| Function | Purpose |
|---|---|
| `h3VideoLatentT(frames)` | mirrors ComfyUI's `video_latent_t` exactly (`nodes_minimax_h3.py:39-40`) |
| `h3VramEstimate(opts)` | resident working set under sequential offload |
| `h3Preflight(opts)` | blocks with `409 h3_insufficient_vram`, or passes with `needsLowVram` |

Latent geometry is taken from the real node shape
`[B, 24, latent_t, height//16, width//16]` at 24 fps — not guessed.

Wired into the animate route beside the LTX guard, receiving real request
geometry plus `mode`/`turbo` (reference mode loads an extra DiT + audio VAE).

### 9.1 Measured: clip length is nearly irrelevant for H3

The most useful finding of Phase 3, and it inverted the design:

| Geometry | Estimate |
|---|---|
| 32 × 32 × 5 frames | **17.00 GB** |
| 608 × 384 × 5 | 17.00 GB |
| 608 × 384 × 124 | 17.03 GB |
| 1344 × 768 × 362 | **17.52 GB** |

**A 500× increase in pixels×frames moves the total by 0.5 GB.** The fixed model
working set (~14 GB + 3 GB margin) is the entire cost; the latent and attention
workspace are megabytes.

Consequences:

1. My first implementation suggested "use fewer frames", then "use a lower
   resolution". **Both are dead ends** — the resolution-search branch was
   literally unreachable code. Removed.
2. The guard now names the floor and the shortfall and points at the only real
   levers: free VRAM, or use a lighter engine.
3. **This reframes the original crash.** At 30 GB free the real job (608×384,
   124 frames) is estimated at 17.0 GB and *passes* preflight. So the crash was
   probably **not** a simple static overcommit — something at runtime consumed
   far more than the resident-weight model predicts, or the level-zero hang had
   another trigger. Phase 1 measurement is needed to resolve this; the guard
   protects the clear-cut cases but the estimate is **uncalibrated**.

### 9.2 Honest limitations

- Component figures (dit 6.0, textEncoder 3.5, videoVae 2.5, overhead 2.0) are
  **conservative engineering estimates, not measurements.** Phase 1 must
  calibrate them against `torch.xpu.max_memory_allocated`.
- The estimate cannot see other processes' VRAM use beyond what `vramFreeGb`
  reports at request time.
- A job that passes preflight can still OOM if the true peak exceeds the model.

**Verification:** 20 new tests (15 unit + 5 route-wiring). The route guard was
proven load-bearing by deleting it — 4 of 5 route tests fail. Suite: **1335
pass, 0 fail**.

```
Mix Studio HEAD : 1455044
Test suite      : 1335 pass, 0 fail
Phases complete : 0 (forensics), 2 (h3R2V), 3 (preflight), 4 (DiT cleared)
Phase 1         : BLOCKED — TdrLevel still 0, no reboot yet
ComfyUI         : running, port 49000
GPU             : Intel Arc Pro B70, driver 32.0.101.8805, Status OK
Last boot       : 2026-08-18 15:55:39
H3 i2v loader   : FIXED (c25f644) — 27/27 vision blocks load
```
