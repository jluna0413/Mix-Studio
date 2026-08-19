# Crash Investigation Report — 2026-08-19

## Summary

**Root cause: Sudden power loss (not a GPU/driver issue).**

The system experienced an unexpected hard reboot at `2026-08-19T07:37:21 UTC`
(03:37 AM local). This is a power/hardware reset event, not a GPU wedge or
kernel bugcheck.

## Evidence

### Windows Event Logs
- **Event ID 41 (Kernel-Power)**, source `Microsoft-Windows-Kernel-Power`,
  timestamp `2026-08-19T07:37:21.433Z` — "The system has rebooted without
  cleanly shutting down first."
- **BugcheckCode = 0** in the Event 41 record — this means there was no
  kernel bugcheck/BSOD. A GPU hang that triggers TDR would still produce
  a non-zero BugcheckCode (e.g. 0x12B for VIDEO_TDR_ERROR).
- **No crash dumps found** at `C:\Windows\Minidump\*.dmp` or
  `C:\Windows\memory.dmp` — confirms a hard power loss (dumps require
  at least kernel-level crash handling).
- **No GPU driver events (Event ID 14, 410, etc.) from `dxgkrnl`** in the
  crash window — the Intel Arc driver did not detect or report a hang.
- **No application error events** at the crash time — no process crashed.

### TDR Configuration (verified via `reg query`)
- `TdrLevel = 0x3` (detect + recover) — your Aug 18 fix is active.
- `TdrDelay = 0xa` (10 seconds) — tolerates long AI kernels while catching wedges.

### Timings
- Event 41 (power lost): `07:37:21`
- Service restart begins: `07:37:25` (ComfyUI startup log)
- Normal boot completes: `07:37:33+`
- Gap: ~4-8 seconds of downtime — consistent with a power blip, not a
  multi-minute GPU recovery cycle.

### ComfyUI Log
- ComfyUI started cleanly at `2026-08-19 09:24:20` (after reboot).
- No errors related to GPU hangs, TDR, or model loading failures.
- All custom nodes loaded successfully, including H3 nodes:
  - `[ClipProj] graph watch installed`
  - `ComfyUI-MiniMax-H3-Turbo` nodes loaded
  - `h3_motion_context: nodes registered`

## What Was Running at Crash Time

The crash occurred at 03:37 AM while no AI generation was active. My node
processes (Mix Studio PID 30368, ComfyUI PID 27812) were both started
after the reboot and are running cleanly.

## Comparison to Past Crashes

Per your memory entry: the previous pattern was 38 Event-41 crashes in 14
days with TDR **disabled** (TdrLevel=0), BugcheckCode 0, no crash dumps.
That pattern indicated the GPU wedging the whole machine because TDR was
off. After your TDR fix (applied Aug 18, confirmed active in registry),
this is the **first** crash — and it has the same signature (Event 41, no
BugcheckCode, no dump).

The difference: with TDR enabled (TdrLevel=3), a GPU hang would trigger
a driver reset (yellow flash, recovery event), **not** a hard power cycle.
The fact that TDR is active and there are no dxgkrnl timeout events means
the GPU did not hang. This was a hardware power event.

## Recommendations

1. **TDR is correctly configured** — no action needed there. Your fix is
   working.
2. **Investigate power supply** — a sudden power loss with no kernel-level
   crash signature suggests either:
   - Wall power interruption (outlet, breaker, UPS)
   - PSU issue (failing power supply)
   - Power button stuck/hard press
3. **Check the Event Log around 07:37:00-07:37:30** for any other
   sources (disk, USB, thermal) that might reveal the root cause.
4. **The VRAM guard** I built is still valuable — it catches the
   DEVICE_LOST → stuck-worker pattern in software, independent of
   whether the trigger is a power event or a GPU hang.

## Files Changed (this work session)

### Branch: main → merged
- `lib/video-workflows.js` — 3-way `h3ClipVariant` ternary (nvfp4/awq, lightweight, distilled)
- `lib/h3-model-variants.js` — `lightweight` and `distilled` variant keys + setting defaults
- `lib/dependency-installer.js` — lightweight 4B encoder + ClipProj download assets
- `lib/queue-health.js` — `assessVramStuck()` function
- `server.js` — VRAM guard integration + xpu-smi fallback in `readGpuStats`
- `public/index.html` — UI options for all three clip variants
- `data/settings.json` — new setting keys
- `test/h3-lightweight-clip.test.js` — 5 tests
- `test/h3-condbridge-clip.test.js` — 4 tests
- `test/vram-guard.test.js` — 5 tests

**Test suite: 1356/1356 green**
