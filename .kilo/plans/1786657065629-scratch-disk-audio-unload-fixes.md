# Fix Plan: Scratch Disk Redirect + Audio Track Unload

## Problem Statement

Three related issues on the Mix Studio Windows installation:

1. **Scratch disk offloading to D:** PyTorth temporary files go to D: (limited SSD) instead of E: (NVMe with space), causing crashes during H3 R2V reference preparation.
2. **No audio track unload button:** The `vidAudioX` remove button (`✕`) exists in `index.html` and `setAudioChipVisual()` toggles its `hidden` property, but no click handler is attached. The `animAudioChip` has no remove button at all.
3. **H3 R2V reference preparation crashes:** A direct consequence of issue 1 — the `MiniMaxH3ReferenceToVideo` node loads 4 large models simultaneously and writes temp tensors to D:, exhausting disk space.

## Root Cause Analysis

### Issue 1 & 3: Scratch Disk

- `start.bat` sets `TEMP`/`TMP` to `E:\tmp\pytorch` for the **Node.js server process** only.
- `lib/comfy-restart.js:pythonLaunchEnvironment()` sets `TEMP`/`TMP`/`PYTHONPYCACHEPREFIX` for ComfyUI processes **launched by Mix Studio**.
- `installer/redirect-scratch-disk.bat` uses `setx` to set user-level `TEMP`/`TMP` — **but has not been run** on this machine.
- ComfyUI is on **port 49000** (not 8188), suggesting it was launched externally (Comfy Desktop or manually), bypassing Mix Studio's env var injection.
- When Comfy Desktop launches ComfyUI, it does **not** propagate `TEMP`/`TMP` from the parent environment — the Python subprocess inherits Comfy Desktop's own env (system default → D:).
- `restart_comfyui_for_arc.bat` hardcodes `D:\apps\AI Playground\resources\ComfyUI\run_intel_gpu.bat` but the `cmd /v /c` wrapper does propagate vars. However, this script may not have been used, or Comfy Desktop was launched separately.

### Issue 2: Audio Unload

- `public/index.html:875` — `vidAudioX` button exists, `wireAudioChip` calls `setAudioChipVisual()` which toggles `hidden`, but **no `#vidAudioX` click handler exists**.
- `vidFaceX`, `vidAttachX`, `vidDriveX`, `animEndX` all have explicit click handlers — `vidAudioX` does not.
- The chip-level click handler (line 8660) does toggle audio off, but the `✕` button inside the chip doesn't stop propagation and has no dedicated handler, making it unreliable as a remove action.
- `animAudioChip` (line 2130 in HTML) has **no** `attach-x` child at all.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Scratch disk source | Make `SCRATCH_DISK_DIR` configurable via `settings.json` (`scratchDiskDir`) with fallback to `E:\tmp\pytorch` | Future-proof; allows the owner to redirect without code changes |
| ComfyUI restart method | Add a "Restart ComfyUI with scratch disk redirect" path in the K-menu (`/api/comfy/restart`) that kills and relaunches ComfyUI through Mix Studio, passing the env vars | The restart button already exists; this ensures env vars propagate |
| Audio unload UI | Add `#vidAudioX` click handler + add `animAudioX` button + handler | Mirrors existing pattern (`vidFaceX`, `vidDriveX`, etc.) |
| H3 R2V crash guard | Add a pre-flight disk space check before queuing H3 R2V jobs | Prevents system crashes by refusing to queue when D: has insufficient space |

## Tasks

### Task 1: Make scratch disk configurable in `lib/comfy-restart.js`

**File:** `lib/comfy-restart.js`

- Add `scratchDiskDir` resolution: read from `options.settings.scratchDiskDir` if provided, otherwise default to `'E:\\tmp\\pytorch'`.
- Update `pythonLaunchEnvironment()` to accept `options.scratchDiskDir` and use it instead of the hardcoded `SCRATCH_DISK_DIR`.
- Update `ensureScratchDiskDir()` to accept a custom dir.
- Export `SCRATCH_DISK_DIR` for testing.
- Update `spawnPythonComfy`, `startComfy` (desktop path), and `restartComfy` (portable/runScript paths) to pass the scratch disk dir into `pythonLaunchEnvironment()`.

**Verification:** `node --check lib/comfy-restart.js`

### Task 2: Wire scratch disk dir from server.js into ComfyUI restart/start

**File:** `server.js`

- Read `settings.scratchDiskDir` (if present) when calling `startComfy`/`restartComfy`.
- Pass `{ scratchDiskDir: settings.scratchDiskDir }` through the options to `comfy-restart`.
- The ComfyUI port is **49000** (from `data/settings.json`), not the default 8188 — ensure `restartComfy` uses the correct port from `settings.comfyUrl`.

**Verification:** `node --check server.js`

### Task 3: Add disk space pre-flight check for H3 R2V

**File:** `server.js`

- In the `/api/animate` POST handler, when `engine === 'h3'` and `h3Mode === 'reference'`, check available disk space on the drive where `TEMP`/`TMP` points (or D: as the default system temp).
- If less than ~5GB free, return a 409 with `code: 'h3_r2v_insufficient_disk_space'` and a clear message.
- Use `node:fs` `statSync` on the temp dir path to check free space.

**Verification:** `node --check server.js`

### Task 4: Add `#vidAudioX` click handler

**File:** `public/app.js`

- Add a click handler for `$('#vidAudioX')` that:
  1. Stops event propagation (so it doesn't also trigger the chip's toggle handler).
  2. Sets `state.vidAudio = null`.
  3. Calls `setAudioChipVisual($('#vidAudioChip'), false)`.
  4. Hides `$('#vidAudioTrim')`.
  5. Calls `saveForm()`.
- Place it near the other remove button handlers (around line 16042, after `vidFaceX`).

**Verification:** Test file `test/video-ui.test.js` — add assertion that `#vidAudioX` has a click handler.

### Task 5: Add `animAudioX` remove button to HTML and wire its handler

**File:** `public/index.html`

- Add `<button class="attach-x" id="animAudioX" aria-label="Remove audio track" hidden>✕</button>` inside `#animAudioChip` (line 2130).

**File:** `public/app.js`

- Extend `setAudioChipVisual()` to also toggle `#animAudioX` visibility (mirror the `vidAudioX` logic).
- Add click handler for `$('#animAudioX')` that clears `state.animAudio` and calls `setAudioChipVisual($('#animAudioChip'), false)`.

**Verification:** `test/video-ui.test.js` — assert `animAudioX` exists in HTML and has a handler in app.js.

### Task 6: Run `redirect-scratch-disk.bat` on the machine

**Operational step (not a code change):**

- Run `installer/redirect-scratch-disk.bat` as administrator to set user-level `TEMP`/`TMP` via `setx`.
- Kill ComfyUI (port 49000) + Comfy Desktop.
- Relaunch via `restart_comfyui_for_arc.bat` (which now sets `TEMP`/`TMP` inline) or launch Comfy Desktop **after** running the redirect bat.
- Verify: `echo $env:TEMP` in a new PowerShell shows `E:\tmp\pytorch`.

**Verification:** Check `E:\tmp\pytorch` is being used by the ComfyUI Python process (check Process Explorer or `wmic process where "name='python.exe'" get CommandLine` for `TEMP`).

### Task 7: Update/restore the redirect bat to be portable

**File:** `installer/redirect-scratch-disk.bat`

- Currently fine — uses `setx` for persistence. No changes needed unless we want to make the E: path configurable.

### Task 8: Update tests

**Files:** `test/comfy-start.test.js`, `test/video-ui.test.js`

- Add test: `pythonLaunchEnvironment()` includes `TEMP`/`TMP` pointing to scratch disk on win32.
- Add test: `pythonLaunchEnvironment()` uses custom `scratchDiskDir` when provided.
- Add test: `vidAudioX` click handler exists in `app.js`.
- Add test: `animAudioX` button exists in `index.html` and handler exists in `app.js`.
- Add test: H3 R2V disk space check in `server.js` (if feasible with mocking).

**Verification:** `node --test` — all tests pass.

## Rollout / Migration

1. Apply code changes (Tasks 1-5, 8).
2. Run `node --check` on all modified JS files.
3. Run `node --test` — must be green.
4. Run `installer/redirect-scratch-disk.bat` as admin (Task 6) — this is a **manual operational step** the user must perform.
5. Restart ComfyUI via `restart_comfyui_for_arc.bat` or Comfy Desktop.
6. Verify generation writes temp files to `E:\tmp\pytorch`.

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| `setx` requires admin / may fail silently | Check return code; fall back to inline env in batch scripts |
| Comfy Desktop doesn't propagate `TEMP`/`TMP` env vars | The `cmd /v /c` wrapper in `restart_comfyui_for_arc.bat` does propagate — ensure user restarts via that script, not directly launching Comfy Desktop |
| `vidAudioX` click handler conflicts with chip toggle | Call `event.stopPropagation()` before the chip handler fires |
| Disk space check may not work on all Windows drive configurations | Check the actual `TEMP` dir path, not assume D: |
| H3 R2V pre-flight check could block legitimate generations if threshold is too high | Set threshold to 5GB (conservative — H3 R2V temp tensors are ~2-3GB peak) |

## Out of Scope

- Changing the ComfyUI Python environment's `tempfile.gettempdir()` behavior (controlled by `TEMP`/`TMP` env vars, which is what we're fixing).
- Modifying ComfyUI's internal output directory (ComfyUI saves to its own `output/` dir — that is the disaster-recovery source of truth per AGENTS.md and should not change).
- Adding a UI for the scratch disk path — the owner can set it in `data/settings.json` if needed; defaults to `E:\tmp\pytorch`.
