# Plan: Merge Intel Arc GPU Support to Main Branch

## Summary

The `feature/intel-arc-gpu-support` branch contains Intel Arc GPU support that is **partially committed and partially uncommitted in the working tree**. The branch was merged from `origin/main` at commit `fe0ff8f` (which was at v1.1.0 era commit `c189f37`), but `origin/main` has since advanced to v1.2.4 (`746e555`) with 37 additional commits that **have not been merged into the feature branch**. Commit `8bd16ec` (HEAD) also explicitly restored `release.json` to v1.1.0, a **regression**.

The goal is to rebase the feature branch onto the current `origin/main`, incorporate the 37 missing v1.2.0–v1.2.4 changes, then merge to `main`.

---

## Current State

### Branch state
- **Currently checked out**: `feature/intel-arc-gpu-support` (HEAD `8bd16ec`)
- **Origin/main HEAD**: `746e555` (v1.2.4, released 2026-08-09)
- **Feature branch last merged from main**: `c189f37` (v1.1.0 era) via `fe0ff8f`
- **Missing on feature branch**: 37 commits between `c189f37` and `746e555` covering v1.2.0–v1.2.4

### Intel Arc changes status
- **Committed** (in commit `ec0d86f`): `lib/dependency-installer.js`, `lib/edit-image-input.js`, `lib/generation-capabilities.js`, `lib/hardware-info.js`, `lib/ltx-director-workflows.js`, `lib/model-loader.js`, `lib/offload-config.js` (committed version), `lib/queue-health.js`, `lib/setup-guide.js`, `lib/upscale-workflows.js`, `lib/video-upscale-workflows.js`, `public/app.js`, `public/index.html`, `public/style.css`, `release.json` (was bumped to 1.2.4 then reverted), `installer/feature-manifest.json`, `server.js` (committed version), and many test files.
- **Uncommitted in working tree** (11 files, 450 insertions / 84 deletions): `lib/comfy-restart.js`, `lib/offload-config.js` (additional changes on top of committed), `lib/video-workflows.js`, `public/app.js` (additional), `public/index.html` (additional), `server.js` (additional), `start.bat`, `test/comfy-start.test.js`, `test/offload-config.test.js`, `test/portable-installer.test.js`, `test/video-ui.test.js`
- **Untracked**: `.kilo/`, `app.asar`, `lib/error-handler.js`, `lib/storage-manager.js`, `migrate_data.py`, `server.js.bak` (9455-line pre-Intel-Arc backup)

### Critical regression on HEAD
- `release.json` at HEAD shows v1.1.0 (was bumped to 1.2.4 in `ec0d86f`, then reverted by `8bd16ec`)

### Missing v1.2.0–v1.2.4 features (not in feature branch)
| Version | Key features |
|---------|-------------|
| v1.2.0 | Mix Packs (Create/Edit/video prompt fields, preset-card persistence, gallery metadata, documentation exports) |
| v1.2.1 | Wan Animate 2 (image identity, driving-video motion, performance ref, output sizing, center-crop), H3 Replace mode, H3 Reference Turbo (Kijai LightX2V adapter), H3 restyle presets, H3 Standard/Full BF16/DynTime model variants, Jobs/Downloads task-queue tabs, model cleanup, local prompt AI providers (text+vision), H3 Long Context (up to 120s sequential clips), Match video aspect option |
| v1.2.2 | Wan Animate 2 resize validation fix, Revise Prompt stage-aware progress card |
| v1.2.3 | Mobile Library responsiveness fixes |
| v1.2.4 | Mobile video proxy clips, profile-level Preview quality preferences, update showcase carousel (swipe, keyboard nav, auto-cycle) |

### feature-manifest.json divergence
- **Local (feature branch)**: has `video.4k` (SeedVR2 4K upscaling), missing `video.wanAnimate2` (present on main)
  - `video.h3` hardware variant text differs: local omits "NVIDIA NVFP4/AWQ and" → "uses INT8 ConvRot and NVFP4/AWQ kernels"
  - `video.ltx` variant text: local adds "Intel Arc XPU with dynamic VRAM offload"
- **Main (v1.2.4)**: has `video.wanAnimate2`, no `video.4k`

### Key Intel Arc changes (committed + uncommitted combined)
1. `lib/hardware-info.js`: Intel Arc B-series VRAM detection via `intelArcVramBytes()` (B70/B80 = 32GB), `xpu-smi` / WMI GPU listing, `gpuVendor()` returns `'intel'` for ARC devices
2. `lib/offload-config.js`: XPU device, SDPA attention, sequential CPU offload (UNETLoader `offload_device: 'cpu'`), H3 R2V forced offload for ≤48GB VRAM, `resolveVramProfile()` helper
3. `lib/comfy-restart.js`: `run_intel_gpu.bat` detection, scratch disk dir (`E:\tmp\pytorch`) passed to `pythonLaunchEnvironment()`, `TEMP`/`TMP`/`PYTHONPYCACHEPREFIX` env vars on Windows
4. `lib/dependency-installer.js`: INT8 ConvRot text encoder model variant selection for Intel Arc
5. `lib/video-workflows.js`: H3 memory-efficient attention patch (`MiniMaxH3MemoryEfficientSageAttentionPatch`), `unloadModel` (`UnloadModel` node) support, scheduler `modelRef` decoupling from SageAttention
6. `lib/video-upscale-workflows.js`: SeedVR2 CPU fallback for Intel Arc
7. `lib/ltx-director-workflows.js`: XPU device handling (LTXAVTextEncoderLoader accepts `default`/`cpu` on XPU)
8. `server.js`: Intel Arc gating in `/api/animate` route (H3 constrained VRAM detection, H3 memory-efficient attention, R2V disk space pre-flight check `H3_R2V_MIN_DISK_GB=5`, scratch disk dir in settings), `resolveVramProfile()` usage
9. `start.bat`: Scratch disk configuration (`E:\tmp\pytorch`), Node.js version check (≥22), `MIXBOX_RESTART_MODE=launcher`
10. `public/app.js` + `public/index.html`: Audio track remove buttons (`vidAudioX` / `animAudioX`), sequential batch video submission (for-loop instead of `Promise.allSettled`), Intel Arc UI gating
11. `public/style.css`: Intel Arc-specific styles (committed)

### Validation state
- **Unknown**: whether `node --test` currently passes on the feature branch HEAD (tests assume Intel-specific behaviors that may conflict with main's v1.2.x test expectations)
- **Unknown**: whether `node --check server.js` passes with the combined changes
- **Unknown**: whether the uncommitted test changes conflict with any committed v1.2.x test additions

### Risks
1. **v1.2.0–v1.2.4 merge conflicts**: `server.js` is 172 lines of Intel Arc changes on top of a 9455-line file; main has 315+ lines of changes across 37 commits in the same file. The uncommitted Intel Arc diff in `server.js` touches routing, hardware profile, VRAM checking, and H3 R2V pre-flight — likely overlapping with main's H3/H3R2V Long Context changes.
2. **`start.bat` rewrite**: The uncommitted change rewrites `start.bat` from a minimal launcher to a full Node.js version-checking, scratch-disk-configuring launcher. Main's `start.bat` may have evolved too.
3. **`release.json` regression**: `8bd16ec` reverted it to v1.1.0 — needs to be set to v1.2.4 (or higher after merge).
4. **`feature-manifest.json` divergence**: Local has `video.4k` (not on main), missing `video.wanAnimate2` (on main). Both need reconciliation.
5. **Test conflicts**: 4 test files have uncommitted Intel Arc changes; main v1.2.x may have added conflicting test assertions (especially v1.2.2's Wan Animate 2 batch validation, v1.2.4's showcase carousel, mobile preview tests).
6. **Untracked files**: `server.js.bak` is a 9455-line backup. `lib/error-handler.js`, `lib/storage-manager.js`, `migrate_data.py` may be Intel Arc-related or unrelated.
7. **`lib/model-loader.js`**: Committed in `ec0d86f` with Intel Arc changes — needs to coexist with main's changes.

### Out of scope
- `data/` directory (live user data, never in git)
- `docs/superpowers/plans/` (old planning docs, gitignored)
- `.kilo/` (config)
- `app.asar` (packaging artifact)

---

## Plan

### Phase 1: Pre-merge validation of current state

1. **Validate current feature branch compiles and tests pass**
   - `node --check server.js`
   - `node --test` (verify all tests pass with current uncommitted Intel Arc changes)
   - Record pass/fail status

2. **Inventory committed vs. uncommitted Intel Arc changes**
   - `git diff HEAD --stat` (already have: 11 files)
   - `git show ec0d86f --stat` (already have: 30 files)
   - Document the overlap: which files have BOTH committed and uncommitted Intel Arc changes

3. **Identify untracked files' relevance**
   - Read `lib/error-handler.js`, `lib/storage-manager.js`, `migrate_data.py` to determine if they're Intel Arc-related
   - Confirm `server.js.bak` is a pre-Intel-Arc backup (9455 lines confirms this)

### Phase 2: Reconcile uncommitted changes

4. **Commit uncommitted Intel Arc changes**
   - Stage and commit the 11 modified files: `lib/comfy-restart.js`, `lib/offload-config.js`, `lib/video-workflows.js`, `public/app.js`, `public/index.html`, `server.js`, `start.bat`, `test/comfy-start.test.js`, `test/offload-config.test.js`, `test/portable-installer.test.js`, `test/video-ui.test.js`
   - Commit message: `feat(intel-arc): serialize pipeline gate, H3 R2V offload, scratch disk, audio remove UI`

5. **Resolve `release.json` regression**
   - Update `release.json` to `{"version": "1.2.4", "releasedAt": "2026-08-09"}` (match `origin/main`)
   - OR set to a new version like `1.2.5` if Intel Arc support warrants a new release — **decision point, recommend matching main at 1.2.4**

### Phase 3: Rebase onto origin/main

6. **Rebase onto origin/main (v1.2.4)**
   - `git rebase origin/main` while on `feature/intel-arc-gpu-support`
   - This will attempt to apply all Intel Arc commits (`ec0d86f`, `8bd16ec`, and the new commit from step 4) on top of v1.2.4
   - **Expected conflicts** in:
     - `server.js` (H3/H3R2V Long Context + Intel Arc offload changes)
     - `lib/offload-config.js` (SeedVR2 CPU fallback vs. Intel Arc offload config)
     - `lib/video-workflows.js` (H3 Long Context vs. H3 memory-efficient attention)
     - `public/app.js` (H3 Long Context UI + mobile preview fixes vs. Intel Arc UI gating + sequential batch submission)
     - `public/index.html` (mobile preview proxy clips vs. Intel Arc audio remove buttons)
     - `test/` files (Wan Animate 2, showcase carousel, mobile preview tests vs. Intel Arc test changes)

### Phase 4: Resolve conflicts

7. **Resolve `server.js` conflicts**
   - Intel Arc: serialization gate (`runSerialized`, `waitForComfyIdle`), H3 constrained VRAM, H3 memory-efficient attention, R2V disk pre-flight, scratch disk in settings
   - Main v1.2.x: H3 Long Context (120s sequential clips), H3 Reference Turbo, H3 Replace mode, Wan Animate 2 routing
   - **Strategy**: Keep both sets of logic — Intel Arc offloading config applies on top of the H3 Long Context workflow structure; serialization gate wraps all prompt submissions regardless of engine

8. **Resolve `lib/video-workflows.js` conflicts**
   - Intel Arc: `h3_mem_attention` patch, `unloadModel`/`UnloadModel` node, `modelRef` decoupling
   - Main v1.2.x: H3 Long Context sequential clip joining, H3 Motion Context node
   - **Strategy**: Intel Arc memory-efficiency patches are orthogonal to Long Context clip joining — merge both

9. **Resolve `public/app.js` conflicts**
   - Intel Arc: audio remove buttons (`vidAudioX`/`animAudioX`), sequential batch submission
   - Main v1.2.4: mobile video proxy clips, profile-level preview quality, showcase carousel, H3 Long Context UI
   - **Strategy**: Keep all — sequential batch submission is compatible with v1.2.4's preview optimizations

10. **Resolve `lib/offload-config.js` conflicts**
    - Intel Arc: sequential offload config, `h3R2V` option, `getOffloadConfig` signature change
    - Main v1.2.x: SeedVR2 CPU fallback (4K upscale)
    - **Strategy**: Intel Arc offload config for Intel vendor; SeedVR2 fallback is separate path in `video-upscale-workflows.js`

11. **Resolve `test/` file conflicts**
    - Intel Arc: new tests for scratch disk env, offload config, audio remove buttons, sequential batch
    - Main v1.2.x: Wan Animate 2 batch validation, showcase carousel, mobile preview proxy tests
    - **Strategy**: Merge both sets of test assertions; ensure no contradictory expectations

12. **Resolve `feature-manifest.json` divergence**
    - Local has `video.4k` (Intel Arc SeedVR2 4K upscaling) — **keep** (new feature)
    - Local missing `video.wanAnimate2` — **add** (from main)
    - Update `video.h3` hardware variant text to include Intel Arc compatibility (add `"intel"` to supported vendors or add Intel Arc notes)

13. **Resolve `start.bat` conflicts**
    - Intel Arc: full Node.js version check, scratch disk config, `MIXBOX_RESTART_MODE=launcher`
    - Main v1.2.x: may have its own launcher changes
    - **Strategy**: Keep Intel Arc version (more robust) if it's a superset; otherwise merge

14. **Resolve `CHANGELOG.md`**
    - Merge Intel Arc features into the Unreleased section alongside v1.2.0–v1.2.4 entries
    - Ensure v1.2.0–v1.2.4 sections are preserved

### Phase 5: Post-merge validation

15. **Validate after rebase**
    - `node --check server.js`
    - `node --test` (all tests pass)
    - Verify no regressions in v1.2.0–v1.2.4 features

16. **Feature verification (read-only)**
    - Confirm Intel Arc GPU detection (`lib/hardware-info.js`) is present and matches B-series 32GB
    - Confirm H3 R2V offload (`lib/offload-config.js`, `lib/video-workflows.js`) is wired through `server.js`
    - Confirm scratch disk env vars propagate (`lib/comfy-restart.js` → `start.bat`)
    - Confirm INT8 ConvRot selection (`lib/dependency-installer.js`) is present
    - Confirm SeedVR2 CPU fallback (`lib/video-upscale-workflows.js`) is present
    - Confirm sequential pipeline gate (`server.js`) wraps all submissions
    - Confirm audio remove UI (`public/app.js` + `public/index.html`) is present

17. **Merge to main**
    - Checkout `main`
    - `git merge feature/intel-arc-gpu-support`
    - Run final `node --test` on merged result
    - Update `release.json` to `1.2.4` (if not already at v1.2.4)

### Phase 6: Cleanup

18. **Remove untracked artifacts**
    - `server.js.bak` (9455-line pre-Intel-Arc backup) — safe to delete
    - `lib/error-handler.js`, `lib/storage-manager.js`, `migrate_data.py` — evaluate; if Intel Arc-related and needed, commit; if not, leave or remove
    - `app.asar` — packaging artifact, ignore

19. **Update stash**
    - `git stash drop` if stash is now obsolete (the stash was for `feature/intel-arc-gpu-support` at `8bd16ec` — if rebased work was committed in step 4, stash can be dropped)

---

## Decision Points

1. **`release.json` target version**: Should the merged result be tagged as v1.2.4 (matching main) or a new v1.2.5/v1.3.0? **Recommend: v1.2.4** — the Intel Arc support is being merged into the existing v1.2.4 main branch state, not released as a separate version. The Intel Arc support becomes part of the v1.2.4 release.

2. **`feature-manifest.json` `video.h3` vendor support**: Main marks H3 as `unsupportedVendors: ["apple", "amd"]`. Should Intel be added as supported? **Recommend: Yes** — Intel Arc B70/B80 (32GB) with sequential CPU offload can run H3. Add `"minVramGb": 24` and Intel Arc notes to the variant text, but keep Intel in the "autoSelect: recommended" tier (not always-supported like NVIDIA).

3. **`start.bat` final form**: Keep the Intel Arc version (with Node.js version check + scratch disk) or merge with main's version? **Recommend: Keep Intel Arc version** — it's a strict superset with better error handling.

---

## Validation Criteria

- `node --check server.js` passes
- `node --test` passes all test files (0 failures)
- `release.json` shows version `1.2.4`
- Intel Arc GPU detected in `lib/hardware-info.js` with B70/B80 = 32GB
- Intel Arc offload config in `lib/offload-config.js` returns sequential offload for Intel vendor
- `comfy-restart.js` propagates scratch disk env vars to ComfyUI subprocess
- `server.js` H3 R2V pre-flight checks disk space before submission
- `server.js` serialization gate wraps all `queuePrompt`/`uploadToComfy`/`uploadFileToComfy` calls
- `feature-manifest.json` includes both `video.4k` (Intel Arc) and `video.wanAnimate2` (main)
- `CHANGELOG.md` contains both Intel Arc entries and v1.2.0–v1.2.4 entries
- No untracked `.bak` files or artifacts remain in the repo

---

## Rollback Plan

If merge conflicts are intractable or tests fail after rebase:
1. Abort rebase: `git rebase --abort`
2. Merge instead: `git merge origin/main` on `feature/intel-arc-gpu-support` (non-fast-forward)
3. Resolve conflicts in merge commit
4. If still unworkable, the Intel Arc changes remain on the feature branch and `origin/main` is left untouched
