# Dev-Oops SOPs — Development Practices to Prevent Fix→Break Cycles

## Guiding Principle
Every fix must: (1) reproduce the bug, (2) identify root cause, (3) fix the cause, (4) verify with tests, (5) confirm no regressions. Skipping any step invites the next regression.

---

## 1. Reproduce Before You Touch Anything

- Write a minimal repro that triggers the bug. One command, one file, one error.
- If you can't reproduce it reliably, you don't understand it yet. Gather more data.
- For visual/AI output bugs: capture the exact input + the exact output + the exact error. Don't eyeball.
- Commit the repro to `tests/` so it lives beyond this session.

**Anti-pattern:** "It looked wrong, let me tweak it."

---

## 2. Root-Cause-First Debugging

Read the error trace fully before proposing any fix. Trace from the top of the stack down to the failing line, not the other way around.

For each proposed fix, state the hypothesis explicitly:
- "I believe X is the cause because Y."
- "If my fix works, Z should change in this specific way."

Test one hypothesis at a time. If a fix doesn't work, revert it and form a new hypothesis — don't stack fixes.

**Anti-pattern:** "The stack trace mentions Q, so I'll patch Q." (The mention may be a side effect, not the cause.)

---

## 3. Test-Engineered Fixes

For behavioral bugs: write a test that fails before the fix and passes after. The test encodes the contract, not the implementation.

For AI-generated output (images, video, captions): hard to write a traditional assert. Use:
- Reproducible seeds where possible.
- Structural checks (shapes, types, field presence, node wiring) that can be asserted without judging aesthetic quality.
- Known-good reference outputs stored as fixtures; diff against them.

**Anti-pattern:** "I'll just regenerate and look at it."

---

## 4. Change Sizing and Isolation

- One logical change per commit. If a commit fixes a bug AND refactors, split it.
- Keep the diff small enough to hold in your head. When a file grows past a comfortable boundary, consider extracting.
- Isolate workflows into their own branches/files/modules so a bug in one workflow doesn't silently regress another. (This is why we're doing the branch-per-workflow refactor.)

**Anti-pattern:** "While I'm here, I'll also clean up X."

---

## 5. Regression Discipline

After any fix:
- Run the repro again. It must no longer trigger.
- Run the full relevant test suite. Nothing else may break.
- Specifically check the *adjacent* workflows that share code paths (same loader, same encoder, same device handling, same ComfyUI node) — those are the most likely regression targets.

**Anti-pattern:** "The repro is fixed, ship it."

---

## 6. Review Your Own Changes

Before committing or pushing, re-read your diff as if someone else wrote it. Ask:
- Does this file's logic still make sense?
- Are there now two ways to do the same thing?
- Did I introduce a new failure mode (e.g. a path that now returns an unexpected type)?
- Are configs, defaults, or device-handling paths still consistent across workflows?

**Anti-pattern:** "It works locally, that's enough."

---

## 7. Document the Why, Not Just the What

- Commit messages and PR descriptions should explain *why* a fix addresses the root cause, not just "fixed crash."
- For AI-related changes, note what assumption changed (model, checkpoint, caption, device, dtype) and how you know it's the right one.
- When a fix is fragile or provisional, say so explicitly. "Works for this checkpoint; may not generalize" is more honest than silent optimism.

**Anti-pattern:** "Fixed it." (with no context for the next person)

---

## 8. Handle AI/Model Output Realities

When working with generative models (Flux LoRA, H3 video, etc.):

- Distinguish **deterministic bugs** (crash, wrong type, missing field, wiring error) from **quality issues** (bad pose, weird face, caption drift). Fix deterministic bugs with code; treat quality issues as validation + prompt/caption/seed iteration.
- A model "not producing good output" is often not a code bug. Confirm the pipeline actually runs end-to-end first, then iterate on prompts/captions/settings.
- For fine-tuning (LoRA): caption quality, image diversity, and training settings matter as much as the code. Keep captioning SOPs separate from pipeline SOPs.

**Anti-pattern:** Treating every bad output as a code defect.

---

## Checklist (per change)

- [ ] Bug reproduced with a minimal repro
- [ ] Root cause hypothesized and stated
- [ ] Fix addresses root cause, not just symptom
- [ ] Test added or updated (structural/assertable where possible)
- [ ] Test passes after fix
- [ ] Full relevant suite passes (no regressions)
- [ ] Adjacent workflows checked for regressions
- [ ] Change is one logical unit, appropriately sized
- [ ] Commit explains *why*, flags any fragility
- [ ] Config/device/type/wiring consistency verified across affected workflows

---

## When You Catch Yourself Doing This — Stop

- "I'll just patch it and see if it works."
- "The error mentions X, so I'll change X."
- "I'll fix the crash and also clean this up."
- "I don't have a test for this, but I'll check manually."

Each of these is a regression in waiting. Rewind to the reproduce → root-cause → test → fix → verify loop.
