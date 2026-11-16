# Handoff: Split upstream-Intel candidates vs fork-only (Mix Studio)

**Purpose:** prepare a reviewable, validated branch containing only **upstream-aligned Intel/general fixes** (candidate PRs to [BlackMixture/Mix-Studio](https://github.com/BlackMixture/Mix-Studio)), while keeping fork-only/integration work separate. This is a **local, non-destructive** prep task for a coding agent (OpenCode/Codex) — **do not push/merge/open PRs without human review and explicit approval**.

Read and follow `docs/agent-dev-workflow.md` (sandboxing, guardrails, validation, evidence). Treat this spec as the task scope.

## Repo & state (as of 2026-11-16, verify before acting)

- **Repo:** `E:\Mix Studio`
- **Fork remote:** `fork` → `https://github.com/jluna0413/Mix-Studio.git`
- **Upstream remote:** `upstream` → `https://github.com/BlackMixture/Mix-Studio.git`
- **Validated fork integration branch (pushed):** `fork/merge/upstream-smart-3d` @ `aa598bb` (local `merge/upstream-smart-3d` in sync)
  - `aa598bb` test: align memory diagnostics assertions with merged error handling
  - `c5b90c3` merge: integrate upstream/main (comparison zoom & memory diagnostics)
  - `3cf2d0e` fix: dedupe prompt provider test request args
  - `5a8cd1a` fix: allow H3 SageAttention render when runtime probe cannot be inspected
  - `2fc0e53` fix: allow H3 SLA render when runtime probe cannot be inspected
  - `faf9e4b` docs: align README with low-vram and installer copy tests
- **Upstream base at merge time:** `upstream/main` @ `6a234c9` (fetch to confirm latest before branching)
- **fork/main:** unchanged at `ffb1ee5` (not part of this handoff)

> Always `git fetch --all --prune` first and report the exact SHAs you based work on.

## Goal

1. Create a **sandboxed** branch `upstream-intel` based on `upstream/main` containing only upstream-candidate fixes that apply cleanly and are validated against upstream context.
2. Produce a clear **classification & evidence report** for every commit in scope (what was ported, what was skipped and why).
3. Leave the main checkout and existing branches untouched.

## Commit classification (starting point — agent must verify)

| Commit | Summary | Initial classification | Notes / verification guidance |
|---|---|---|---|
| `c5b90c3` | merge `upstream/main@6a234c9` | **integration (not upstream)** | Already upstream; never cherry-pick a merge commit into upstream-intel. |
| `aa598bb` | test: align memory diagnostics assertions with merged error handling | **fork-integration-only** | Adapts tests to our merged server error-handling shape; upstream’s test/server differ. Do not port unless you prove an equivalent issue exists on `upstream/main` (unlikely). |
| `3cf2d0e` | fix: dedupe prompt provider test request args (`server.js`) | **likely N/A on upstream** | On `upstream/main`, `/api/prompt/provider/test` (~route in server.js) appears to pass request args once (no duplicates). **Verify on your fetched `upstream/main`:** if clean, skip (do not manufacture a change). If a duplicate-arg bug exists upstream, port a minimal fix + confirm with relevant tests. |
| `5a8cd1a` | fix: allow H3 SageAttention render when runtime probe cannot be inspected | **upstream-candidate (evaluate)** | Adds `inspected` signaling in `lib/sage-attention.js` and relaxes the render gate so a node-present + non-inspectable probe does not hard-block (while still blocking when node missing or probe explicitly not ready). Upstream currently blocks on `!sageRuntime.ready || !nodeReady` without `inspected`. **Port only if:** it cleanly/adaptably applies to `upstream/main`, preserves safety (must still reject missing node / definitively-not-ready probe), and passes tests. Be careful to frame as robust probe-failure handling (general benefit). |
| `2fc0e53` | fix: allow H3 SLA render when runtime probe cannot be inspected (+ server gate) | **upstream-candidate (evaluate)** | Same pattern for SLA (`lib/h3-sla-attention.js`, server). Same validation/caution as `5a8cd1a`. Note server structure/guard locations differ from our merged fork — adapt carefully, do not copy fork-only surrounding code. |
| `faf9e4b` | docs: align README with low-vram and installer copy tests | **fork-only** | README on this fork contains Intel-fork-specific content and differs substantially from upstream; not upstreamable as-is. Skip for `upstream-intel`. |

**Important:** classifications are hypotheses. The agent must confirm applicability against the actual `upstream/main` checkout and report decisions with reasoning.

## Step-by-step (sandboxed)

### 1) Prepare (in main checkout, read-only intent)
```bash
cd "E:\Mix Studio"   # PowerShell (or /e/Mix\ Studio in Git Bash)
git fetch --all --prune
git status --short --branch   # must be clean before creating worktrees
git rev-parse --short upstream/main fork/merge/upstream-smart-3d HEAD
```
Record these SHAs in your report.

### 2) Create sandbox worktree for upstream-intel
```bash
# from E:\Mix Studio
git worktree add ../MixStudio-upstream-intel upstream/main -b upstream-intel
```
Do **all** upstream-candidate work inside `../MixStudio-upstream-intel`. Do not modify the main checkout.

### 3) Port/evaluate candidates (inside ../MixStudio-upstream-intel)

Work one logical change at a time. Prefer small, reviewable commits.

**A. `5a8cd1a` (Sage probe-inspection tolerance) — evaluate/port**
- Inspect `lib/sage-attention.js` and the H3 Sage render/availability gating in `server.js` on `upstream/main`.
- Goal: distinguish “probe could not be inspected” (e.g. verification threw) from “probe inspected and not ready”; allow render when the required node is present and the probe was **not inspectable**, but **still reject** when the node is missing or when an inspected probe reports not ready.
- Reproduce the intent of commit `5a8cd1a` adapted to upstream code structure (do not blindly cherry-pick if contexts differ; manual port is fine). Example to inspect source change: `git -C "E:\Mix Studio" show 5a8cd1a`.
- Commit only if the adapted change is correct for upstream, minimal, and validated.

**B. `2fc0e53` (SLA probe-inspection tolerance) — evaluate/port**
- Same approach for `lib/h3-sla-attention.js` and the H3 SLA gate in `server.js`. Inspect source: `git -C "E:\Mix Studio" show 2fc0e53`.
- Keep behavior consistent with (A). Commit separately if appropriate.

**C. `3cf2d0e` (prompt provider test dedupe) — verify, likely skip**
- Check `/api/prompt/provider/test` in `server.js` on `upstream/main` for duplicate `instruction`/`userInput`/`maxTokens` args (source: `git -C "E:\Mix Studio" show 3cf2d0e`).
- If not present, **skip** and record “not applicable on upstream/main@<sha>”.
- If present, apply the minimal dedupe fix and validate.

**D. `faf9e4b` / `aa598bb` / `c5b90c3` — do not port**
- Record as skipped with reasons per table above.

> If a cherry-pick conflicts heavily or pulls in fork-only context, stop and prefer a clean manual port of just the upstream-relevant logic. Never resolve by importing fork-specific behavior.

### 4) Validate (inside ../MixStudio-upstream-intel)
Run and capture full results:
```bash
node --check server.js
node --check public/app.js
node --test
```
- If you touched specific areas, also run targeted tests (e.g. any H3/SLA/Sage/external-llm related suites present on `upstream/main`) and report them, but **full `node --test` is preferred**.
- If any test fails, determine whether it’s caused by your change or pre-existing on that base; **do not claim success**. Fix only issues introduced by your port, or revert that port and report.
- Manual ComfyUI/runtime smoke is environment-dependent; if not run, state that explicitly.

### 5) Report (required evidence)
Provide a concise report including:
- Base: `upstream/main` SHA used, and SHAs of source commits inspected (`faf9e4b, 2fc0e53, 5a8cd1a, 3cf2d0e, c5b90c3, aa598bb`).
- Sandbox: worktree path and branch (`../MixStudio-upstream-intel`, `upstream-intel`).
- Per-commit disposition: **ported** (new commit hash + files/stat) / **skipped-not-applicable** / **skipped-fork-only** with reasoning.
- Validation evidence: exact commands run and outcomes (pass/fail counts, exit codes). Include `node --check` and `node --test` summaries.
- `git log --oneline --decorate -n 10` and `git diff --stat upstream/main..HEAD` from the sandbox branch.
- Risks/notes: any behavioral gating changes (especially Sage/SLA probe handling), and anything needing human/runtime verification.
- Confirmation: **no push, no PR opened, no merge performed, no changes to main checkout or existing fork branches**.

### 6) Cleanup guidance (for human, after review)
- Keep the worktree/branch until reviewed. To discard if not wanted: `git worktree remove ../MixStudio-upstream-intel` and `git branch -D upstream-intel` (only if unpushed/throwaway).
- To share for review: push `upstream-intel` to `fork` (with approval) and open a PR (CodeRabbit can review). **Do not target upstream without explicit human approval and a final scope check (upstream-aligned only).**

## OpenCode / Codex usage notes (for the operator)

- **OpenCode:** run in the sandbox workdir (e.g. `opencode run '...'` from `../MixStudio-upstream-intel`, or an interactive session scoped to that path). Ensure its provider/model is configured (OpenCode Go plan or a local model endpoint) and it has shell/git access.
- **Codex:** run inside the sandbox git repo/workdir (e.g. `codex exec --sandbox workspace-write '...'` with `pty=true` as appropriate per your environment), with auth configured.
- Give the agent this file + `docs/agent-dev-workflow.md`, and require the evidence checklist above. Prefer one commit per logical fix.

## Out of scope
- Pushing to any remote, opening PRs, merging branches, or modifying `fork/main` / `merge/upstream-smart-3d`.
- Porting fork-only features, Intel-fork branding/docs, or integration-only merge/test glue.
- Rewriting history of existing shared branches.
