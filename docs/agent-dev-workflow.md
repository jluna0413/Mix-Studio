# Agent Dev Workflow (sandboxed, reviewable)

This document defines a **safe, non-destructive workflow** for using coding agents (OpenCode, Codex, etc.) on Mix Studio, with **CodeRabbit** as an automated PR reviewer. It is designed so a non-coder can supervise: agents do isolated work, produce evidence, and humans review before anything “goes live.”

> **Scope note (this fork):** `jluna0413/Mix-Studio` is an Intel/XPU-oriented fork that may diverge from upstream. **Upstream-aligned fixes** (general/Intel-compat, broadly useful) are candidates to contribute to [BlackMixture/Mix-Studio](https://github.com/BlackMixture/Mix-Studio). **Fork-only functionality/direction** stays in this fork. Keep these streams separate (see “Promotion paths”).

## Roles

| Role | Tool | Responsibility |
|---|---|---|
| **Implementer** | OpenCode / Codex / other coding agent | Works in an isolated sandbox (worktree/branch), makes changes, runs checks, and reports evidence. Does **not** merge or push to protected branches. |
| **PR reviewer (automated)** | CodeRabbit (GitHub) | Reviews pull requests for bugs/risks/style and summarizes changes. Reviews PRs — it does **not** run your local environment or do commits/merges. |
| **Reviewer/approver (human)** | You | Reviews diffs, commit hashes, and test results; approves merges/pushes; decides upstream vs fork-only. |

**Trust rule:** treat agent claims as self-reports until backed by evidence (commit hashes, `git log/diff`, and real command output like `node --check` / `node --test` results).

## Safety guardrails (non-negotiable)

- **Never work directly on production/mainline branches.** No direct edits/commits on `fork/main`, `main`, or `upstream/*`.
- **Isolate all agent work** in a dedicated git worktree and/or a dedicated branch (see below). One task/concern per branch.
- **No force-push, no history rewrite** on shared branches. No destructive git operations (`reset --hard` on shared refs, deleting remote branches, etc.) unless explicitly approved for a throwaway sandbox branch.
- **No secrets in commits.** Never commit `.env`, tokens, credentials, or private data (see `AGENTS.md`).
- **No destructive testing against live data.** Do not run destructive routes/tests against real `data/`/live gallery state; use fixtures/throwaway data only.
- **Validate before proposing merge:** run required checks and capture output.
- **Human approval before promotion:** you review and explicitly approve any push to `fork`, any PR, and any merge to `fork/main` or upstream.

## Sandbox setup (git worktrees)

A **worktree** is a separate working folder checked out to its own branch while sharing the same repo history — ideal for safe experiments and parallel/agent work. Deleting a worktree/branch does not affect your main checkout.

Run from the main repo: `E:\Mix Studio` (PowerShell: `cd "E:\Mix Studio"`; Git Bash: `cd /e/Mix\ Studio`).

```bash
# 1) Make sure refs are current
git fetch --all --prune

# 2) Create an isolated sandbox worktree on a NEW branch (example: from current fork integration)
git worktree add ../MixStudio-sandbox -b sandbox/<task-name> fork/merge/upstream-smart-3d

#    — or, to base on upstream/main for an upstream-targeted fix:
# git worktree add ../MixStudio-upstream-fix -b intel/<fix-name> upstream/main

# 3) Point the agent at the sandbox path and have it work ONLY there
#    (e.g. ../MixStudio-sandbox)

# 4) When finished (after you’ve reviewed/saved what you need)
git worktree remove ../MixStudio-sandbox
# and delete the branch if it was throwaway and not needed:
# git branch -D sandbox/<task-name>
```

Tips:
- Keep the main checkout (`E:\Mix Studio`) on your known-good branch; do agent work in the sibling worktree folder.
- Use a fresh branch per task; avoid mixing unrelated changes.
- If an experiment goes bad, you can discard the sandbox worktree/branch without touching validated work.

## Validation (required evidence)

For Mix Studio changes, the implementer must run and report results (from the sandbox worktree):

```bash
node --check server.js
node --check public/app.js   # if frontend changed
node --test                  # full suite (preferred) or targeted tests relevant to the change
```

**Evidence to capture for review:**
- Base ref and branch name, and final commit hash(es): `git log --oneline --decorate -n 5`
- Exact diff summary: `git show --stat <commit>` (and/or `git diff --stat <base>..HEAD`)
- Full command output (or clear pass/fail counts) for the checks/tests run
- Notes on anything not run and why (e.g. manual/ComfyUI-dependent smoke not possible headless)

A task is **not** considered done without this evidence.

## Promotion paths

### A) Upstream-aligned Intel fix (candidate for BlackMixture/Mix-Studio)
- Branch from `upstream/main` (e.g. `upstream-intel/<topic>` or a single `upstream-intel` prep branch).
- Include **only** changes that are generally useful/upstream-aligned (Intel-compat or general bugfixes). No fork-only features/branding/direction.
- Keep commits focused and reviewable; include/adjust tests as needed.
- Validate against `upstream/main` context (`node --check`, `node --test`).
- Push the branch to `fork` and open a PR to the appropriate upstream target (usually `BlackMixture/Mix-Studio` `main`) **after human review**. Expect CodeRabbit (if enabled) and maintainer review.

### B) Fork-only / divergent work (stays in this fork)
- Keep on fork branches (e.g. `merge/upstream-smart-3d`, `fork/main`, or feature branches).
- May include fork-specific behavior, Intel-fork docs/UX, and experimental direction not intended upstream.
- Still use sandbox + validation + human review before merging to `fork/main` or deploying.

### C) Integration
- Periodically merge/rebase upstream changes into the fork integration line, resolve conflicts in a sandbox/worktree, revalidate, then update the fork branch. Keep upstream-candidate fixes separable where possible.

## CodeRabbit (PR review) in the tandem

- **What it does:** automated review on GitHub PRs (findings, summaries). Enable it on the repo and/or add `.coderabbit.yaml` (this repo includes a starter config) to tune behavior.
- **When to use it:** open a PR from a pushed branch (to `fork` or upstream) after local validation — let CodeRabbit review, then you decide.
- **What it does not replace:** local checks/tests, runtime smoke validation, or human judgment. Treat its comments as review input, not automatic approval.
- **Typical flow:** agent prepares & validates in sandbox → you review evidence → push branch to fork → open PR → CodeRabbit + human review → address feedback → merge when approved.

## Recommended agent choices (this environment)

- **OpenCode** (`opencode`): good fit if you have an OpenCode plan or a local-model endpoint configured; provider-agnostic CLI. Use `opencode run` for bounded tasks or the TUI in a PTY for iterative work. Prefer a scoped `workdir`/worktree per task.
- **Codex** (`codex`): strong for bounded coding tasks in a git repo (typically OpenAI auth/API). Use `codex exec` in a scoped workdir/worktree.

Whichever you use: give it this workflow + the task handoff spec, require it to stay inside the sandbox, and require the evidence checklist above.

## Quick checklist (per task)

- [ ] Sandbox created (worktree + new branch) from the correct base
- [ ] Agent changes limited to task scope
- [ ] `node --check server.js` (and `public/app.js` if touched) pass
- [ ] `node --test` (or agreed targeted tests) pass — results captured
- [ ] Diff/stat and commit hash(es) captured
- [ ] Human review complete
- [ ] (If sharing) pushed to fork branch and PR opened; CodeRabbit/human feedback addressed
- [ ] Merged/promoted only with explicit approval; fork-only vs upstream-candidate kept separate
