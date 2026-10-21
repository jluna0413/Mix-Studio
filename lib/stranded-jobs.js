'use strict';

/**
 * Stranded-job detection.
 *
 * A Mix Studio job can be left behind in the server's `jobs` Map when ComfyUI
 * dies mid-generation (GPU device loss, desktop crash) and restarts without
 * history. ComfyUI's `/history/<prompt_id>` then returns `{}` forever, so the
 * normal reconciliation loop never sees a terminal state and never clears the
 * job.
 *
 * The consequence is worse than a stale row: `assertDesktopIsIdle` refuses any
 * dependency/setup change while `jobs.size` is non-zero, so one crashed job
 * blocks ComfyUI reconnection, model installs, and path changes with a
 * misleading "Wait for the Mix Studio queue to finish" error.
 *
 * A job is only considered stranded when ALL of the following hold:
 *   - ComfyUI is reachable and reports a healthy queue
 *   - the prompt is absent from both the running and pending queues
 *   - ComfyUI has no history entry for the prompt
 *   - the job has been in that state for longer than the grace period
 *
 * The grace period matters: a job that was just submitted has not appeared in
 * ComfyUI's queue yet, and a job whose output is still being written has not
 * appeared in history yet. Reaping either would kill live work.
 */

const DEFAULT_STRANDED_JOB_GRACE_MS = 90_000;

/**
 * ComfyUI prompt ids are normally `ks-<uuid>`, but some builds and older
 * releases emit a bare uuid. Accept both shapes so a stranded job is still
 * recognised; the id only has to be a plausible prompt identifier.
 */
function isComfyPromptId(value) {
  if (typeof value !== 'string') return false;
  const id = value.trim();
  if (!id) return false;
  return /^ks-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(id)
    || /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(id);
}

/**
 * Decide whether a tracked job may be reaped as stranded.
 *
 * @param {object} job                the tracked job
 * @param {object} state
 * @param {boolean} state.comfyReachable   ComfyUI answered the queue/history probe
 * @param {Set<string>} state.queuedIds    prompt ids currently running or pending
 * @param {boolean} state.hasHistory       ComfyUI still has a history entry for the prompt
 * @param {number} [state.now]
 * @param {number} [state.graceMs]
 * @returns {{reap: boolean, reason: string}}
 */
function strandedJobVerdict(job, state = {}) {
  if (!job) return { reap: false, reason: 'no job' };
  const now = Number(state.now) || Date.now();
  const graceMs = Number.isFinite(Number(state.graceMs))
    ? Number(state.graceMs)
    : DEFAULT_STRANDED_JOB_GRACE_MS;

  // A job that is still finalizing may be mid-download of its outputs.
  if (job.completing) return { reap: false, reason: 'job is still finalizing' };

  const promptId = String(job.promptId || job.pid || '');
  if (!isComfyPromptId(promptId)) return { reap: false, reason: 'job has no ComfyUI prompt id' };

  if (state.comfyReachable === false) return { reap: false, reason: 'ComfyUI is unreachable' };
  if (state.hasHistory) return { reap: false, reason: 'ComfyUI still has history for this prompt' };
  if (state.queuedIds instanceof Set && state.queuedIds.has(promptId)) {
    return { reap: false, reason: 'prompt is still in the ComfyUI queue' };
  }

  const base = Number(job.startedAt || job.enqueuedAt || job.createdAt) || now;
  const ageMs = now - base;
  if (ageMs < graceMs) {
    return { reap: false, reason: `job is only ${Math.round(ageMs / 1000)}s old` };
  }

  return { reap: true, reason: 'ComfyUI no longer knows this prompt and it is not queued' };
}

module.exports = {
  DEFAULT_STRANDED_JOB_GRACE_MS,
  isComfyPromptId,
  strandedJobVerdict,
};
