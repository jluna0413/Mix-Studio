'use strict';

// A ComfyUI prompt can stop emitting progress while it is still "running":
// the level_zero backend wedges (UR_RESULT_ERROR_DEVICE_LOST), a sampler step
// takes minutes per iteration on CPU fallback, or the WebSocket dies silently.
// Mark those jobs as failed so the queue cannot sit at 0% forever.

const JOB_STALL_MS = 20 * 60 * 1000;
const JOB_QUEUE_STALL_MS = 30 * 60 * 1000;

function numericTime(value) {
  const n = Number(value);
  return Number.isFinite(n) && n > 0 ? n : null;
}

function jobActivityMs(job, now = Date.now()) {
  const last = numericTime(job?.lastActivityAt) || numericTime(job?.startedAt)
    || numericTime(job?.enqueuedAt) || numericTime(job?.createdAt);
  if (!last) return null;
  return Math.max(0, now - last);
}

function shouldFailStalledJob(job, now = Date.now(), options = {}) {
  if (!job || typeof job !== 'object') return false;
  if (job.stallWatchdogDisabled === true) return false;
  if (job.completing || job.cancelRequested) return false;
  // Smart runs have their own stall/recovery path and must not be double-failed.
  if (job.smartRunId) return false;
  const last = numericTime(job.lastActivityAt);
  if (!last) return false;
  const idle = now - last;
  const queued = !numericTime(job.startedAt) && numericTime(job.enqueuedAt) > 0;
  const limit = queued ? (options.queueStallMs ?? JOB_QUEUE_STALL_MS) : (options.stallMs ?? JOB_STALL_MS);
  return idle > limit;
}

function findStalledJobs(jobs, now = Date.now(), options = {}) {
  const map = jobs instanceof Map ? jobs : new Map();
  const stalled = [];
  for (const [id, job] of map) {
    if (shouldFailStalledJob(job, now, options)) stalled.push({ id, job, idleMs: now - Number(job.lastActivityAt) });
  }
  return stalled;
}

function stalledJobMessage(idleMs) {
  const minutes = Math.max(1, Math.round(idleMs / 60000));
  return `Generation stopped reporting progress for ${minutes} minutes and was marked failed. `
    + 'If this repeats, restart ComfyUI from Settings → Generation setup.';
}

module.exports = {
  JOB_QUEUE_STALL_MS,
  JOB_STALL_MS,
  findStalledJobs,
  jobActivityMs,
  shouldFailStalledJob,
  stalledJobMessage,
};
