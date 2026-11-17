'use strict';

const RECOVERABLE_RUN_STATUSES = new Set(['ready', 'running', 'queueing', 'attention']);
const INTERRUPTED_STEP_STATUSES = new Set(['running', 'queueing', 'attention']);

function smartRunHasWork(run) {
  return Array.isArray(run?.steps)
    && run.steps.some((step) => ['pending', 'running', 'queueing', 'attention'].includes(step?.status));
}

function markSmartRunsInterruptedForRecovery(runs, now = Date.now()) {
  const ids = [];
  for (const run of Array.isArray(runs) ? runs : []) {
    if (!run || !RECOVERABLE_RUN_STATUSES.has(run.status) || !smartRunHasWork(run)) continue;
    run.status = 'attention';
    run.error = 'Mix Studio restarted while this production was active. Reconnecting to ComfyUI and resuming automatically…';
    run.updatedAt = now;
    for (const step of run.steps) {
      if (INTERRUPTED_STEP_STATUSES.has(step?.status)) step.status = 'attention';
    }
    ids.push(run.id);
  }
  return ids;
}

function interruptedSmartJobIds(run) {
  return [...new Set((Array.isArray(run?.steps) ? run.steps : [])
    .filter((step) => INTERRUPTED_STEP_STATUSES.has(step?.status) && step.jobId)
    .map((step) => String(step.jobId))
    .filter(Boolean))];
}

function prepareSmartRunResume(run) {
  if (!run || run.status !== 'attention') return false;
  for (const step of Array.isArray(run.steps) ? run.steps : []) {
    if (!INTERRUPTED_STEP_STATUSES.has(step?.status)) continue;
    step.status = 'pending';
    step.jobId = null;
    step.progress = null;
    step.startedAt = null;
    step.error = '';
  }
  run.status = 'running';
  run.error = '';
  return true;
}

const SMART_STALL_MS = 15 * 60 * 1000;

function markStalledSmartRuns(runs, now = Date.now()) {
  const ids = [];
  for (const run of Array.isArray(runs) ? runs : []) {
    if (!run || !['running', 'queueing'].includes(run.status)) continue;
    const stalled = (Array.isArray(run.steps) ? run.steps : []).find((step) => (
      ['running', 'queueing'].includes(step?.status)
      && Number.isFinite(Number(step?.startedAt))
      && now - Number(step.startedAt) > SMART_STALL_MS
    ));
    if (!stalled) continue;
    run.status = 'attention';
    run.error = `A ${stalled.kind} step stopped reporting progress after ${Math.round(SMART_STALL_MS / 60000)} minutes. ComfyUI will be checked and the step resumed.`;
    run.updatedAt = now;
    for (const step of run.steps) {
      if (INTERRUPTED_STEP_STATUSES.has(step?.status)) step.status = 'attention';
    }
    ids.push(run.id);
  }
  return ids;
}

module.exports = {
  SMART_STALL_MS,
  interruptedSmartJobIds,
  markSmartRunsInterruptedForRecovery,
  markStalledSmartRuns,
  prepareSmartRunResume,
  smartRunHasWork,
};
