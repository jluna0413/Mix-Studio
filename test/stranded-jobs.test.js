'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');

const {
  DEFAULT_STRANDED_JOB_GRACE_MS,
  isComfyPromptId,
  strandedJobVerdict,
} = require('../lib/stranded-jobs');

const PROMPT_ID = 'ks-1f2e3d4c-5b6a-7988-9abc-def012345678';

test('stranded job detection recognises both prompt id shapes', () => {
  assert.equal(isComfyPromptId(PROMPT_ID), true);
  assert.equal(isComfyPromptId('d050beae-feda-40f0-bc2e-c82f5283846a'), true);
  assert.equal(isComfyPromptId('not-a-prompt'), false);
  assert.equal(isComfyPromptId(''), false);
  assert.equal(isComfyPromptId(undefined), false);
  assert.equal(isComfyPromptId(42), false);
});

test('a job ComfyUI no longer tracks is reaped once it is old enough', () => {
  const verdict = strandedJobVerdict(
    { promptId: PROMPT_ID, startedAt: 1_000 },
    { comfyReachable: true, queuedIds: new Set(), hasHistory: false, now: 1_000 + DEFAULT_STRANDED_JOB_GRACE_MS + 1 },
  );
  assert.equal(verdict.reap, true);
});

test('a young job is never reaped even when ComfyUI has forgotten it', () => {
  const verdict = strandedJobVerdict(
    { promptId: PROMPT_ID, startedAt: 1_000 },
    { comfyReachable: true, queuedIds: new Set(), hasHistory: false, now: 1_000 + 5_000 },
  );
  assert.equal(verdict.reap, false);
  assert.match(verdict.reason, /old/);
});

test('a job still in the ComfyUI queue is left alone', () => {
  const verdict = strandedJobVerdict(
    { promptId: PROMPT_ID, startedAt: 1_000 },
    {
      comfyReachable: true,
      queuedIds: new Set([PROMPT_ID]),
      hasHistory: false,
      now: 1_000 + DEFAULT_STRANDED_JOB_GRACE_MS * 10,
    },
  );
  assert.equal(verdict.reap, false);
  assert.match(verdict.reason, /queue/);
});

test('a job whose output ComfyUI still has in history is left alone', () => {
  const verdict = strandedJobVerdict(
    { promptId: PROMPT_ID, startedAt: 1_000 },
    {
      comfyReachable: true,
      queuedIds: new Set(),
      hasHistory: true,
      now: 1_000 + DEFAULT_STRANDED_JOB_GRACE_MS * 10,
    },
  );
  assert.equal(verdict.reap, false);
  assert.match(verdict.reason, /history/);
});

test('nothing is reaped while ComfyUI is unreachable', () => {
  const verdict = strandedJobVerdict(
    { promptId: PROMPT_ID, startedAt: 1_000 },
    {
      comfyReachable: false,
      queuedIds: new Set(),
      hasHistory: false,
      now: 1_000 + DEFAULT_STRANDED_JOB_GRACE_MS * 10,
    },
  );
  assert.equal(verdict.reap, false);
  assert.match(verdict.reason, /unreachable/);
});

test('a job that is still finalizing is never reaped', () => {
  const verdict = strandedJobVerdict(
    { promptId: PROMPT_ID, startedAt: 1_000, completing: true },
    {
      comfyReachable: true,
      queuedIds: new Set(),
      hasHistory: false,
      now: 1_000 + DEFAULT_STRANDED_JOB_GRACE_MS * 10,
    },
  );
  assert.equal(verdict.reap, false);
  assert.match(verdict.reason, /finalizing/);
});

test('a job without a ComfyUI prompt id is never reaped', () => {
  const verdict = strandedJobVerdict(
    { startedAt: 1_000 },
    {
      comfyReachable: true,
      queuedIds: new Set(),
      hasHistory: false,
      now: 1_000 + DEFAULT_STRANDED_JOB_GRACE_MS * 10,
    },
  );
  assert.equal(verdict.reap, false);
  assert.match(verdict.reason, /prompt id/);
});
