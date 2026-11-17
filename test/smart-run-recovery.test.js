'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');

const {
  interruptedSmartJobIds,
  markSmartRunsInterruptedForRecovery,
  prepareSmartRunResume,
} = require('../lib/smart-run-recovery');

test('restart recovery pauses only resumable Smart runs and preserves completed work', () => {
  const runs = [
    {
      id: 'active', status: 'running', updatedAt: 1, error: '',
      steps: [
        { id: 'done', status: 'complete', jobId: null },
        { id: 'active-step', status: 'running', jobId: 'prompt-1', progress: { completed: 1, total: 3 } },
        { id: 'next', status: 'pending', jobId: null },
      ],
    },
    { id: 'ready', status: 'ready', steps: [{ id: 'first', status: 'pending' }] },
    { id: 'review', status: 'review', steps: [{ id: 'next', status: 'pending' }] },
    { id: 'complete', status: 'complete', steps: [{ id: 'done', status: 'complete' }] },
  ];
  assert.deepEqual(markSmartRunsInterruptedForRecovery(runs, 42), ['active', 'ready']);
  assert.equal(runs[0].status, 'attention');
  assert.equal(runs[0].steps[0].status, 'complete');
  assert.equal(runs[0].steps[1].status, 'attention');
  assert.equal(runs[0].steps[2].status, 'pending');
  assert.equal(runs[0].updatedAt, 42);
  assert.equal(runs[2].status, 'review');
  assert.equal(runs[3].status, 'complete');
});

test('restart recovery retires the old prompt before making its step pending again', () => {
  const run = {
    id: 'active', status: 'attention', error: 'Interrupted',
    steps: [
      { id: 'done', status: 'complete', jobId: null },
      { id: 'active-step', status: 'attention', jobId: 'prompt-1', progress: { completed: 2, total: 3 }, error: 'Interrupted' },
      { id: 'next', status: 'pending', jobId: null },
    ],
  };
  assert.deepEqual(interruptedSmartJobIds(run), ['prompt-1']);
  assert.equal(prepareSmartRunResume(run), true);
  assert.equal(run.status, 'running');
  assert.equal(run.error, '');
  assert.equal(run.steps[0].status, 'complete');
  assert.deepEqual(run.steps[1], {
    id: 'active-step', status: 'pending', jobId: null, progress: null, startedAt: null, error: '',
  });
  assert.equal(run.steps[2].status, 'pending');
  assert.equal(prepareSmartRunResume(run), false, 'resume preparation is idempotent');
});

test('markStalledSmartRuns flags only long-running steps for recovery', () => {
  const { markStalledSmartRuns, SMART_STALL_MS } = require('../lib/smart-run-recovery');
  const now = 1_000_000;
  const runs = [{
    id: 'stale', status: 'running',
    steps: [
      { id: 'done', kind: 'reference', status: 'complete' },
      { id: 'hung', kind: 'video', status: 'running', startedAt: now - SMART_STALL_MS - 1000 },
    ],
  }, {
    id: 'fresh', status: 'running',
    steps: [{ id: 'ok', kind: 'video', status: 'running', startedAt: now - 1000 }],
  }, {
    id: 'finished', status: 'complete',
    steps: [{ id: 'old', kind: 'video', status: 'complete', startedAt: now - SMART_STALL_MS - 5000 }],
  }];
  assert.deepEqual(markStalledSmartRuns(runs, now), ['stale']);
  assert.equal(runs[0].status, 'attention');
  assert.equal(runs[0].steps[1].status, 'attention');
  assert.match(runs[0].error, /stopped reporting progress/);
  assert.equal(runs[1].status, 'running');
  assert.equal(runs[2].status, 'complete');
});

test('prepareSmartRunResume clears the stall timestamp so recovery cannot loop immediately', () => {
  const { prepareSmartRunResume } = require('../lib/smart-run-recovery');
  const run = {
    status: 'attention',
    steps: [{ id: 's', status: 'attention', jobId: 'p1', startedAt: 123, progress: { completed: 1, total: 8 }, error: 'x' }],
  };
  assert.equal(prepareSmartRunResume(run), true);
  assert.equal(run.steps[0].startedAt, null);
  assert.equal(run.steps[0].jobId, null);
});
