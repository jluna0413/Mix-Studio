'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');


test('job stall watchdog fails only silent non-smart jobs past the threshold', () => {
  const { findStalledJobs, shouldFailStalledJob, JOB_STALL_MS } = require('../lib/job-stall-watchdog');
  const now = 100_000_000;
  const jobs = new Map([
    ['hung', { kind: 'video', enqueuedAt: now - JOB_STALL_MS - 2000, startedAt: now - JOB_STALL_MS - 1000, lastActivityAt: now - JOB_STALL_MS - 1 }],
    ['fresh', { kind: 'video', enqueuedAt: now - 1000, startedAt: now - 2000, lastActivityAt: now - 30_000 }],
    ['smart', { kind: 'video', smartRunId: 'run1', startedAt: now - 5000, lastActivityAt: now - JOB_STALL_MS - 1 }],
    ['completing', { kind: 'video', completing: true, startedAt: now - JOB_STALL_MS - 1000, lastActivityAt: now - JOB_STALL_MS - 1 }],
  ]);
  assert.deepEqual(findStalledJobs(jobs, now).map((entry) => entry.id), ['hung']);
  assert.equal(shouldFailStalledJob(jobs.get('fresh'), now), false);
  assert.equal(shouldFailStalledJob(jobs.get('smart'), now), false);
});

test('queued jobs use a longer stall window than running jobs', () => {
  const { shouldFailStalledJob, JOB_QUEUE_STALL_MS } = require('../lib/job-stall-watchdog');
  const now = 100_000_000;
  const queued = { kind: 'gen', enqueuedAt: now - JOB_QUEUE_STALL_MS - 1, startedAt: null, lastActivityAt: now - JOB_QUEUE_STALL_MS - 1 };
  assert.equal(shouldFailStalledJob(queued, now), true);
  const runningLong = { kind: 'gen', startedAt: now - 60_000, lastActivityAt: now - 60_000 };
  assert.equal(shouldFailStalledJob(runningLong, now, { stallMs: 60_000 + 1 }), false);
});
