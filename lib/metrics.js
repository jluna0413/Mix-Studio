'use strict';

/*
 * Metrics system for Mix Studio (observability audit, dimension 5).
 *
 * Two surfaces:
 *   1. Durable JSON-line log in logs/metrics.log (every record* call also
 *      emits a log.metric line for time-series querying).
 *   2. In-memory rolling aggregates exposed via snapshot() for a future
 *      /api/metrics endpoint (per-engine success/duration/failure-reason,
 *      queue depth/wait, SSE connection count, stalled job count, VRAM).
 *
 * No external dependencies. Safe to call from any request path; all mutators
 * are O(1) or bounded.
 */

const log = require('./log');

const WINDOW_MS = 10 * 60 * 1000; // rolling 10-minute window for aggregates

// --- in-memory state -------------------------------------------------------
const engineRuns = new Map();   // engine -> { total, ok, fail, failReasons{}, durations[] }
const gauges = new Map();       // name -> last numeric value
let recentRuns = [];            // { ts, engine, kind, success, durationMs, failureReason }

function now() { return Date.now(); }

function pruneRuns() {
  const cutoff = now() - WINDOW_MS;
  while (recentRuns.length && recentRuns[0].ts < cutoff) recentRuns.shift();
}

function recordEngineRun(opts) {
  const { engine = 'unknown', kind = 'gen', success = true, durationMs = 0, failureReason = null } = opts || {};
  log.metric('engine.run', durationMs, {
    engine, kind, success: success ? 1 : 0, failureReason: success ? '' : (failureReason || 'unknown'),
  });
  let agg = engineRuns.get(engine);
  if (!agg) {
    agg = { total: 0, ok: 0, fail: 0, failReasons: {}, durations: [] };
    engineRuns.set(engine, agg);
  }
  agg.total += 1;
  if (success) agg.ok += 1; else { agg.fail += 1; }
  if (!success && failureReason) {
    agg.failReasons[failureReason] = (agg.failReasons[failureReason] || 0) + 1;
  }
  agg.durations.push(durationMs);
  if (agg.durations.length > 200) agg.durations.shift();
  recentRuns.push({ ts: now(), engine, kind, success, durationMs, failureReason: success ? null : failureReason });
  pruneRuns();
}

function recordQueueDepth(depth, waitMs) {
  log.metric('queue.depth', depth, { waitMs: waitMs || 0 });
  gauges.set('queue.depth', depth);
  if (waitMs) gauges.set('queue.waitMs', waitMs);
}

function recordSseConnections(count) {
  log.metric('sse.connections', count);
  gauges.set('sse.connections', count);
}

function recordStalledJobs(count) {
  log.metric('jobs.stalled', count);
  gauges.set('jobs.stalled', count);
}

function recordVram(usedMb, totalMb, utilizationPct) {
  log.metric('vram.usedMb', usedMb, { totalMb: totalMb || 0, utilizationPct: utilizationPct == null ? 0 : utilizationPct });
  if (usedMb != null) gauges.set('vram.usedMb', usedMb);
  if (totalMb != null) gauges.set('vram.totalMb', totalMb);
  if (utilizationPct != null) gauges.set('vram.utilizationPct', utilizationPct);
}

function recordCounter(name, value, tags) {
  log.metric(name, value, Object.assign({ _kind: 'counter' }, tags || {}));
}

function recordGauge(name, value, tags) {
  log.metric(name, value, Object.assign({ _kind: 'gauge' }, tags || {}));
  gauges.set(name, value);
}

function percentile(sortedArr, p) {
  if (!sortedArr.length) return 0;
  const idx = Math.min(sortedArr.length - 1, Math.max(0, Math.floor((p / 100) * sortedArr.length)));
  return sortedArr[idx];
}

function snapshot() {
  pruneRuns();
  const engines = {};
  for (const [engine, agg] of engineRuns) {
    const sorted = [...agg.durations].sort((a, b) => a - b);
    engines[engine] = {
      total: agg.total,
      ok: agg.ok,
      fail: agg.fail,
      successRate: agg.total ? +(agg.ok / agg.total).toFixed(4) : 1,
      durationMs: {
        avg: sorted.length ? Math.round(sorted.reduce((s, v) => s + v, 0) / sorted.length) : 0,
        p50: percentile(sorted, 50),
        p95: percentile(sorted, 95),
        max: sorted.length ? sorted[sorted.length - 1] : 0,
      },
      failureReasons: agg.failReasons,
    };
  }
  return {
    generatedAt: new Date().toISOString(),
    windowMs: WINDOW_MS,
    engines,
    gauges: Object.fromEntries(gauges),
    recentRuns: recentRuns.slice(-50),
  };
}

module.exports = {
  recordEngineRun,
  recordQueueDepth,
  recordSseConnections,
  recordStalledJobs,
  recordVram,
  recordCounter,
  recordGauge,
  snapshot,
  WINDOW_MS,
};
