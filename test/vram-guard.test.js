'use strict';

/* Tests the VRAM guard logic in lib/queue-health.js. The guard detects
 * the "stuck VRAM after DEVICE_LOST" pattern: high VRAM usage + ~0 GPU
 * utilization sustained for > 45s. On your Intel Arc Pro B70, a crashed
 * ComfyUI worker can survive holding ~30 GB, which is exactly the scenario
 * this prevents. */

const test = require('node:test');
const assert = require('node:assert/strict');
const { assessVramStuck, VRAM_STUCK_THRESHOLD_MB, VRAM_STUCK_SUSTAINED_MS } = require('../lib/queue-health');

test('assessVramStuck returns false when VRAM is healthy', () => {
  const gpu = { utilization: 50, memoryUsedMb: 8000, memoryTotalMb: 32000 };
  const state = { vramStuckSince: null };
  const result = assessVramStuck(gpu, state, 1000);
  assert.equal(result.vramStuck, false);
  assert.equal(result.vramStuckSince, null); // idle + healthy resets the timer
});

test('assessVramStuck detects stuck VRAM (high usage + idle)', () => {
  // Simulate: 30 GB used, 1.16 GB free, GPU idle
  const gpu = { utilization: 2, memoryUsedMb: 29700, memoryTotalMb: 31160 };
  let state = { vramStuckSince: null };
  const t0 = 1000;

  // First check: not sustained yet
  let result = assessVramStuck(gpu, state, t0);
  assert.equal(result.vramStuck, false, 'must not trigger immediately');
  state = { vramStuckSince: result.vramStuckSince };

  // Second check: past the sustain threshold
  result = assessVramStuck(gpu, state, t0 + VRAM_STUCK_SUSTAINED_MS + 1000);
  assert.equal(result.vramStuck, true, 'must trigger after sustained duration');
  assert.ok(result.vramStuckSince, 'must track when the stuck state started');
});

test('assessVramStuck resets when VRAM recovers', () => {
  const stuckGpu = { utilization: 2, memoryUsedMb: 29700, memoryTotalMb: 31160 };
  let state = { vramStuckSince: null };
  const t0 = 1000;

  let result = assessVramStuck(stuckGpu, state, t0);
  state = { vramStuckSince: result.vramStuckSince };
  result = assessVramStuck(stuckGpu, state, t0 + VRAM_STUCK_SUSTAINED_MS + 1000);
  assert.equal(result.vramStuck, true);

  // Now VRAM is freed (restart happened, or process released)
  const healthyGpu = { utilization: 80, memoryUsedMb: 3000, memoryTotalMb: 31160 };
  state = { vramStuckSince: result.vramStuckSince };
  result = assessVramStuck(healthyGpu, state, t0 + VRAM_STUCK_SUSTAINED_MS + 5000);
  assert.equal(result.vramStuck, false);
  assert.equal(result.vramStuckSince, null, 'must clear the timer when VRAM recovers');
});

test('assessVramStuck ignores high VRAM when GPU is active', () => {
  // High VRAM but GPU is actually working (not stuck)
  const gpu = { utilization: 75, memoryUsedMb: 28000, memoryTotalMb: 31160 };
  const state = { vramStuckSince: null };
  const result = assessVramStuck(gpu, state, 1000);
  assert.equal(result.vramStuck, false, 'must not trigger during active generation');
  assert.equal(result.vramStuckSince, null, 'must not start the timer during active work');
});

test('threshold constants are sane for Arc B70 (31.16 GB budget)', () => {
  // 2 GB free threshold — leaves headroom but catches the 30 GB wedge
  assert.equal(VRAM_STUCK_THRESHOLD_MB, 2048);
  // 45s sustain — long enough to ride clock frequency dips, short enough
  // to not block the user for minutes after a DEVICE_LOST
  assert.equal(VRAM_STUCK_SUSTAINED_MS, 45 * 1000);
});
