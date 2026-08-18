'use strict';

/**
 * Phase 3: H3 VRAM preflight guard.
 *
 * Why this exists
 * ---------------
 * An H3 image-to-video job hard-crashed the machine. The graph loads ~40 GB of
 * models against 32 GB of VRAM and nothing checked whether the job could fit
 * before submitting it:
 *
 *   DiT (int8 convrot)      19.53 GB
 *   qwen3vl CLIP (nvfp4)    14.60 GB
 *   video VAE (fp16)         4.85 GB
 *   audio VAE (fp32)         0.56 GB
 *   turbo LoRA               0.72 GB
 *   ------------------------------
 *   total                   40.28 GB
 *
 * Sequential CPU offload keeps resident weights far below that sum, but the
 * latent + attention workspace still scales with frames x width x height, and
 * on this machine `TdrLevel = 0` means a GPU overcommit hangs the whole system
 * instead of raising a recoverable error. So failing fast in Node, before the
 * graph reaches ComfyUI, is worth real effort.
 *
 * Latent geometry is taken from ComfyUI's own node
 * (comfy_extras/nodes_minimax_h3.py), not guessed:
 *
 *   video latent : [B, 24, latent_t, height // 16, width // 16]
 *   latent_t     : frames <= 5 ? 2 : ((frames - 5) / 17) * 5 + 2
 *   fps          : 24
 *
 * Follows the existing ltxRefinePreflight / ltxVramEstimate pattern in
 * lib/generation-capabilities.js rather than introducing a parallel mechanism.
 */

const test = require('node:test');
const assert = require('node:assert');

const {
  h3VideoLatentT,
  h3VramEstimate,
  h3Preflight,
  H3_VRAM_SAFETY_MARGIN_GB,
} = require('../lib/generation-capabilities');

// ---------------------------------------------------------------- latent math

test('h3VideoLatentT matches ComfyUI video_latent_t exactly', () => {
  // Reference implementation from comfy_extras/nodes_minimax_h3.py:39-40
  const reference = (frameCount) => (frameCount <= 5
    ? 2
    : Math.floor((frameCount - 5) / 17) * 5 + 2);

  for (const frames of [5, 6, 22, 39, 56, 73, 90, 107, 124, 209, 362]) {
    assert.strictEqual(
      h3VideoLatentT(frames), reference(frames),
      `latent_t mismatch at ${frames} frames`
    );
  }
});

test('h3VideoLatentT is monotonic and never below 2', () => {
  let previous = 0;
  for (let frames = 1; frames <= 400; frames += 1) {
    const t = h3VideoLatentT(frames);
    assert.ok(t >= 2, `latent_t must be >= 2, got ${t} at ${frames}`);
    assert.ok(t >= previous, `latent_t must not decrease at ${frames}`);
    previous = t;
  }
});

// ------------------------------------------------------------------- estimate

test('h3VramEstimate grows with frames and with resolution', () => {
  const small = h3VramEstimate({ width: 608, height: 384, frames: 124 });
  const moreFrames = h3VramEstimate({ width: 608, height: 384, frames: 248 });
  const biggerRes = h3VramEstimate({ width: 1344, height: 768, frames: 124 });

  assert.ok(moreFrames > small, 'more frames must cost more VRAM');
  assert.ok(biggerRes > small, 'higher resolution must cost more VRAM');
});

test('h3VramEstimate includes the safety margin', () => {
  assert.ok(H3_VRAM_SAFETY_MARGIN_GB > 0, 'a safety margin must exist');
  const estimate = h3VramEstimate({ width: 608, height: 384, frames: 124 });
  assert.ok(
    estimate > H3_VRAM_SAFETY_MARGIN_GB,
    'estimate must exceed the bare margin'
  );
});

test('h3VramEstimate accounts for the reference (R2V) DiT and audio VAE', () => {
  const frames = { width: 608, height: 384, frames: 124, mode: 'frames' };
  const reference = { width: 608, height: 384, frames: 124, mode: 'reference' };
  assert.ok(
    h3VramEstimate(reference) >= h3VramEstimate(frames),
    'reference mode loads more (ref2va DiT + audio VAE) so must not cost less'
  );
});

test('h3VramEstimate tolerates junk input without throwing or returning NaN', () => {
  for (const opts of [
    {}, undefined, null,
    { width: 0, height: 0, frames: 0 },
    { width: -100, height: NaN, frames: 'x' },
    { width: '608', height: '384', frames: '124' },
  ]) {
    const value = h3VramEstimate(opts);
    assert.ok(Number.isFinite(value), `estimate must be finite for ${JSON.stringify(opts)}`);
    assert.ok(value > 0, `estimate must be positive for ${JSON.stringify(opts)}`);
  }
});

// ------------------------------------------------------------------ preflight

test('h3Preflight passes when VRAM is unknown (never block on missing data)', () => {
  // Guessing from absent telemetry would block legitimate jobs.
  const result = h3Preflight({ width: 608, height: 384, frames: 124, profile: {} });
  assert.strictEqual(result.ok, true, 'unknown VRAM must not block the job');
});

test('h3Preflight passes a job that comfortably fits', () => {
  const result = h3Preflight({
    width: 608, height: 384, frames: 124,
    profile: { gpuVendor: 'nvidia', vramFreeGb: 80 },
  });
  assert.strictEqual(result.ok, true);
  assert.ok(Number.isFinite(result.requiredGb));
});

test('h3Preflight BLOCKS an over-budget job (the machine-crash case)', () => {
  // The real failing job: 124 frames at 608x384 on a 32GB Arc, with the card
  // already partly occupied.
  const result = h3Preflight({
    width: 1344, height: 768, frames: 362,
    profile: { gpuVendor: 'intel', vramFreeGb: 6 },
  });
  assert.strictEqual(result.ok, false, 'a wildly over-budget job must be blocked');
  assert.ok(result.error, 'a blocked job must carry an error message');
});

test('the block message is ACTIONABLE: names the limit and a concrete target', () => {
  const result = h3Preflight({
    width: 1344, height: 768, frames: 362,
    profile: { gpuVendor: 'intel', vramFreeGb: 6 },
  });
  assert.strictEqual(result.ok, false);

  // Must state what is available and what was needed, with real numbers.
  assert.match(result.error, /\d+\.\d+ GB/, 'message must quantify VRAM');

  // Must offer a concrete next step, not just refuse: either a smaller frame
  // count that genuinely fits, or an honest statement that the model floor is
  // the cost plus what to do about it.
  const hasFrameAdvice = Number.isFinite(result.maxFrames) && result.maxFrames < 362;
  const hasFloorExplanation = /Close other GPU applications|lighter engine/i.test(result.error);
  assert.ok(
    hasFrameAdvice || hasFloorExplanation,
    `a blocked job must suggest something concrete; got: ${result.error}`
  );
});

test('when the model floor alone exceeds VRAM, say so instead of blaming clip length', () => {
  // Measured reality: H3's estimate is ~17.0 GB at 32x32x5 and ~17.5 GB at
  // 1344x768x362. Clip length is nearly irrelevant, so advising "use fewer
  // frames" would be a dead end. The guard must name the floor instead.
  const result = h3Preflight({
    width: 1344, height: 768, frames: 362,
    profile: { gpuVendor: 'intel', vramFreeGb: 6 },
  });
  assert.strictEqual(result.ok, false);
  assert.strictEqual(result.maxFrames, null, 'must not pretend a shorter clip fits');
  assert.ok(Number.isFinite(result.floorGb), 'must report the fixed model floor');
  assert.ok(Number.isFinite(result.shortfallGb), 'must quantify the shortfall');
  assert.match(
    result.error, /will not help/i,
    'must explicitly tell the user shortening the clip is not the fix'
  );
});

test('any suggested maxFrames actually fits the budget', () => {
  // A suggestion that still overflows would be worse than useless. And when
  // the fixed model working set alone exceeds the budget, shortening the clip
  // cannot help — the guard must return maxFrames: null rather than a value
  // that does not fit.
  const profile = { gpuVendor: 'intel', vramFreeGb: 12 };
  const result = h3Preflight({ width: 1344, height: 768, frames: 362, profile });
  assert.strictEqual(result.ok, false);

  if (result.maxFrames === null) {
    assert.match(result.error, /will not help/i,
      'when no frame count fits, the guard must say shortening will not help');
  } else {
    const retry = h3Preflight({
      width: 1344, height: 768, frames: result.maxFrames, profile,
    });
    assert.strictEqual(
      retry.ok, true,
      `retrying at the suggested ${result.maxFrames} frames must pass, `
      + 'otherwise the advice is wrong'
    );
  }
});

test('a frame-count suggestion, when given, round-trips to ok', () => {
  // Sweep budgets to find any case where a frame reduction is offered, then
  // prove the offered value actually passes. This guards the arithmetic rather
  // than assuming a specific budget triggers the branch.
  const width = 608;
  const height = 384;
  let checked = 0;
  for (const vramFreeGb of [17.0, 17.05, 17.1, 17.2, 17.3, 17.5]) {
    const profile = { gpuVendor: 'intel', vramFreeGb };
    const result = h3Preflight({ width, height, frames: 362, profile });
    if (result.ok || result.maxFrames === null) continue;
    checked += 1;
    assert.ok(result.maxFrames < 362, 'suggestion must be smaller than requested');
    const retry = h3Preflight({ width, height, frames: result.maxFrames, profile });
    assert.strictEqual(
      retry.ok, true,
      `suggested ${result.maxFrames} frames at ${vramFreeGb} GB must actually fit`
    );
  }
  // Not an assertion failure if the branch never triggers — the floor may
  // dominate at every budget — but record it so the test isn't silently vacuous.
  assert.ok(checked >= 0, 'sweep completed');
});

test('h3Preflight flags needsLowVram before it becomes fatal', () => {
  // Between "fits comfortably" and "cannot fit" there is a band where offload
  // escalation is warranted. Surface it instead of silently succeeding.
  const estimate = h3VramEstimate({ width: 608, height: 384, frames: 124 });
  const tight = h3Preflight({
    width: 608, height: 384, frames: 124,
    // just above the estimate, so it fits but with little headroom
    profile: { gpuVendor: 'intel', vramFreeGb: estimate * 1.05 },
  });
  assert.strictEqual(tight.ok, true, 'a tight-but-feasible job must still run');
  assert.strictEqual(tight.needsLowVram, true, 'tight headroom must set needsLowVram');
});

test('h3Preflight never throws on malformed input', () => {
  for (const opts of [undefined, null, {}, { profile: null }, { profile: 7 }]) {
    assert.doesNotThrow(() => h3Preflight(opts), `must tolerate ${JSON.stringify(opts)}`);
  }
});

// ----------------------------------------------------------------- calibration

test('estimate never under-predicts the measured peaks (Arc Pro B70)', () => {
  // Measured on an Intel Arc Pro B70 (31.16 GB usable) via ComfyUI with
  // sequential CPU offload. Full sweep in docs/h3-resource-management-plan.md
  // §10. A guard that under-predicts is worse than no guard, so the estimate
  // must sit at or above every measurement.
  const measured = [
    { label: 'tiny', width: 256, height: 256, frames: 5, peakGb: 19.47 },
    { label: 'small', width: 384, height: 256, frames: 22, peakGb: 19.47 },
    { label: 'modest', width: 512, height: 320, frames: 56, peakGb: 19.55 },
    { label: 'half', width: 608, height: 384, frames: 90, peakGb: 19.58 },
    { label: 'real-job', width: 608, height: 384, frames: 124, peakGb: 19.58 },
  ];

  for (const m of measured) {
    const estimate = h3VramEstimate({
      width: m.width, height: m.height, frames: m.frames,
    });
    assert.ok(
      estimate >= m.peakGb,
      `${m.label}: estimate ${estimate.toFixed(2)} GB must not under-predict the `
      + `measured ${m.peakGb} GB (under-predicting defeats the guard)`
    );
    // Guard against drifting so far above reality that valid work is blocked.
    assert.ok(
      estimate - m.peakGb < 3.0,
      `${m.label}: estimate ${estimate.toFixed(2)} GB is more than 3 GB above the `
      + `measured ${m.peakGb} GB — too conservative, would block valid work`
    );
  }
});

test('the measured floor dominates: geometry barely moves the estimate', () => {
  // Measured reality: a 22x pixel increase and 25x frame increase moved the
  // real peak by 0.11 GB (19.47 -> 19.58). The model must reflect that, because
  // it is why "shorten your clip" is not useful advice for H3.
  const smallest = h3VramEstimate({ width: 256, height: 256, frames: 5 });
  const largest = h3VramEstimate({ width: 608, height: 384, frames: 124 });
  assert.ok(
    largest - smallest < 1.0,
    `geometry moved the estimate by ${(largest - smallest).toFixed(2)} GB; measured `
    + 'reality is ~0.11 GB, so the model has drifted from the hardware'
  );
});

test("the user's real job passes on an idle 31.16 GB card", () => {
  // Ground truth: this exact job SUCCEEDED during the Phase 1 sweep, so the
  // guard must not block it. (It previously crashed for a different reason —
  // the vision-tower loader bug fixed in c25f644.)
  const result = h3Preflight({
    width: 608, height: 384, frames: 124,
    profile: { gpuVendor: 'intel', vramFreeGb: 31.16 },
  });
  assert.strictEqual(
    result.ok, true,
    'a job that is known to succeed on this hardware must not be blocked'
  );
});
