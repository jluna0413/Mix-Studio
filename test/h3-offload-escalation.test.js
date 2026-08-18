'use strict';

/**
 * Regression tests for the H3 offload escalation (`options.h3R2V`).
 *
 * The bug these guard against
 * ---------------------------
 * `getOffloadConfig(gpuVendor, vramGb, options)` gates its H3 OOM safety net on
 * `options.h3R2V === true`. Every caller in server.js passed either a NUMBER
 * (`p.vramFreeGb`) or nothing in that third position:
 *
 *   server.js:4547  getOffloadConfig(p.gpuVendor, p.vramGb, p.vramFreeGb)
 *   server.js:4567  getOffloadConfig(p.gpuVendor, p.vramGb, p.vramFreeGb)
 *   server.js:9357  getOffloadConfig(hw.gpuVendor, hw.vramGb)
 *   server.js:9584  getOffloadConfig(vGpuVendor, vGpuVramGb)
 *
 * A number has no `.h3R2V` property, so the flag was never true and the
 * escalation branch never executed. On NVIDIA cards <= 48GB the intended
 * H3 reference-to-video OOM protection silently did nothing.
 *
 * Context: H3 loads ~40GB of models (19.5GB DiT + 14.6GB CLIP + 4.9GB video VAE
 * + 0.6GB audio VAE + 0.7GB turbo LoRA). Without offload escalation that
 * overcommits any card under ~48GB.
 *
 * These tests exercise the REAL exported getOffloadConfig, not a mirror.
 */

const test = require('node:test');
const assert = require('node:assert');

const { getOffloadConfig, OFFLOAD_CONFIG } = require('../lib/offload-config');

test('h3R2V escalates NVIDIA offload on cards at or under 48GB', () => {
  const plain = getOffloadConfig('nvidia', 24);
  assert.strictEqual(plain.offloadMode, 'disabled', 'baseline NVIDIA has offload disabled');

  const escalated = getOffloadConfig('nvidia', 24, { h3R2V: true });
  assert.strictEqual(escalated.offloadMode, 'sequential', 'h3R2V must force sequential offload');
  assert.strictEqual(escalated.unetOffload, 'model');
  assert.strictEqual(escalated.textEncoderOffload, 'model');
  assert.strictEqual(escalated.vaeOffload, 'model');
  assert.strictEqual(escalated.device, 'cuda:0', 'device must stay CUDA');
});

test('h3R2V does not escalate above the 48GB threshold', () => {
  const big = getOffloadConfig('nvidia', 80, { h3R2V: true });
  assert.strictEqual(big.offloadMode, 'disabled', '80GB card needs no offload');
});

test('h3R2V escalation requires a known VRAM size', () => {
  // vram 0 means "unknown" — do not guess an escalation from missing data.
  const unknown = getOffloadConfig('nvidia', 0, { h3R2V: true });
  assert.strictEqual(unknown.offloadMode, 'disabled');
});

test('a NUMBER in the options position must not be read as options (THE BUG)', () => {
  // This is the exact defect: callers passed p.vramFreeGb here. A number has no
  // .h3R2V property, so the escalation silently never fired. The call must be
  // robust to it rather than appearing to work.
  const withNumber = getOffloadConfig('nvidia', 24, 12);
  assert.strictEqual(
    withNumber.offloadMode, 'disabled',
    'a bare number must not accidentally enable escalation'
  );

  // And the correct form must differ — proving the flag is load-bearing.
  const withOptions = getOffloadConfig('nvidia', 24, { h3R2V: true });
  assert.notStrictEqual(
    withNumber.offloadMode, withOptions.offloadMode,
    'passing a number vs {h3R2V:true} must NOT produce the same config; if these '
    + 'match, the escalation is dead code'
  );
});

test('escalation never mutates the frozen shared config objects', () => {
  getOffloadConfig('nvidia', 24, { h3R2V: true });
  assert.strictEqual(OFFLOAD_CONFIG.nvidia.offloadMode, 'disabled',
    'the shared nvidia config must not be mutated by an escalated call');
  assert.strictEqual(OFFLOAD_CONFIG.nvidia.unetOffload, 'none');
});

test('Intel is unaffected by h3R2V (already always sequential)', () => {
  const intel = getOffloadConfig('intel', 32);
  const intelH3 = getOffloadConfig('intel', 32, { h3R2V: true });
  assert.strictEqual(intel.offloadMode, 'sequential');
  assert.strictEqual(intelH3.offloadMode, 'sequential');
  assert.strictEqual(intelH3.device, 'xpu:0');
});

test('options is optional and tolerates junk without throwing', () => {
  for (const junk of [undefined, null, 0, '', 'nonsense', NaN, [], false]) {
    assert.doesNotThrow(
      () => getOffloadConfig('nvidia', 24, junk),
      `getOffloadConfig must tolerate options=${String(junk)}`
    );
    assert.strictEqual(
      getOffloadConfig('nvidia', 24, junk).offloadMode, 'disabled',
      `options=${String(junk)} must not enable escalation`
    );
  }
});
