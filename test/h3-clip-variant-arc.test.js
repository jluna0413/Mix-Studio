'use strict';

/**
 * Regression tests: MiniMax H3 text-encoder (CLIP) selection on Intel Arc.
 *
 * Background — the bug these lock down:
 * The int8_convrot qwen3vl encoder relies on comfy_kitchen's CUDA-only ConvRot
 * kernels. On Intel Arc (XPU) those kernels do not exist, so the quantized
 * weights never materialize and the first forward pass dies with
 * `AttributeError: 'NoneType' object has no attribute 'device'`.
 *
 * The nvfp4-awq encoder uses the eager backend, which IS available on XPU, and
 * is verified working end-to-end (124 frames generated on an Arc Pro B70).
 *
 * Three defects had Arc pinned to the broken encoder:
 *   1. dependency-installer.js — an explicit `nvfp4-awq` request was ignored on
 *      Arc because of a `&& !isIntelArc` guard.
 *   2. dependency-installer.js — Arc was unconditionally rewritten to
 *      int8_convrot, even under `auto`.
 *   3. model-loader.js — the h3Clip fallback list contained only int8_convrot
 *      filenames, so discovery could never land on nvfp4.
 *
 * Reference: docs/comfyui-h3-arc-patch.md
 */

const test = require('node:test');
const assert = require('node:assert');

const { resolveH3ClipAsset } = require('../lib/dependency-installer');

const NVFP4 = 'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors';
const INT8 = 'qwen3vl_32b_minimax_h3_int8_convrot.safetensors';
const UNCENSORED = 'qwen3vl_32b_minimax_h3_ultra_uncensored_heretic_int8_convrot.safetensors';

// The catalogue entry as MODEL_ASSETS.h3 declares it (nvfp4 is the default).
const H3_CLIP_ASSET = Object.freeze([
  'h3Clip',
  'text_encoders',
  'https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors',
]);

/**
 * Resolve the encoder filename the installer would configure, exercising the
 * real `resolveH3ClipAsset` implementation rather than a local copy.
 *
 * @param {string} h3Variant  requested variant ('auto' when unset)
 * @param {boolean} isIntelArc whether the host GPU is an Intel Arc (XPU)
 */
function selectH3Clip(h3Variant, isIntelArc) {
  const [, , source] = resolveH3ClipAsset(H3_CLIP_ASSET, h3Variant, isIntelArc);
  return String(source).split('/').pop();
}

test('Intel Arc + auto selects the nvfp4 encoder (not CUDA-only int8_convrot)', () => {
  assert.strictEqual(
    selectH3Clip('auto', true), NVFP4,
    'Arc must not be routed to int8_convrot: its ConvRot kernels are CUDA-only'
  );
});

test('explicit nvfp4-awq request is honoured on Intel Arc', () => {
  // Regression: a `&& !isIntelArc` guard used to discard this request on Arc.
  assert.strictEqual(
    selectH3Clip('nvfp4-awq', true), NVFP4,
    'an explicit nvfp4-awq request must not be overridden on Arc'
  );
});

test('explicit nvfp4-awq request is honoured on CUDA', () => {
  assert.strictEqual(selectH3Clip('nvfp4-awq', false), NVFP4);
});

test('explicit int8-convrot request is still honoured (user choice)', () => {
  assert.strictEqual(selectH3Clip('int8-convrot', false), INT8);
  assert.strictEqual(
    selectH3Clip('int8-convrot', true), INT8,
    'naming int8-convrot explicitly must surface the user choice, not silently swap'
  );
});

test('uncensored variant is honoured on both vendors', () => {
  assert.strictEqual(selectH3Clip('int8-convrot-uncensored', false), UNCENSORED);
  assert.strictEqual(selectH3Clip('int8-convrot-uncensored', true), UNCENSORED);
});

test('no code path routes Intel Arc to int8_convrot unless explicitly asked', () => {
  for (const variant of ['auto', 'nvfp4-awq', undefined, '', null]) {
    const picked = selectH3Clip(variant || 'auto', true);
    assert.notStrictEqual(
      picked, INT8,
      `variant=${String(variant)} must not yield int8_convrot on Arc`
    );
  }
});

test('h3Clip compatible names include the nvfp4 encoder', () => {
  // Regression for model-loader.js: the candidate list held only int8_convrot
  // filenames, so model discovery could never resolve to nvfp4.
  const { COMPATIBLE_MODEL_NAMES } = require('../lib/model-loader');
  const candidates = COMPATIBLE_MODEL_NAMES.h3Clip;

  assert.ok(Array.isArray(candidates), 'h3Clip candidates must be an array');
  assert.ok(
    candidates.includes(NVFP4),
    `nvfp4 encoder must be a discoverable h3Clip candidate; got ${JSON.stringify(candidates)}`
  );
});

test('nvfp4 is preferred over int8_convrot in the candidate ordering', () => {
  const { COMPATIBLE_MODEL_NAMES } = require('../lib/model-loader');
  const candidates = COMPATIBLE_MODEL_NAMES.h3Clip;
  const nvfp4At = candidates.indexOf(NVFP4);
  const int8At = candidates.indexOf(INT8);

  assert.ok(nvfp4At !== -1, 'nvfp4 must be present');
  if (int8At !== -1) {
    assert.ok(
      nvfp4At < int8At,
      'nvfp4 must rank before int8_convrot so XPU hosts resolve to a working encoder'
    );
  }
});

test('h3ClipVariant normalizes known ids and rejects junk', () => {
  const { normalizeH3ClipVariant } = require('../lib/h3-model-variants');

  for (const id of ['auto', 'nvfp4-awq', 'int8-convrot', 'int8-convrot-uncensored']) {
    assert.strictEqual(normalizeH3ClipVariant(id), id, `${id} must round-trip`);
  }
  // Case and whitespace tolerance.
  assert.strictEqual(normalizeH3ClipVariant('  NVFP4-AWQ  '), 'nvfp4-awq');
  // Anything unrecognised must fall back to auto, never to a broken encoder.
  for (const junk of ['nonsense', '', null, undefined, 42, {}]) {
    assert.strictEqual(
      normalizeH3ClipVariant(junk), 'auto',
      `unrecognised value ${JSON.stringify(junk)} must fall back to auto`
    );
  }
});

test('h3ClipVariant is a recognised variant id, not dead config', () => {
  // Regression: h3ClipVariant sat in settings.json but had zero readers, so it
  // could silently disagree with h3Clip. It must now normalize and be plumbed
  // through modelVariants.h3 to the installer.
  const { H3_CLIP_VARIANTS, normalizeH3ClipVariant } = require('../lib/h3-model-variants');

  assert.ok(H3_CLIP_VARIANTS, 'H3_CLIP_VARIANTS must be exported');
  assert.ok(
    Object.prototype.hasOwnProperty.call(H3_CLIP_VARIANTS, 'nvfp4-awq'),
    'nvfp4-awq must be a recognised variant'
  );
  // The value persisted in this workspace must survive normalization.
  assert.strictEqual(normalizeH3ClipVariant('nvfp4-awq'), 'nvfp4-awq');
});

test('every recognised clip variant resolves to a real encoder file', () => {
  const known = ['auto', 'nvfp4-awq', 'int8-convrot', 'int8-convrot-uncensored'];
  const valid = new Set([NVFP4, INT8, UNCENSORED]);

  for (const variant of known) {
    for (const isArc of [true, false]) {
      const picked = selectH3Clip(variant, isArc);
      assert.ok(
        valid.has(picked),
        `variant=${variant} arc=${isArc} produced unknown encoder: ${picked}`
      );
    }
  }
});

module.exports = { selectH3Clip };
