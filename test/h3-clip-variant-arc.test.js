'use strict';

/**
 * Regression tests: MiniMax H3 text-encoder (CLIP) selection.
 *
 * The H3 clip variant dropdown offers three options that work on all GPUs:
 *   - nvfp4-awq (32B encoder, 14.6 GB) — universal default
 *   - lightweight (4B encoder + projection, ~2.6 GB)
 *   - distilled (1.14B adapter + Qwen3.5-4B student, ~3.8 GB)
 *
 * NVIDIA-only INT8 ConvRot variants were removed: the ClipProjLoader and
 * CLIPLoader paths both work via backends available on Intel Arc (XPU).
 *
 * Reference: docs/comfyui-h3-arc-patch.md
 */

const test = require('node:test');
const assert = require('node:assert');

const { resolveH3ClipAsset } = require('../lib/dependency-installer');

const NVFP4 = 'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors';
const LIGHTWEIGHT = 'qwen3vl_4b_int4_convrot.safetensors';

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
 * @param {string} h3Variant  requested variant
 * @param {boolean} isIntelArc whether the host GPU is an Intel Arc (XPU)
 */
function selectH3Clip(h3Variant, isIntelArc) {
  const [, , source] = resolveH3ClipAsset(H3_CLIP_ASSET, h3Variant, isIntelArc);
  return String(source).split('/').pop();
}

test('nvfp4-awq request returns the NVFP4 AWQ encoder on all GPUs', () => {
  for (const isArc of [true, false]) {
    assert.strictEqual(
      selectH3Clip('nvfp4-awq', isArc), NVFP4,
      `nvfp4-awq must resolve to the NVFP4 encoder regardless of vendor (arc=${isArc})`
    );
  }
});

test('unrecognized variant falls back to nvfp4-awq (never int8_convrot)', () => {
  for (const junk of ['nonsense', '', null, undefined, 42, {}]) {
    assert.strictEqual(
      selectH3Clip(junk, true), NVFP4,
      `unrecognised value ${JSON.stringify(junk)} must fall back to nvfp4-awq`
    );
    assert.strictEqual(
      selectH3Clip(junk, false), NVFP4,
      `unrecognised value ${JSON.stringify(junk)} must fall back to nvfp4-awq on CUDA`
    );
  }
});

test('Intel Arc never resolves to int8_convrot', () => {
  // All variants must resolve to a GPU-portable encoder on Arc.
  for (const variant of ['nvfp4-awq', 'lightweight', 'distilled', undefined]) {
    const picked = selectH3Clip(variant, true);
    assert.ok(
      !/int8_convrot/i.test(picked),
      `variant=${String(variant)} must not yield int8_convrot on Arc: ${picked}`
    );
  }
});

test('lightweight variant returns the 4B int4 encoder', () => {
  assert.strictEqual(selectH3Clip('lightweight', false), LIGHTWEIGHT);
  assert.strictEqual(selectH3Clip('lightweight', true), LIGHTWEIGHT);
});

test('h3Clip candidate list contains the nvfp4 encoder', () => {
  const { COMPATIBLE_MODEL_NAMES } = require('../lib/model-loader');
  const candidates = COMPATIBLE_MODEL_NAMES.h3Clip;

  assert.ok(Array.isArray(candidates), 'h3Clip candidates must be an array');
  assert.ok(
    candidates.includes(NVFP4),
    `nvfp4 encoder must be a discoverable h3Clip candidate; got ${JSON.stringify(candidates)}`
  );
  assert.ok(
    !candidates.some((c) => /int8_convrot/i.test(c)),
    'h3Clip candidates must not include CUDA-only int8_convrot files'
  );
});

test('h3ClipVariant normalizes known ids and rejects to nvfp4-awq', () => {
  const { normalizeH3ClipVariant } = require('../lib/h3-model-variants');

  for (const id of ['nvfp4-awq', 'lightweight', 'distilled']) {
    assert.strictEqual(normalizeH3ClipVariant(id), id, `${id} must round-trip`);
  }
  // Case and whitespace tolerance.
  assert.strictEqual(normalizeH3ClipVariant('  NVFP4-AWQ  '), 'nvfp4-awq');
  // Anything unrecognised must fall back to nvfp4-awq, never to a broken encoder.
  for (const junk of ['nonsense', '', null, undefined, 42, {}]) {
    assert.strictEqual(
      normalizeH3ClipVariant(junk), 'nvfp4-awq',
      `unrecognised value ${JSON.stringify(junk)} must fall back to nvfp4-awq`
    );
  }
});

test('H3_CLIP_VARIANTS exports only universal variants', () => {
  const { H3_CLIP_VARIANTS } = require('../lib/h3-model-variants');

  const ids = Object.keys(H3_CLIP_VARIANTS);
  assert.ok(ids.includes('nvfp4-awq'), 'must include nvfp4-awq');
  assert.ok(ids.includes('lightweight'), 'must include lightweight');
  assert.ok(ids.includes('distilled'), 'must include distilled');
  assert.ok(!ids.includes('auto'), 'must not include removed auto variant');
  assert.ok(!ids.includes('int8-convrot'), 'must not include CUDA-only int8-convrot');
  assert.ok(!ids.includes('int8-convrot-uncensored'), 'must not include CUDA-only int8-convrot-uncensored');
});

test('default setting is nvfp4-awq', () => {
  const { h3VariantSettingDefaults } = require('../lib/h3-model-variants');
  assert.strictEqual(h3VariantSettingDefaults().h3ClipVariant, 'nvfp4-awq');
});

module.exports = { selectH3Clip };
