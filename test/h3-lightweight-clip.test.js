'use strict';

/* Verifies the lightweight H3 text-encoder path produces ClipProjLoader
 * instead of CLIPLoader when h3ClipVariant === 'lightweight'.

 * This branch (h3-lightweight-stack) only covers the lightweight path.
 * The distilled (CondBridge) path lives on the h3-condbridge-stack branch.

 * MEASURED on Intel Arc Pro B70 32GB:
 *   heavy (32B nvfp4, CLIPLoader):     11,957 MB resident, OOMs at 768x432
 *   lightweight (4B + projection):       2,684 MB + 26 MB adapter, 29.22 GB @768x432

 * The 26MB projection matrix (mmh3-4b-ClipProj-v3.1) absorbs the 4B
 * encoder hidden states into H3's 5376-dim injection space.
 */
const test = require('node:test');
const assert = require('node:assert/strict');
const { buildMiniMaxH3Graph } = require('../lib/video-workflows');

const baseSettings = {
  h3Unet: 'minimax_h3_fl2va_pruned_int8_convrot.safetensors',
  h3Clip: 'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors',
  h3ClipLightweight: 'qwen3vl_4b_int4_convrot.safetensors',
  h3VideoVae: 'minimax_h3_video_vae_fp16.safetensors',
  h3AudioVae: 'minimax_h3_audio_vae_bf16.safetensors',
};

const baseOpts = {
  mode: 'frames', engine: 'h3', width: 672, height: 384,
  seed: 42, steps: 6, cfg: 1.0, turbo: false,
};

function buildGraph(variant) {
  return buildMiniMaxH3Graph(baseOpts, { ...baseSettings, h3ClipVariant: variant }, {});
}

test('lightweight variant uses ClipProjLoader with 4B encoder + projection', async () => {
  const g = await buildGraph('lightweight');
  assert.equal(g.clip.class_type, 'ClipProjLoader',
    'the lightweight path must swap in ClipProjLoader');
  assert.equal(g.clip.inputs.clip_name,
    'qwen3vl_4b_int4_convrot.safetensors',
    'must use the 2.6 GB 4B encoder');
  assert.equal(g.clip.inputs.projection,
    'mmh3-4b-ClipProj-v3.1.safetensors',
    'the 26 MB projection matrix must be wired');
  assert.equal(g.clip.inputs.type, 'auto',
    'type:auto lets ClipProjLoader size-detect xpu/cuda');
});

test('heavy variants (nvfp4-awq) keep CLIPLoader', async () => {
  for (const v of ['nvfp4-awq', undefined]) {
    const g = await buildGraph(v);
    assert.equal(g.clip.class_type, 'CLIPLoader', `variant ${v} must use CLIPLoader`);
  }
});

test('unset variant defaults to nvfp4-awq (CLIPLoader)', async () => {
  const g = await buildGraph(undefined);
  assert.equal(g.clip.class_type, 'CLIPLoader');
});

test('lightweight encoder differs from heavy (4B vs 32B)', async () => {
  const light = await buildGraph('lightweight');
  const heavy = await buildGraph('nvfp4-awq');
  assert.equal(light.clip.inputs.clip_name, 'qwen3vl_4b_int4_convrot.safetensors');
  assert.equal(heavy.clip.inputs.clip_name, 'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors');
});
