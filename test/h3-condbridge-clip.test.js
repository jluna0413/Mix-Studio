'use strict';

/* Verifies the distilled (CondBridge) H3 text-encoder path produces
 * MiniMaxH3AdapterLoader instead of CLIPLoader when
 * h3ClipVariant === 'distilled'.

 * This branch (h3-condbridge-stack) only covers the distilled path.
 * The lightweight (ClipProj) path lives on the h3-lightweight-stack branch.

 * CondBridge: 1.14B adapter bridges Qwen3.5-4B student (2.6 GB) into the
 * H3 DiT's 33B teacher injection space. ~3.8 GB total, vs 14.9 GB for
 * the heavy 32B nvfp4 encoder. The student can be any HF-format model
 * with hidden_size=2560 (Qwen3.5-4B; a fine-tuned variant may serve as
 * a prompt improver). */

const test = require('node:test');
const assert = require('node:assert/strict');
const { buildMiniMaxH3Graph } = require('../lib/video-workflows');

const baseSettings = {
  h3Unet: 'minimax_h3_fl2va_pruned_int8_convrot.safetensors',
  h3Clip: 'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors',
  h3ClipDistilledStudent: 'Qwen3.5-4B',
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

test('distilled variant uses MiniMaxH3AdapterLoader with student + adapter', async () => {
  const g = await buildGraph('distilled');
  assert.equal(g.clip.class_type, 'MiniMaxH3AdapterLoader',
    'the distilled path must swap in MiniMaxH3AdapterLoader');
  assert.equal(g.clip.inputs.adapter, 'adapter_stage2_best.safetensors',
    'must reference the CondBridge adapter file');
  assert.equal(g.clip.inputs.student, 'Qwen3.5-4B',
    'must default to Qwen3.5-4B (no -Instruct/VL suffix)');
});

test('distilled allows a custom student (fine-tuned prompt improver)', async () => {
  const g = await buildGraph('distilled');
  const custom = await buildMiniMaxH3Graph(baseOpts, {
    ...baseSettings,
    h3ClipVariant: 'distilled',
    h3ClipDistilledStudent: 'my-finetuned-qwen3.5-4b',
  }, {});
  assert.equal(custom.clip.inputs.student, 'my-finetuned-qwen3.5-4b',
    'must let the user override the student encoder');
  assert.equal(g.clip.inputs.student, 'Qwen3.5-4B',
    'without override must use the default');
});

test('heavy variants (auto/nvfp4/int8-convrot) keep CLIPLoader', async () => {
  for (const v of ['auto', 'nvfp4-awq', 'int8-convrot']) {
    const g = await buildGraph(v);
    assert.equal(g.clip.class_type, 'CLIPLoader', `variant ${v} must use CLIPLoader`);
  }
});

test('unset variant defaults to auto (CLIPLoader)', async () => {
  const g = await buildGraph(undefined);
  assert.equal(g.clip.class_type, 'CLIPLoader');
});
