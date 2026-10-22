'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const {
  classifyLora,
  compatibleCategoriesForContext,
  loraCompatibilityWarning,
  isH3TurboAdapterLora,
  h3TurboStackConflicts,
} = require('../lib/lora-compat');

test('classifies Krea2 LoRAs from metadata and key samples', () => {
  assert.equal(classifyLora({
    name: 'subject_likeness_krea2 3500.safetensors',
    metadata: { ss_base_model_version: 'krea2' },
    keys: ['diffusion_model.blocks.0.attn.wq.lora_A.weight'],
  }), 'krea2');
});

test('classifies Klein 9B LoRAs from metadata and filename hints', () => {
  assert.equal(classifyLora({
    name: 'portrait-klein-9b.safetensors',
    metadata: { ss_base_model_version: 'flux2_klein_9b' },
    keys: ['diffusion_model.double_blocks.0.img_attn.proj.lokr_w1'],
  }), 'klein9');
});

test('classifies Qwen edit LoRAs from key patterns', () => {
  assert.equal(classifyLora({
    name: 'qwen_image_edit_2511_multiple-angles-lora.safetensors',
    metadata: {},
    keys: ['transformer.transformer_blocks.0.attn.add_k_proj.lora_A.weight'],
  }), 'qwen-edit');
});

test('classifies MiniMax H3 LoRAs before overlapping transformer-key heuristics', () => {
  assert.equal(classifyLora({
    name: 'character-style.safetensors',
    metadata: { ss_base_model_version: 'MiniMax-H3' },
    keys: ['transformer.transformer_blocks.0.attn.q_proj.lora_A.weight'],
  }), 'h3');
  assert.equal(classifyLora({
    name: 'MiniMax_H3/film-look.safetensors',
    metadata: {},
    keys: [],
  }), 'h3');
  assert.equal(classifyLora({
    name: 'realism-people.safetensors',
    metadata: {},
    keys: ['diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight'],
  }), 'h3');
});

test('unknown LoRAs are allowed by context filters', () => {
  assert.equal(classifyLora({ name: 'mystery.safetensors', metadata: {}, keys: [] }), 'unknown');
  assert.deepEqual(compatibleCategoriesForContext('edit', 'klein9'), ['klein9', 'unknown']);
  assert.deepEqual(compatibleCategoriesForContext('edit', 'krea2ref'), ['krea2', 'unknown']);
  assert.deepEqual(compatibleCategoriesForContext('edit', 'krea2remix'), ['krea2', 'unknown']);
  assert.deepEqual(compatibleCategoriesForContext('video', null, 'h3'), ['h3', 'unknown']);
  assert.deepEqual(compatibleCategoriesForContext('video', null, 'wan'), ['video', 'unknown']);
});

test('H3 compatibility warning identifies LoRAs made for other video models', () => {
  const warning = loraCompatibilityWarning(
    [{ name: 'wan-style.safetensors', on: true }],
    { 'wan-style.safetensors': { category: 'video' } },
    'video',
    null,
    'h3'
  );
  assert.match(warning, /wan-style/);
});

test('warning lists incompatible selected LoRAs', () => {
  const warning = loraCompatibilityWarning(
    [{ name: 'portrait.safetensors', on: true }],
    { 'portrait.safetensors': { category: 'krea2' } },
    'edit',
    'qwen'
  );
  assert.match(warning, /portrait/);
});

test('warning ignores app-managed workflow LoRAs', () => {
  const warning = loraCompatibilityWarning(
    [
      { name: 'krea2_identity_edit_v1_2.safetensors', on: true, managed: 'edit-workflow-auto' },
      { name: 'portrait.safetensors', on: true },
    ],
    {
      'krea2_identity_edit_v1_2.safetensors': { category: 'klein4' },
      'portrait.safetensors': { category: 'klein4' },
    },
    'edit',
    'krea2ref'
  );
  assert.doesNotMatch(warning, /identity_edit/);
  assert.match(warning, /portrait/);
});

test('classifies Wan21-style names as video, not unknown', () => {
  // Regression: \bwan\b never matched "wan21", so the Wan-on-H3 accident
  // (0/400 key overlap, inert no-op) classified as 'unknown' — which every
  // context filter allows — and slipped through the picker AND the warning.
  assert.equal(classifyLora({
    name: 'Wan21_PusaV1_LoRA_14B_rank512_bf16.safetensors',
    metadata: {},
    keys: ['diffusion_model.blocks.0.cross_attn.q.lora_A.weight'],
  }), 'video');
  assert.equal(classifyLora({ name: 'wan22-style-pack.safetensors', metadata: {}, keys: [] }), 'video');
  assert.equal(classifyLora({ name: 'LTX-Video-film-look.safetensors', metadata: {}, keys: [] }), 'video');
});

test('detects H3 Turbo/distill adapters by managed list and filename pattern', () => {
  assert.equal(isH3TurboAdapterLora('minimax_h3_turbo_v4_step600_ema.safetensors'), true);
  assert.equal(isH3TurboAdapterLora('Wan21\\minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy_resized_avg_rank_21_bf16.safetensors'), true);
  assert.equal(isH3TurboAdapterLora('minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_resized_avg_rank_64_bf16.safetensors'), true);
  assert.equal(isH3TurboAdapterLora('film-look.safetensors'), false);
  assert.equal(isH3TurboAdapterLora('minimax_h3_style_portrait.safetensors'), false);
  assert.equal(isH3TurboAdapterLora('Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors'), false);
});

test('rejects a Turbo adapter stacked on the managed Turbo switch adapter', () => {
  const conflicts = h3TurboStackConflicts(
    [
      { name: 'minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_resized_avg_rank_64_bf16.safetensors', on: true },
      { name: 'film-look.safetensors', on: true },
    ],
    {
      'minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_resized_avg_rank_64_bf16.safetensors': { category: 'h3' },
      'film-look.safetensors': { category: 'h3' },
    },
    {
      turboActive: true,
      managedTurboLoras: ['minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy_resized_avg_rank_21_bf16.safetensors'],
    }
  );
  assert.equal(conflicts.length, 1);
  assert.equal(conflicts[0].reason, 'turbo');
  assert.match(conflicts[0].name, /8step/);
});

test('managed-name duplicates keep the dedicated duplicate path, not the stack guard', () => {
  const conflicts = h3TurboStackConflicts(
    [{ name: 'minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy_resized_avg_rank_21_bf16.safetensors', on: true }],
    {},
    {
      turboActive: true,
      managedTurboLoras: ['minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy_resized_avg_rank_21_bf16.safetensors'],
    }
  );
  assert.deepEqual(conflicts, []);
});

test('rejects wrong-model-family LoRAs on H3 and allows compatible ones', () => {
  const lorasInfo = {
    'Wan21_PusaV1_LoRA_14B_rank512_bf16.safetensors': { category: 'video' },
    'subject.safetensors': { category: 'h3' },
    'mystery.safetensors': { category: 'unknown' },
  };
  const conflicts = h3TurboStackConflicts(
    [
      { name: 'Wan21_PusaV1_LoRA_14B_rank512_bf16.safetensors', on: true },
      { name: 'subject.safetensors', on: true },
      { name: 'mystery.safetensors', on: true },
      { name: 'off-lora.safetensors', on: false },
    ],
    lorasInfo,
    { turboActive: false, managedTurboLoras: [] }
  );
  assert.equal(conflicts.length, 1);
  assert.equal(conflicts[0].reason, 'family');
  assert.equal(conflicts[0].category, 'video');
});
