'use strict';

/**
 * Regression tests: H3 LoRA loading path device-agnosticism.
 *
 * Root cause analysis (validated by source tracing, not assumption):
 *   - H3 user LoRAs load via LoraLoaderModelOnly (ComfyUI stock). Internally
 *     this calls comfy.lora.load_lora() → LoRAAdapter.load() which loads
 *     weights as CPU tensors and uses cast_to_device() at compute time.
 *     No .cuda() calls anywhere in this path.
 *
 *   - H3 Turbo LoRA loads via MiniMaxH3TurboLoRA (custom node). Its
 *     _FrugalLoRA.bypass_forward uses F.linear(...) which works on any
 *     device. No .cuda() calls in apply_lora, _apply_bypass_lora, or
 *     _inject_adaln_egrid.
 *
 *   - The crash previously attributed to H3 LoRA loading was actually in
 *     the text encoder (sd1_clip.py NoneType weight) — fixed by the
 *     comfyui-h3-arc-patch. LoRA loading itself has no device-specific code.
 *
 * These tests assert that the graph-building code never emits CUDA-only
 * node types for H3 LoRA application, and that the LoRA node type selection
 * is correct for both user and Turbo LoRAs.
 *
 * Reference: docs/comfyui-h3-arc-patch.md
 */
const test = require('node:test');
const assert = require('node:assert/strict');

const baseSettings = {
  h3Unet: 'minimax_h3_fl2va_pruned_int8_convrot.safetensors',
  h3RefUnet: 'minimax_h3_ref2va_pruned_int8_convrot.safetensors',
  h3Clip: 'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors',
  h3VideoVae: 'minimax_h3_video_vae_fp16.safetensors',
  h3AudioVae: 'minimax_h3_audio_vae_bf16.safetensors',
};

const baseOpts = {
  mode: 'frames',
  prompt: 'a cat',
  negativePrompt: '',
  width: 768,
  height: 1344,
  frames: 121,
  seed: 42,
  steps: 20,
  smooth: 1,
  fps: 24,
  references: { images: [], videos: [], audios: [] },
};

const USER_LORAS = [
  { name: 'spatial_physics.safetensors', strength: 1.0, on: true, managed: false },
  { name: 'wushu_action.safetensors', strength: 0.8, on: true, managed: false },
];

/** Collect all class_type values from a graph */
function collectClassTypes(graph) {
  const types = [];
  for (const key of Object.keys(graph)) {
    if (typeof graph[key] === 'object' && graph[key] !== null && graph[key].class_type) {
      types.push(graph[key].class_type);
    }
  }
  return types;
}

test('H3 graph with user LoRAs uses LoraLoaderModelOnly (device-agnostic)', async () => {
  const { buildMiniMaxH3Graph } = require('../lib/video-workflows');
  const g = await buildMiniMaxH3Graph(
    { ...baseOpts, loras: USER_LORAS },
    { ...baseSettings },
    {},
  );

  const types = collectClassTypes(g);
  // LoraLoaderModelOnly is ComfyUI stock — it calls comfy.lora.load_lora
  // internally, which has no device-specific code paths.
  assert.ok(
    types.includes('LoraLoaderModelOnly'),
    `Expected LoraLoaderModelOnly for user LoRAs on Arc, got: ${JSON.stringify(types)}`
  );

  // The count should match user LORA count (each LoRA gets its own node).
  const loraNodes = types.filter((t) => t === 'LoraLoaderModelOnly');
  assert.equal(
    loraNodes.length, USER_LORAS.length,
    `Expected ${USER_LORAS.length} user LoRA nodes, got ${loraNodes.length}`,
  );
});

test('H3 Turbo graph with standard LoRA uses LoraLoaderModelOnly (lightx8 preset)', async () => {
  const { buildMiniMaxH3Graph } = require('../lib/video-workflows');
  const lightx8Settings = {
    ...baseSettings,
    h3TurboLora: 'minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors',
  };
  const g = await buildMiniMaxH3Graph(
    { ...baseOpts, turbo: true, loras: USER_LORAS },
    lightx8Settings,
    {},
  );

  const types = collectClassTypes(g);
  // lightx8 preset uses standard loader (h3TurboUsesStandardLoader returns true)
  assert.ok(
    types.includes('LoraLoaderModelOnly'),
    'lightx8 Turbo preset must use LoraLoaderModelOnly: ' + JSON.stringify(types),
  );
});

test('H3 Turbo graph with default LoRA uses MiniMaxH3TurboLoRA (no crash on Arc)', async () => {
  const { buildMiniMaxH3Graph } = require('../lib/video-workflows');
  // The default Turbo LoRA uses the custom MiniMaxH3TurboLoRA node.
  // This node's _FrugalLoRA.bypass_forward uses F.linear (device-agnostic).
  // The _inject_adaln_egrid function uses comfy.utils.get_attr (device-agnostic).
  const g = await buildMiniMaxH3Graph(
    { ...baseOpts, turbo: true },
    { ...baseSettings },
    {},
  );

  const types = collectClassTypes(g);
  assert.ok(
    types.includes('MiniMaxH3TurboLoRA'),
    'Default Turbo LoRA must use MiniMaxH3TurboLoRA node: ' + JSON.stringify(types),
  );
});

test('H3 graph never emits CUDA-only node types', async () => {
  const { buildMiniMaxH3Graph } = require('../lib/video-workflows');
  const g = await buildMiniMaxH3Graph(
    { ...baseOpts, turbo: true, loras: USER_LORAS },
    { ...baseSettings },
    {},
  );

  const types = collectClassTypes(g);
  const cudaOnlyNodes = types.filter((t) =>
    /cuda|nvidia|nvrtc|cu12|cu13|tensorrt/i.test(t),
  );
  assert.equal(
    cudaOnlyNodes.length, 0,
    `Graph must not contain CUDA-only node types: ${JSON.stringify(cudaOnlyNodes)}`,
  );
});

test('H3 user LoRAs load with strength_model (not strength) on LoraLoaderModelOnly', async () => {
  const { buildMiniMaxH3Graph } = require('../lib/video-workflows');
  const g = await buildMiniMaxH3Graph(
    { ...baseOpts, loras: USER_LORAS },
    { ...baseSettings },
    {},
  );

  // Find the first user LoRA node.
  const loraKeys = Object.keys(g).filter(
    (k) => typeof g[k] === 'object' && g[k] && g[k].class_type === 'LoraLoaderModelOnly',
  );
  assert.ok(loraKeys.length > 0, 'Must have at least one LoraLoaderModelOnly node');
  const firstLora = g[loraKeys[0]];
  assert.ok(
    'strength_model' in firstLora.inputs,
    'LoraLoaderModelOnly must use strength_model (ComfyUI convention)',
  );
});
