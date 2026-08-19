'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const {
  OFFLOAD_CONFIG,
  getOffloadConfig,
  applyOffloadToGraph,
  maybeAddSuperResolution,
  seedVR2UpscaleNode,
} = require('../lib/offload-config');

test('getOffloadConfig returns NVIDIA config for nvidia vendor', () => {
  const config = getOffloadConfig('nvidia');
  assert.equal(config.device, 'cuda:0');
  assert.equal(config.attention, 'sageattn_2');
  assert.equal(config.offloadMode, 'disabled');
  assert.equal(config.unetOffload, 'none');
  assert.equal(config.maxVramUsage, 0.95);
});

test('getOffloadConfig returns Intel config for intel vendor', () => {
  const config = getOffloadConfig('intel');
  assert.equal(config.device, 'xpu:0');
  assert.equal(config.attention, 'sdpa');
  assert.equal(config.offloadMode, 'sequential');
  assert.equal(config.unetOffload, 'model');
  assert.equal(config.textEncoderOffload, 'model');
  assert.equal(config.vaeOffload, 'model');
  assert.equal(config.maxVramUsage, 0.90);
});

test('getOffloadConfig defaults to NVIDIA for unknown vendors', () => {
  assert.equal(getOffloadConfig('amd').device, 'cuda:0');
  assert.equal(getOffloadConfig('apple').device, 'cuda:0');
  assert.equal(getOffloadConfig('').device, 'cuda:0');
  assert.equal(getOffloadConfig(undefined).device, 'cuda:0');
  assert.equal(getOffloadConfig(null).device, 'cuda:0');
});

test('getOffloadConfig is case-insensitive for intel', () => {
  assert.equal(getOffloadConfig('Intel').device, 'xpu:0');
  assert.equal(getOffloadConfig('INTEL').device, 'xpu:0');
  assert.equal(getOffloadConfig('InTeL').device, 'xpu:0');
});

test('applyOffloadToGraph returns graph unchanged for NVIDIA (disabled offload)', () => {
  const graph = {
    model: { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: 'test.safetensors' } },
  };
  const config = getOffloadConfig('nvidia');
  const result = applyOffloadToGraph(graph, config);
  assert.equal(result.model.class_type, 'CheckpointLoaderSimple');
  assert.equal(result.model.inputs.ckpt_name, 'test.safetensors');
  assert.equal(result.model.inputs.offload_device, undefined);
});

test('applyOffloadToGraph converts loaders to LowVRAM for Intel', () => {
  const graph = {
    model: { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: 'test.safetensors' } },
    clip: { class_type: 'CLIPLoader', inputs: { clip_name: 'test-clip', type: 'krea2', device: 'default' } },
    vae: { class_type: 'VAELoader', inputs: { vae_name: 'test-vae' } },
  };
  const config = getOffloadConfig('intel');
  const result = applyOffloadToGraph(graph, config);

  assert.equal(result.model.class_type, 'LowVRAMCheckpointLoader');
  assert.equal(result.model.inputs.ckpt_name, 'test.safetensors');
  assert.equal(result.model.inputs.offload_device, 'cpu');

  // CLIPLoader is moved to CPU via `device` (its only placement input — it has
  // no `offload_device`). MEASURED on Arc Pro B70 32GB: with 'default' the
  // ~14.6GB encoder full-loads to VRAM and even 256x256 x5 OOMs at 30.88GB;
  // with 'cpu', 512x320 x56 succeeds at 25.15GB.
  // See test/h3-clip-cpu-placement.test.js.
  assert.equal(result.clip.inputs.device, 'cpu');
  assert.equal(result.clip.inputs.offload_device, undefined);

  assert.equal(result.vae.inputs.offload_device, 'cpu');
});

test('applyOffloadToGraph preserves explicit non-default device', () => {
  const graph = {
    model: { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: 'test.safetensors', device: 'cuda:0' } },
  };
  const config = getOffloadConfig('intel');
  const result = applyOffloadToGraph(graph, config);
  assert.equal(result.model.inputs.device, 'cuda:0');
});

test('applyOffloadToGraph never overwrites LTXAVTextEncoderLoader device with a backend string', () => {
  // LTXAVTextEncoderLoader only accepts 'default' or 'cpu' for its device
  // input. The graph builders set 'default' on CUDA/XPU; the offload pass
  // must not replace it with 'xpu:0' (or 'cuda:0') or ComfyUI rejects the
  // graph with "device: 'xpu:0' not in ['default', 'cpu']".
  const graph = {
    te: { class_type: 'LTXAVTextEncoderLoader', inputs: { text_encoder: 'gemma.safetensors', ckpt_name: 'ltx.safetensors', device: 'default' } },
  };
  const config = getOffloadConfig('intel');
  const result = applyOffloadToGraph(graph, config);
  assert.equal(result.te.inputs.device, 'default');
  assert.equal(result.te.inputs.offload_device, 'cpu');
});

test('applyOffloadToGraph never overwrites CLIPLoader device with a backend string', () => {
  // CLIPLoader only accepts 'default' or 'cpu' for its device input. The
  // offload pass must never replace it with 'xpu:0' or 'cuda:0' or ComfyUI
  // rejects the graph with "device: 'xpu:0' not in ['default', 'cpu']".
  // It IS allowed to select 'cpu' — that is the intended placement, and the
  // only way to keep the encoder off the GPU (CLIPLoader has no
  // offload_device input). See test/h3-clip-cpu-placement.test.js.
  const graph = {
    clip: { class_type: 'CLIPLoader', inputs: { clip_name: 'test-clip', type: 'krea2', device: 'default' } },
  };
  const config = getOffloadConfig('intel');
  const result = applyOffloadToGraph(graph, config);
  assert.ok(['default', 'cpu'].includes(result.clip.inputs.device),
    `CLIPLoader.device must stay in ['default','cpu'], got ${result.clip.inputs.device}`);
  assert.equal(result.clip.inputs.device, 'cpu');
  assert.equal(result.clip.inputs.offload_device, undefined);
});

test('maybeAddSuperResolution uses addCompatibleVideoUpscaler when fourK is true', () => {
  const graph = {};
  const opts = { fourK: true };
  const deps = {
    addCompatibleVideoUpscaler: (g, img, key) => {
      g[`${key}_model`] = { class_type: 'UpscaleModelLoader' };
      return [`${key}_scale`, 0];
    },
  };
  const result = maybeAddSuperResolution(graph, ['decode', 0], opts, deps);
  assert.deepEqual(result, ['vsr_scale', 0]);
  assert.equal(graph.vsr_model.class_type, 'UpscaleModelLoader');
});

test('maybeAddSuperResolution skips 4K upscaling when fourK is false', () => {
  const graph = {};
  const opts = { fourK: false };
  const deps = {
    addCompatibleVideoUpscaler: () => { throw new Error('should not call'); },
  };
  const result = maybeAddSuperResolution(graph, ['decode', 0], opts, deps);
  assert.deepEqual(result, ['decode', 0]);
  assert.equal(graph.vsr, undefined);
});

test('maybeAddSuperResolution returns unchanged images when fourK is false', () => {
  const graph = {};
  const opts = { fourK: false, gpuVendor: 'nvidia' };
  const deps = {
    rtxVideoSuperResolutionNode: () => ({}),
    seedVR2UpscaleNode: () => ({}),
  };
  const result = maybeAddSuperResolution(graph, ['decode', 0], opts, deps);
  assert.deepEqual(result, ['decode', 0]);
  assert.equal(graph.vsr, undefined);
  assert.equal(graph.vsr_fallback, undefined);
});

test('seedVR2UpscaleNode builds SeedVR2VideoUpscaler with correct inputs', () => {
  const config = getOffloadConfig('intel');
  const node = seedVR2UpscaleNode(['decode', 0], 2, config, ['dit', 0], ['vae', 0], 2048, 4096, 42);
  assert.equal(node.class_type, 'SeedVR2VideoUpscaler');
  assert.deepEqual(node.inputs.image, ['decode', 0]);
  assert.deepEqual(node.inputs.dit, ['dit', 0]);
  assert.deepEqual(node.inputs.vae, ['vae', 0]);
  assert.equal(node.inputs.seed, 42);
  assert.equal(node.inputs.scale, 2);
  assert.equal(node.inputs.resolution, 2048);
  assert.equal(node.inputs.max_resolution, 4096);
  assert.equal(node.inputs.batch_size, 1);
  assert.equal(node.inputs.offload_device, 'cpu');
  assert.equal(node.inputs.color_correction, 'lab');
});

test('seedVR2UpscaleNode omits model/device/attention_mode from raw inputs', () => {
  const config = getOffloadConfig('nvidia');
  const node = seedVR2UpscaleNode(['decode', 0], 4, config, ['dit', 0], ['vae', 0], 2048, 4096, 7);
  assert.equal(node.class_type, 'SeedVR2VideoUpscaler');
  assert.equal(node.inputs.model, undefined);
  assert.equal(node.inputs.attention_mode, undefined);
  assert.equal(node.inputs.device, undefined);
  assert.equal(node.inputs.images, undefined);
  assert.equal(node.inputs.scale, 4);
  assert.equal(node.inputs.resolution, 2048);
});

test('applyOffloadToGraph preserves UNETLoader class and sets device for Intel', () => {
  const graph = {
    model: { class_type: 'UNETLoader', inputs: { unet_name: 'test.safetensors' } },
  };
  const config = getOffloadConfig('intel');
  const result = applyOffloadToGraph(graph, config);

  assert.equal(result.model.class_type, 'UNETLoader');
  assert.equal(result.model.inputs.device, 'xpu:0');
});

test('applyOffloadToGraph sets offload_device cpu for UNETLoader on Intel sequential', () => {
  const graph = {
    model: { class_type: 'UNETLoader', inputs: { unet_name: 'test.safetensors' } },
  };
  const config = getOffloadConfig('intel');
  const result = applyOffloadToGraph(graph, config);

  // The heavy DiT (e.g. H3 ref2va ~14GB) must be CPU-offloaded under
  // sequential mode or it stays pinned in VRAM and OOMs on 32GB Intel Arc.
  assert.equal(result.model.inputs.offload_device, 'cpu');
  assert.equal(result.model.inputs.device, 'xpu:0');
});

test('applyOffloadToGraph converts CheckpointLoaderSimple to LowVRAM for Intel', () => {
  const graph = {
    model: { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: 'test.safetensors' } },
  };
  const config = getOffloadConfig('intel');
  const result = applyOffloadToGraph(graph, config);

  assert.equal(result.model.class_type, 'LowVRAMCheckpointLoader');
  assert.equal(result.model.inputs.ckpt_name, 'test.safetensors');
  assert.equal(result.model.inputs.offload_device, 'cpu');
  assert.equal(result.model.inputs.device, 'xpu:0');
});

test('applyOffloadToGraph omits device for ModelSamplingFlux and Flux2Scheduler', () => {
  const graph = {
    sched: { class_type: 'Flux2Scheduler', inputs: { steps: 4, denoise: 1 } },
  };
  const config = getOffloadConfig('intel');
  const result = applyOffloadToGraph(graph, config);

  assert.equal(result.sched.inputs.device, undefined);
});

test('getOffloadConfig returns sequential offload for Intel with high VRAM (32GB)', () => {
  // H3 R2V loads ~14GB ref2va model + text encoder + VAE + reference assets,
  // which exceeds 32GB VRAM. Sequential offload is required even on large GPUs.
  const config = getOffloadConfig('intel', 32);
  assert.equal(config.device, 'xpu:0');
  assert.equal(config.attention, 'sdpa');
  assert.equal(config.offloadMode, 'sequential');
  assert.equal(config.unetOffload, 'model');
  assert.equal(config.textEncoderOffload, 'model');
  assert.equal(config.vaeOffload, 'model');
  assert.equal(config.maxVramUsage, 0.90);
});

test('getOffloadConfig returns sequential offload for Intel with low VRAM (16GB)', () => {
  const config = getOffloadConfig('intel', 16);
  assert.equal(config.device, 'xpu:0');
  assert.equal(config.attention, 'sdpa');
  assert.equal(config.offloadMode, 'sequential');
  assert.equal(config.unetOffload, 'model');
  assert.equal(config.textEncoderOffload, 'model');
  assert.equal(config.vaeOffload, 'model');
  assert.equal(config.maxVramUsage, 0.90);
});

test('getOffloadConfig returns sequential offload for Intel with medium VRAM (28GB)', () => {
  // All Intel GPUs use sequential offload regardless of VRAM to avoid OOM on large models.
  const config = getOffloadConfig('intel', 28);
  assert.equal(config.device, 'xpu:0');
  assert.equal(config.attention, 'sdpa');
  assert.equal(config.offloadMode, 'sequential');
  assert.equal(config.unetOffload, 'model');
  assert.equal(config.textEncoderOffload, 'model');
  assert.equal(config.vaeOffload, 'model');
  assert.equal(config.maxVramUsage, 0.90);
});

test('getOffloadConfig defaults to sequential offload for Intel with no VRAM param', () => {
  const config = getOffloadConfig('intel');
  assert.equal(config.offloadMode, 'sequential');
  assert.equal(config.maxVramUsage, 0.90);
});

test('applyOffloadToGraph converts CheckpointLoaderSimple to LowVRAM for Intel high VRAM (32GB)', () => {
  // Even with 32GB VRAM, Intel uses sequential offload to avoid OOM on large models like H3 R2V.
  const graph = {
    model: { class_type: 'CheckpointLoaderSimple', inputs: { ckpt_name: 'test.safetensors' } },
  };
  const config = getOffloadConfig('intel', 32);
  const result = applyOffloadToGraph(graph, config);

  assert.equal(result.model.class_type, 'LowVRAMCheckpointLoader');
  assert.equal(result.model.inputs.offload_device, 'cpu');
  assert.equal(result.model.inputs.device, 'xpu:0');
});
