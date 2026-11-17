'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const {
  DEFAULT_SEEDVR2_DIT,
  SHARP_SEEDVR2_DIT,
  LEGACY_KREA_SEEDVR2_DIT,
  normalizeSeedVr2Defaults,
  installedSeedVr2Models,
  seedVr2Profile,
  seedVr2DitInputs,
  seedVr2AttentionForVendor,
  seedVr2DeviceForVendor,
  targetResolutionForUpscale,
  addCompatibleVideoUpscaler,
  seedVr2UpscaleNodes,
  buildUltimateSdUpscaleGraph,
} = require('../lib/upscale-workflows');

test('normalizes legacy KreaStudio SeedVR2 default to the Pixel Relay 7B model', () => {
  const settings = normalizeSeedVr2Defaults({
    seedvr2Dit: LEGACY_KREA_SEEDVR2_DIT,
    seedvr2Vae: 'ema_vae_fp16.safetensors',
    seedvr2Attention: 'sageattn_2',
  });

  assert.equal(settings.seedvr2Dit, DEFAULT_SEEDVR2_DIT);
});

test('uses 7B-friendly SeedVR2 loader settings', () => {
  const inputs = seedVr2DitInputs({
    seedvr2Dit: DEFAULT_SEEDVR2_DIT,
    seedvr2Attention: 'sageattn_2',
  });

  assert.equal(inputs.model, DEFAULT_SEEDVR2_DIT);
  assert.equal(inputs.blocks_to_swap, 32);
  assert.equal(inputs.swap_io_components, true);
  assert.equal(inputs.offload_device, 'cpu');
  assert.equal(inputs.attention_mode, 'sageattn_2');
  // Default vendor targets the host GPU device.
  assert.equal(inputs.device, 'cpu');
});

test('uses 7B-friendly SeedVR2 loader settings with Intel offload', () => {
  const inputs = seedVr2DitInputs({
    seedvr2Dit: DEFAULT_SEEDVR2_DIT,
    seedvr2Attention: 'sageattn_2',
    gpuVendor: 'intel',
  });

  assert.equal(inputs.model, DEFAULT_SEEDVR2_DIT);
  assert.equal(inputs.blocks_to_swap, 32);
  assert.equal(inputs.swap_io_components, true);
  assert.equal(inputs.offload_device, 'cpu');
  assert.equal(inputs.attention_mode, 'sdpa');
  // Intel XPU hosts SeedVR2 on xpu:0 (the node advertises this device).
  assert.equal(inputs.device, 'xpu:0');
});

test('SeedVR2 loader targets the host GPU device per vendor', () => {
  const nvidiaInputs = seedVr2DitInputs({
    seedvr2Dit: DEFAULT_SEEDVR2_DIT,
    seedvr2Attention: 'sdpa',
    gpuVendor: 'nvidia',
  });
  assert.equal(nvidiaInputs.device, 'cuda:0');

  const intelInputs = seedVr2DitInputs({
    seedvr2Dit: DEFAULT_SEEDVR2_DIT,
    seedvr2Attention: 'sdpa',
    gpuVendor: 'intel',
  });
  assert.equal(intelInputs.attention_mode, 'sdpa');
  assert.equal(intelInputs.device, 'xpu:0');

  const noVendorInputs = seedVr2DitInputs({
    seedvr2Dit: DEFAULT_SEEDVR2_DIT,
    seedvr2Attention: 'sdpa',
  });
  assert.equal(noVendorInputs.device, 'cpu');
});

test('seedVr2DeviceForVendor maps the host GPU', () => {
  assert.equal(seedVr2DeviceForVendor('nvidia'), 'cuda:0');
  assert.equal(seedVr2DeviceForVendor('Intel'), 'xpu:0');
  assert.equal(seedVr2DeviceForVendor('intel'), 'xpu:0');
  assert.equal(seedVr2DeviceForVendor('amd'), 'cpu');
  assert.equal(seedVr2DeviceForVendor(''), 'cpu');
  assert.equal(seedVr2DeviceForVendor(undefined), 'cpu');
});

test('CUDA-only SeedVR2 attention falls back to SDPA on non-NVIDIA backends', () => {
  assert.equal(seedVr2AttentionForVendor('sageattn_2', 'amd'), 'sdpa');
  assert.equal(seedVr2AttentionForVendor('flash_attn_3', 'apple'), 'sdpa');
  assert.equal(seedVr2AttentionForVendor('sageattn_2', 'nvidia'), 'sageattn_2');
  assert.equal(seedVr2AttentionForVendor('sageattn_2', 'intel'), 'sdpa');
  assert.equal(seedVr2DitInputs({
    seedvr2Dit: DEFAULT_SEEDVR2_DIT,
    seedvr2Attention: 'flash_attn_2',
    gpuVendor: 'amd',
  }).attention_mode, 'sdpa');
  
  const intelInputs = seedVr2DitInputs({
    seedvr2Dit: DEFAULT_SEEDVR2_DIT,
    seedvr2Attention: 'sageattn_2',
    gpuVendor: 'intel',
  });
  assert.equal(intelInputs.attention_mode, 'sdpa');
  assert.equal(intelInputs.offload_device, 'cpu');
});

test('sharp SeedVR2 profile prefers the sharp 7B model and low detail noise by default', () => {
  const profile = seedVr2Profile(
    { seedvr2Dit: DEFAULT_SEEDVR2_DIT },
    'sharp',
    [DEFAULT_SEEDVR2_DIT, SHARP_SEEDVR2_DIT]
  );

  assert.equal(profile.key, 'sharp');
  assert.equal(profile.ditModel, SHARP_SEEDVR2_DIT);
  assert.equal(profile.colorCorrection, 'wavelet');
  assert.equal(profile.noise, 'low');
  assert.equal(profile.inputNoiseScale, 0.06);
});

test('SeedVR2 detail noise supports off, low, and medium levels', () => {
  const available = [DEFAULT_SEEDVR2_DIT, SHARP_SEEDVR2_DIT];

  assert.equal(seedVr2Profile({ seedvr2Dit: DEFAULT_SEEDVR2_DIT }, 'sharp', available, 'off').inputNoiseScale, 0);
  assert.equal(seedVr2Profile({ seedvr2Dit: DEFAULT_SEEDVR2_DIT }, 'sharp', available, 'low').inputNoiseScale, 0.06);
  assert.equal(seedVr2Profile({ seedvr2Dit: DEFAULT_SEEDVR2_DIT }, 'sharp', available, 'medium').inputNoiseScale, 0.15);
});

test('sharp SeedVR2 profile falls back when the sharp model is unavailable', () => {
  const profile = seedVr2Profile(
    { seedvr2Dit: DEFAULT_SEEDVR2_DIT },
    'sharp',
    [DEFAULT_SEEDVR2_DIT]
  );

  assert.equal(profile.key, 'balanced');
  assert.equal(profile.ditModel, DEFAULT_SEEDVR2_DIT);
  assert.equal(profile.colorCorrection, 'lab');
});

test('installed SeedVR2 model scan ignores partial downloads', () => {
  const filesByDir = {
    'models/SEEDVR2': [
      DEFAULT_SEEDVR2_DIT,
      `${SHARP_SEEDVR2_DIT}.download`,
      'ema_vae_fp16.safetensors',
    ],
  };
  const models = installedSeedVr2Models(['models/SEEDVR2'], {
    readdirSync(dir) { return filesByDir[dir] || []; },
  });

  assert.deepEqual(models, [DEFAULT_SEEDVR2_DIT, 'ema_vae_fp16.safetensors']);
});

test('builds 4K AI Upscale nodes', () => {
  const graph = {};
  const out = addCompatibleVideoUpscaler(graph, ['decode', 0], 'vsr', 2);

  assert.deepEqual(graph.vsr_model, {
    class_type: 'UpscaleModelLoader',
    inputs: { model_name: 'RealESRGAN_x4plus.pth' },
  });
  assert.deepEqual(graph.vsr_upscale, {
    class_type: 'ImageUpscaleWithModel',
    inputs: { image: ['decode', 0], upscale_model: ['vsr_model', 0] },
  });
  assert.deepEqual(graph.vsr_scale, {
    class_type: 'ImageScaleBy',
    inputs: { image: ['vsr_upscale', 0], upscale_method: 'lanczos', scale_by: 0.5, crop: 'disabled' }
  });
  assert.deepEqual(out, ['vsr_scale', 0]);
});

test('builds a temporally coherent SeedVR2 video post-upscale batch', () => {
  const result = seedVr2UpscaleNodes(['src', 0], {
    settings: {
      seedvr2Dit: DEFAULT_SEEDVR2_DIT,
      seedvr2Vae: 'ema_vae_fp16.safetensors',
      seedvr2Attention: 'sdpa',
    },
    availableModels: [DEFAULT_SEEDVR2_DIT],
    gpuVendor: 'nvidia',
    resolution: 1536,
    batchSize: 5,
    uniformBatchSize: true,
    temporalOverlap: 2,
    seed: 42,
  });

  assert.equal(result.nodes.upscale.class_type, 'SeedVR2VideoUpscaler');
  assert.equal(result.nodes.upscale.inputs.resolution, 1536);
  assert.equal(result.nodes.upscale.inputs.batch_size, 5);
  assert.equal(result.nodes.upscale.inputs.uniform_batch_size, true);
  assert.equal(result.nodes.upscale.inputs.temporal_overlap, 2);
  assert.deepEqual(result.output, ['upscale', 0]);
});

test('targetResolutionForUpscale supports multiplier mode from the original short edge', () => {
  const target = targetResolutionForUpscale({
    mode: 'scale',
    scaleFactor: 2,
    width: 1152,
    height: 1536,
    fallbackResolution: 2160,
  });

  assert.equal(target, 2304);
});

test('builds an Ultimate SD Upscale graph with simple prompt-guided defaults', () => {
  const graph = buildUltimateSdUpscaleGraph({
    imageName: 'source.png',
    prompt: 'a detailed faithful portrait prompt',
    scaleFactor: 2,
    seed: 123,
    settings: {
      unet: 'krea2_turbo_fp8_scaled.safetensors',
      clip: 'Huihui-Qwen3-VL-4B-Instruct-abliterated-fp8_scaled.safetensors',
      clipType: 'krea2',
      vae: 'qwen_image_vae.safetensors',
    },
  });

  assert.equal(graph.load.class_type, 'LoadImage');
  assert.equal(graph.upscale_model.class_type, 'UpscaleModelLoader');
  assert.equal(graph.upscale_model.inputs.model_name, '4x_foolhardy_Remacri.pth');
  assert.equal(graph.ultimate.class_type, 'UltimateSDUpscale');
  assert.equal(graph.ultimate.inputs.upscale_by, 2);
  assert.equal(graph.ultimate.inputs.denoise, 0.22);
  assert.equal(graph.ultimate.inputs.tile_width, 768);
  assert.equal(graph.pos.inputs.text, 'a detailed faithful portrait prompt');
});

test('caps the longest edge at the requested SeedVR2 target', () => {
  const result = seedVr2UpscaleNodes(['load', 0], {
    settings: { seedvr2Dit: DEFAULT_SEEDVR2_DIT, seedvr2Vae: 'ema_vae_fp16.safetensors', seedvr2Attention: 'sdpa' },
    availableModels: [DEFAULT_SEEDVR2_DIT],
    gpuVendor: 'intel',
    vramGb: 32,
    resolution: 3840,
    batchSize: 1,
  });
  assert.equal(result.nodes.upscale.inputs.resolution, 3840);
  assert.equal(result.nodes.upscale.inputs.max_resolution, 3840);
});

test('SeedVR2 keeps half the 7B blocks resident on large-VRAM GPUs only', () => {
  const roomy = seedVr2DitInputs({ seedvr2Dit: DEFAULT_SEEDVR2_DIT, gpuVendor: 'intel', vramGb: 32 });
  assert.equal(roomy.blocks_to_swap, 16);
  const small = seedVr2DitInputs({ seedvr2Dit: DEFAULT_SEEDVR2_DIT, gpuVendor: 'intel', vramGb: 16 });
  assert.equal(small.blocks_to_swap, 32);
  const unknown = seedVr2DitInputs({ seedvr2Dit: DEFAULT_SEEDVR2_DIT, gpuVendor: 'intel' });
  assert.equal(unknown.blocks_to_swap, 32);
});
