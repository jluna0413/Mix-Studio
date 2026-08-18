'use strict';

const DEFAULT_SEEDVR2_DIT = 'seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors';
const DEFAULT_SEEDVR2_VAE = 'ema_vae_fp16.safetensors';

const OFFLOAD_CONFIG = Object.freeze({
  nvidia: Object.freeze({
    device: 'cuda:0',
    attention: 'sageattn_2',
    unetOffload: 'none',
    textEncoderOffload: 'none',
    vaeOffload: 'none',
    offloadMode: 'disabled',
    maxVramUsage: 0.95,
  }),
  intel: Object.freeze({
    device: 'xpu:0',
    attention: 'sdpa',
    unetOffload: 'model',
    textEncoderOffload: 'model',
    vaeOffload: 'model',
    offloadMode: 'sequential',
    maxVramUsage: 0.90,
  }),
});

function getOffloadConfig(gpuVendor, vramGb = 0, options = {}) {
  const normalizedVendor = String(gpuVendor || '').toLowerCase();
  const vram = Number(vramGb) || 0;
  const h3R2V = options.h3R2V === true;

  if (normalizedVendor === 'intel') {
    // Intel Arc GPUs benefit from sequential CPU offload even with large VRAM.
    // H3 R2V alone loads a ~14GB ref2va model + text encoder + VAE + reference assets,
    // which can exceed 32GB VRAM and cause OOM crashes without offloading.
    // Keep sequential offload enabled for all Intel GPUs regardless of VRAM size.
    return OFFLOAD_CONFIG.intel;
  }
  // NVIDIA: H3 R2V loads 4 models simultaneously (ref2va DiT + qwen3vl 32B CLIP +
  // video VAE + audio VAE). On cards under ~48GB VRAM, force CPU offload on the
  // heavy loaders to avoid OOM during reference preparation.
  if (h3R2V && vram > 0 && vram <= 48) {
    return Object.assign({}, OFFLOAD_CONFIG.nvidia, {
      offloadMode: 'sequential',
      unetOffload: 'model',
      textEncoderOffload: 'model',
      vaeOffload: 'model',
    });
  }
  return OFFLOAD_CONFIG.nvidia;
}

function applyOffloadToGraph(graph, config) {
  if (!config || config.offloadMode === 'disabled') return graph;

  const isSequentialOffload = config.offloadMode === 'sequential';

  for (const [nodeName, node] of Object.entries(graph)) {
    if (!node || typeof node !== 'object') continue;

    const classType = node.class_type;
    const inputs = node.inputs || {};

    if (classType === 'CheckpointLoaderSimple') {
      if (isSequentialOffload) {
        node.class_type = 'LowVRAMCheckpointLoader';
        inputs.offload_device = config.unetOffload === 'model' ? 'cpu' : 'none';
      }
      if (inputs.device === 'default' || inputs.device === undefined) {
        inputs.device = config.device;
      }
    }

    if (classType === 'UNETLoader' || classType === 'UnetLoaderGGUF') {
      if (isSequentialOffload && config.unetOffload === 'model' && !inputs.offload_device) {
        // The heavy DiT (e.g. H3 ref2va ~14GB) must also move to CPU under
        // sequential offload. Without this, the largest model stays pinned
        // in VRAM and H3 R2V reference preparation OOMs on 32GB Intel Arc.
        inputs.offload_device = 'cpu';
      }
      if (inputs.device === 'default' || inputs.device === undefined) {
        inputs.device = config.device;
      }
    }

    if (classType === 'LTXAVTextEncoderLoader' || classType === 'CLIPLoader') {
      // Only set CPU offload if explicitly configured AND we're not in high-VRAM mode
      if (isSequentialOffload && config.textEncoderOffload === 'model' &&
          !(config.device === 'xpu:0' && config.offloadMode === 'disabled')) {
        inputs.offload_device = 'cpu';
      }
    }

    if (classType === 'VAELoader' || classType === 'LTXVAudioVAELoader') {
      if (isSequentialOffload && config.vaeOffload === 'model') {
        inputs.offload_device = 'cpu';
      }
      if (inputs.device === 'default' || inputs.device === undefined) {
        inputs.device = config.device;
      }
    }

    if (classType === 'KSampler' || classType === 'KSamplerAdvanced' || classType === 'SamplerCustom' || classType === 'SamplerCustomAdvanced') {
      if (inputs.device === 'default' || inputs.device === undefined) {
        inputs.device = config.device;
      }
    }

    if (classType === 'ModelSamplingSD3' || classType === 'ModelSamplingFlux' || classType === 'BasicScheduler' || classType === 'ManualSigmas') {
      if (inputs.device === 'default' || inputs.device === undefined) {
        inputs.model = inputs.model;
      }
    }
  }

  return graph;
}

function maybeAddSuperResolution(graph, images, opts, deps) {
  if (!opts.fourK) return images;

  if (deps.addCompatibleVideoUpscaler && typeof deps.addCompatibleVideoUpscaler === 'function') {
    return deps.addCompatibleVideoUpscaler(graph, images, 'vsr');
  }

  // Fallback: simple scale node if no custom builder available
  if (deps.simpleScaleNode && typeof deps.simpleScaleNode === 'function') {
    graph.scale = deps.simpleScaleNode(images, 2);
    return ['scale', 0];
  }

  return images;
}

function seedVR2UpscaleNode(images, scale, config, ditRef, vaeRef, resolution, maxResolution, seed) {
  return {
    class_type: 'SeedVR2VideoUpscaler',
    inputs: {
      image: images,
      dit: ditRef,
      vae: vaeRef,
      seed: seed,
      scale: scale,
      resolution: resolution,
      max_resolution: maxResolution,
      batch_size: 1,
      uniform_batch_size: false,
      color_correction: 'lab',
      temporal_overlap: 0,
      prepend_frames: 0,
      input_noise_scale: 0.06,
      latent_noise_scale: 0,
      offload_device: 'cpu',
      enable_debug: false,
    },
  };
}

module.exports = {
  OFFLOAD_CONFIG,
  getOffloadConfig,
  applyOffloadToGraph,
  maybeAddSuperResolution,
  seedVR2UpscaleNode,
};
