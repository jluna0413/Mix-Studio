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

function getOffloadConfig(gpuVendor, vramGb = 0) {
  const normalizedVendor = String(gpuVendor || '').toLowerCase();

  if (normalizedVendor === 'intel') {
    if (Number(vramGb) >= 28) {
      return {
        device: 'xpu:0',
        attention: 'sdpa',
        unetOffload: 'none',
        textEncoderOffload: 'none',
        vaeOffload: 'none',
        offloadMode: 'disabled',
        maxVramUsage: 0.95,
      };
    }
    return OFFLOAD_CONFIG.intel;
  }
  if (normalizedVendor === 'intel') return OFFLOAD_CONFIG.intel;
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
  const config = getOffloadConfig(opts.gpuVendor);
  if (!opts.fourK) return images;

  if (config.device.startsWith('cuda')) {
    graph.vsr = deps.rtxVideoSuperResolutionNode(images);
    return ['vsr', 0];
  }

  // Intel Arc: use Real-ESRGAN frame-by-frame upscaler instead of RTX VSR
  const esrganModel = opts.esrganModel || 'realesrgan-x4';
  const scaleFactor = Number(opts.scale) || 2;
  if (deps.buildRealEsrGANNode && typeof deps.buildRealEsrGANNode === 'function') {
    graph.esrgan_upscale = deps.buildRealEsrGANNode(images, scaleFactor);
    return ['esrgan_upscale', 0];
  }

  // Fallback: simple scale node if no custom builder available
  if (deps.simpleScaleNode && typeof deps.simpleScaleNode === 'function') {
    graph.scale = deps.simpleScaleNode(images, scaleFactor);
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
