'use strict';

const APPLE_UNSUPPORTED_ENGINES = Object.freeze({
  ltx25: 'LTX 2.5 currently uses the official NVIDIA INT8 ConvRot transformer and text encoder, which Apple Metal cannot execute.',
  h3: 'MiniMax H3 currently uses NVIDIA NVFP4/AWQ and INT8 ConvRot model files that Apple Metal cannot execute.',
  eros: '10Eros currently requires FP8 checkpoint and text-encoder operations that Apple Metal cannot execute.',
  wan: 'Wan 2.2 currently uses curated FP8 model and text-encoder files that Apple Metal cannot execute.',
  'wan-animate2': 'Wan Animate 2 currently uses the official INT8 ConvRot diffusion model and FP8 text encoder, which Apple Metal cannot execute.',
  scail: 'SCAIL 2 currently uses a curated FP8 Wan model that Apple Metal cannot execute.',
});

const APPLE_UNSUPPORTED_COMPONENTS = Object.freeze({
  ltx25: APPLE_UNSUPPORTED_ENGINES.ltx25,
  h3: APPLE_UNSUPPORTED_ENGINES.h3,
  h3r2v: APPLE_UNSUPPORTED_ENGINES.h3,
  h3turbo: APPLE_UNSUPPORTED_ENGINES.h3,
  eros: APPLE_UNSUPPORTED_ENGINES.eros,
  wan: APPLE_UNSUPPORTED_ENGINES.wan,
  wananimate2: APPLE_UNSUPPORTED_ENGINES['wan-animate2'],
  scail: APPLE_UNSUPPORTED_ENGINES.scail,
  scailinfinity: APPLE_UNSUPPORTED_ENGINES.scail,
});

const AMD_UNSUPPORTED_ENGINES = Object.freeze({
  ltx25: 'LTX 2.5 currently uses the official NVIDIA INT8 ConvRot transformer and text encoder, which AMD ROCm cannot execute.',
  h3: 'MiniMax H3 currently uses NVIDIA NVFP4/AWQ and INT8 ConvRot model files that AMD ROCm cannot execute.',
  'wan-animate2': 'Wan Animate 2 currently uses the official NVIDIA INT8 ConvRot diffusion model, which AMD ROCm cannot execute.',
});

const AMD_UNSUPPORTED_COMPONENTS = Object.freeze({
  ltx25: AMD_UNSUPPORTED_ENGINES.ltx25,
  h3: AMD_UNSUPPORTED_ENGINES.h3,
  h3r2v: AMD_UNSUPPORTED_ENGINES.h3,
  h3turbo: AMD_UNSUPPORTED_ENGINES.h3,
  wananimate2: AMD_UNSUPPORTED_ENGINES['wan-animate2'],
});

const APPLE_LTX_REFINE_TOKEN_LIMIT = 45_000;
const GENERAL_LTX_REFINE_TOKEN_LIMIT = 200_000;

const LTX_VRAM_SAFETY_MARGIN_GB = 3.0;

function ltxVramEstimate(width, height, frames, stage) {
  const w = Math.max(1, Number(width) || 1);
  const h = Math.max(1, Number(height) || 1);
  const f = Math.max(1, Number(frames) || 1);
  const latentBytes = stage === 1
    ? ((w / 2) * (h / 2) * f * 4)
    : (w * h * f * 4);
  const latentGb = latentBytes / 1e9;
  const components = stage === 1
    ? { textEncoder: 5.0, unetLoras: 14.0, latent: latentGb, attentionTemp: 2.0, overhead: 2.0 }
    : { unetLoras: 14.0, upsampler: 2.0, latent: latentGb, attentionTemp: 4.0, vaeDecode: 0.5, overhead: 2.0 };
  return Object.values(components).reduce((sum, v) => sum + v, 0) + LTX_VRAM_SAFETY_MARGIN_GB;
}

function appleGenerationProfile(profile = {}) {
  const vendor = String(profile.gpuVendor || profile.vendor || '').toLowerCase();
  const backend = String(profile.gpuBackend || profile.backend || '').toLowerCase();
  return vendor === 'apple' || backend === 'mps';
}

function amdGenerationProfile(profile = {}) {
  const vendor = String(profile.gpuVendor || profile.vendor || '').toLowerCase();
  const backend = String(profile.gpuBackend || profile.backend || '').toLowerCase();
  return vendor === 'amd' || backend === 'rocm';
}

function intelGenerationProfile(profile = {}) {
  const vendor = String(profile.gpuVendor || profile.vendor || '').toLowerCase();
  const backend = String(profile.gpuBackend || profile.backend || '').toLowerCase();
  return vendor === 'intel' || backend === 'xpu';
}

function fp8Model(value) {
  return /(?:^|[_.-])fp8(?:[_.-]|$)|float8|e4m3/i.test(String(value || ''));
}

function videoEngineCapabilities(profile = {}) {
  const apple = appleGenerationProfile(profile);
  const amd = amdGenerationProfile(profile);
  const result = {};
  for (const engine of ['ltx', 'ltx25', 'h3', 'ltx-edit', 'eros', 'wan', 'wan-animate2', 'scail']) {
    const reason = apple
      ? APPLE_UNSUPPORTED_ENGINES[engine]
      : (amd ? AMD_UNSUPPORTED_ENGINES[engine] : '');
    result[engine] = {
      supported: !reason,
      reason,
      requiresBf16: apple && (engine === 'ltx' || engine === 'ltx-edit'),
      nvidiaOnly: !!AMD_UNSUPPORTED_ENGINES[engine] && !APPLE_UNSUPPORTED_ENGINES[engine],
    };
  }
  return result;
}

function configuredVideoEngineCapability(engine, profile = {}, settings = {}) {
  const base = videoEngineCapabilities(profile)[engine] || { supported: true, reason: '', requiresBf16: false };
  if (!base.supported) return base;
  if (base.requiresBf16 && fp8Model(settings.ltxCkpt)) {
    return {
      supported: false,
      requiresBf16: true,
      reason: 'LTX 2.3 on Apple Silicon needs the official BF16 checkpoint. Open Generation Setup and install the LTX workflow to select it.',
    };
  }
  return base;
}

function configuredVideoEngineCapabilities(profile = {}, settings = {}) {
  return Object.fromEntries(['ltx', 'ltx25', 'h3', 'ltx-edit', 'eros', 'wan', 'wan-animate2', 'scail']
    .map((engine) => [engine, configuredVideoEngineCapability(engine, profile, settings)]));
}

function dependencyComponentBlock(componentId, profile = {}) {
  if (appleGenerationProfile(profile)) return APPLE_UNSUPPORTED_COMPONENTS[componentId] || '';
  if (amdGenerationProfile(profile)) return AMD_UNSUPPORTED_COMPONENTS[componentId] || '';
  return '';
}

function ltxRefineTokenCount(width, height, frames) {
  const spatial = Math.ceil(Math.max(1, Number(width) || 1) / 32)
    * Math.ceil(Math.max(1, Number(height) || 1) / 32);
  const temporal = Math.ceil(Math.max(1, Number(frames) || 1) / 8);
  return spatial * temporal;
}

function ltxRefinePreflight({ width, height, frames, fps, profile = {} } = {}) {
  const spatial = Math.ceil(Math.max(1, Number(width) || 1) / 32)
    * Math.ceil(Math.max(1, Number(height) || 1) / 32);
  const tokens = ltxRefineTokenCount(width, height, frames);
  const limit = appleGenerationProfile(profile)
    ? APPLE_LTX_REFINE_TOKEN_LIMIT
    : GENERAL_LTX_REFINE_TOKEN_LIMIT;
  if (tokens > limit) {
    const temporalGroups = Math.max(1, Math.floor(limit / spatial));
    const maxFrames = Math.max(1, (temporalGroups - 1) * 8 + 1);
    const rate = Math.max(1, Number(fps) || 25);
    const maxSeconds = Math.max(1, Math.floor(((maxFrames - 1) / rate) * 10) / 10);
    return {
      ok: false,
      tokens,
      limit,
      maxFrames,
      maxSeconds,
      error: `This refined LTX request is too large for the connected generation device. Use ${maxSeconds} seconds or less at ${width} × ${height}, or lower the output size.`,
    };
  }

  const vramFreeGb = Number(profile?.vramFreeGb);
  if (!Number.isFinite(vramFreeGb) || vramFreeGb <= 0) {
    return { ok: true, tokens, limit };
  }

  const requiredStage1 = ltxVramEstimate(width, height, frames, 1);
  const requiredStage2 = ltxVramEstimate(width, height, frames, 2);
  const requiredGb = Math.max(requiredStage1, requiredStage2);
  const lowVramThreshold = vramFreeGb * 0.85;

  if (requiredGb > vramFreeGb) {
    const rate = Math.max(1, Number(fps) || 25);
    const stage2BaseGb = 14.0 + 2.0 + 4.0 + 0.5 + 2.0 + LTX_VRAM_SAFETY_MARGIN_GB;
    const availableForLatent = Math.max(0, vramFreeGb - stage2BaseGb);
    const maxFramesForRes = availableForLatent > 0 ? Math.max(1, Math.floor(availableForLatent / ((width * height * 4) / 1e9))) : 1;
    const maxFrames = Math.max(1, Math.min(maxFramesForRes, 201));
    const maxSeconds = Math.max(1, Math.floor(((maxFrames - 1) / rate) * 10) / 10);
    return {
      ok: false,
      tokens,
      limit,
      maxFrames,
      maxSeconds,
      requiredGb,
      vramFreeGb,
      error: `This refined LTX request exceeds available VRAM (${vramFreeGb.toFixed(1)} GB free, ~${requiredGb.toFixed(1)} GB needed). Use ${maxSeconds} seconds or less at ${width} × ${height}, or lower the output size.`,
    };
  }

  const needsLowVram = requiredGb > lowVramThreshold;
  return {
    ok: true,
    tokens,
    limit,
    requiredGb,
    vramFreeGb,
    needsLowVram,
  };
}

module.exports = {
  AMD_UNSUPPORTED_COMPONENTS,
  AMD_UNSUPPORTED_ENGINES,
  APPLE_LTX_REFINE_TOKEN_LIMIT,
  APPLE_UNSUPPORTED_COMPONENTS,
  APPLE_UNSUPPORTED_ENGINES,
  GENERAL_LTX_REFINE_TOKEN_LIMIT,
  appleGenerationProfile,
  amdGenerationProfile,
  configuredVideoEngineCapability,
  configuredVideoEngineCapabilities,
  dependencyComponentBlock,
  fp8Model,
  intelGenerationProfile,
  ltxRefinePreflight,
  ltxRefineTokenCount,
  ltxVramEstimate,
  videoEngineCapabilities,
};
