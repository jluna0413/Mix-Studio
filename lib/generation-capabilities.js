'use strict';

const APPLE_UNSUPPORTED_ENGINES = Object.freeze({
  ltx25: 'LTX 2.5 currently uses the official NVIDIA INT8 ConvRot transformer and text encoder, which Apple Metal cannot execute.',
  h3: 'MiniMax H3 uses NVFP4/AWQ model files that Apple Metal cannot execute.',
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
  h3: 'MiniMax H3 uses NVFP4/AWQ model files that AMD ROCm cannot execute.',
  'wan-animate2': 'Wan Animate 2 currently uses the official NVIDIA INT8 ConvRot diffusion model, which AMD ROCm cannot execute.',
});

const AMD_UNSUPPORTED_COMPONENTS = Object.freeze({
  ltx25: AMD_UNSUPPORTED_ENGINES.ltx25,
  h3: AMD_UNSUPPORTED_ENGINES.h3,
  h3r2v: AMD_UNSUPPORTED_ENGINES.h3,
  h3turbo: AMD_UNSUPPORTED_ENGINES.h3,
  wananimate2: AMD_UNSUPPORTED_ENGINES['wan-animate2'],
});

// Intel Arc: INT8 ConvRot DiT weights DO work on Arc (ComfyUI exposes native
// convrot ops on XPU). LTX 2.5 uses Gemma 4 12B INT8 ConvRot text encoder
// (15.4 GB). The native CLIPLoader with type="ltxv" routes through
// comfy.text_encoders.gemma4.Gemma4Model → SD1ClipModel → mixed_precision_ops,
// which properly handles .comfy_quant metadata. The ComfyUI-LTXVideo custom
// node's LTXVGemmaCLIPModelLoader.py (which uses Gemma3ForConditionalGeneration
// from_pretrained) is NOT used by the Mix Studio graph builder.
//
// VRAM: TE (15.4 GB INT8) + DiT (21.5 GB INT8) + VAE (2 GB) = ~39 GB,
// exceeds 32 GB Arc B70. Solution: load_text_encoder with device: "cpu".
const INTEL_UNSUPPORTED_ENGINES = Object.freeze({
  ltx25quality: 'LTX 2.5 quality tier uses a 36 GB BF16 text encoder that exceeds Intel Arc VRAM capacity.',
});

const INTEL_UNSUPPORTED_COMPONENTS = Object.freeze({
  ltx25quality: INTEL_UNSUPPORTED_ENGINES.ltx25quality,
});

const APPLE_LTX_REFINE_TOKEN_LIMIT = 45_000;
const GENERAL_LTX_REFINE_TOKEN_LIMIT = 200_000;

const LTX_VRAM_SAFETY_MARGIN_GB = 3.0;

// ---------------------------------------------------------------------------
// MiniMax H3 VRAM budgeting
//
// CALIBRATED against measured peaks on an Intel Arc Pro B70 (31.16 GB usable),
// ComfyUI with sequential CPU offload — see docs/h3-resource-management-plan.md
// §10 for the full sweep:
//
//   256x256 x5     19.47 GB      512x320 x56    19.55 GB
//   384x256 x22    19.47 GB      608x384 x90    19.58 GB
//                                608x384 x124   19.58 GB
//
// Two facts drive this model:
//
//  1. The cost is almost entirely a FIXED model working set. A 22x increase in
//     pixels and a 25x increase in frames moved the peak by 0.11 GB. The latent
//     and attention workspace are megabytes, not gigabytes, so clip length is
//     NOT a useful lever for fitting H3 into VRAM.
//  2. The fixed floor measured ~19.5 GB, about 2.5 GB above the first
//     uncalibrated estimate. Component figures below are scaled to match the
//     measurement rather than left as guesses.
//
// Latent geometry mirrors ComfyUI's own node exactly
// (comfy_extras/nodes_minimax_h3.py): [B, 24, latent_t, height // 16,
// width // 16] at 24 fps.
// ---------------------------------------------------------------------------
const H3_VRAM_SAFETY_MARGIN_GB = 2.5;
const H3_LATENT_CHANNELS = 24;
const H3_SPATIAL_DIVISOR = 16;
const H3_FPS_FOR_PREFLIGHT = 24;
// Frames must sit on the model's 17k+5 grid; the smallest useful clip is 5.
const H3_MIN_FRAMES = 5;

/** ComfyUI's video_latent_t (nodes_minimax_h3.py:39-40). */
function h3VideoLatentT(frameCount) {
  const frames = Math.max(1, Math.floor(Number(frameCount) || 1));
  if (frames <= 5) return 2;
  return Math.floor((frames - 5) / 17) * 5 + 2;
}

/**
 * Estimated peak VRAM (GB) for an H3 job.
 *
 * Component figures are the resident working set under sequential CPU offload.
 * They are deliberately conservative: blocking a job that would have squeezed
 * through is recoverable, while overcommitting the GPU on a machine with TDR
 * disabled hangs the whole system.
 *
 * @param {object} opts { width, height, frames, mode, turbo }
 * @returns {number} estimated peak GB (always finite and > 0)
 */
function h3VramEstimate(opts = {}) {
  const o = (opts && typeof opts === 'object') ? opts : {};
  const width = Math.max(32, Math.floor(Number(o.width) || 608));
  const height = Math.max(32, Math.floor(Number(o.height) || 384));
  const frames = Math.max(H3_MIN_FRAMES, Math.floor(Number(o.frames) || H3_MIN_FRAMES));
  const reference = o.mode === 'reference';

  const latentT = h3VideoLatentT(frames);
  const latentW = Math.max(1, Math.floor(width / H3_SPATIAL_DIVISOR));
  const latentH = Math.max(1, Math.floor(height / H3_SPATIAL_DIVISOR));
  // fp16 latent: 2 bytes/element. Several copies are live at once during
  // sampling (sample + model output + residual), hence the multiplier.
  const latentGb = (H3_LATENT_CHANNELS * latentT * latentH * latentW * 2 * 4) / 1e9;

  // Attention workspace scales with the token count (latent_t * spatial).
  const tokens = latentT * latentH * latentW;
  const attentionGb = Math.min(8.0, (tokens * 2 * 512) / 1e9);

  const components = {
    // Calibrated to the measured ~19.5 GB floor (see header). Under sequential
    // offload the DiT is streamed, so this is its resident share rather than
    // the full 19.5 GB on disk.
    dit: reference ? 8.5 : 7.5,
    textEncoder: 4.5,
    videoVae: 3.0,
    audioVae: reference ? 0.6 : 0.0,
    turboLora: o.turbo ? 0.8 : 0.0,
    latent: latentGb,
    attention: attentionGb,
    overhead: 2.5,
  };

  const total = Object.values(components).reduce((sum, v) => sum + (Number(v) || 0), 0)
    + H3_VRAM_SAFETY_MARGIN_GB;
  return Number.isFinite(total) && total > 0 ? total : H3_VRAM_SAFETY_MARGIN_GB;
}

/**
 * Preflight an H3 job against available VRAM.
 *
 * Mirrors ltxRefinePreflight: returns `{ ok: true }` when the job may proceed
 * (with `needsLowVram` when headroom is thin), or `{ ok: false, error, ... }`
 * with a concrete smaller suggestion when it cannot fit.
 *
 * When free VRAM is unknown this always passes — guessing from absent telemetry
 * would block legitimate work.
 */
function h3Preflight(opts = {}) {
  const o = (opts && typeof opts === 'object') ? opts : {};
  const profile = (o.profile && typeof o.profile === 'object') ? o.profile : {};
  const width = Math.max(32, Math.floor(Number(o.width) || 608));
  const height = Math.max(32, Math.floor(Number(o.height) || 384));
  const frames = Math.max(H3_MIN_FRAMES, Math.floor(Number(o.frames) || H3_MIN_FRAMES));
  const fps = Math.max(1, Number(o.fps) || H3_FPS_FOR_PREFLIGHT);

  const requiredGb = h3VramEstimate({ ...o, width, height, frames });
  const vramFreeGb = Number(profile.vramFreeGb);

  if (!Number.isFinite(vramFreeGb) || vramFreeGb <= 0) {
    return { ok: true, requiredGb };
  }

  if (requiredGb > vramFreeGb) {
    // IMPORTANT (measured, see docs/h3-resource-management-plan.md §9):
    // H3's cost is dominated by the fixed model working set, NOT by clip
    // length. The estimate is ~17.0 GB at 32x32x5 and only ~17.5 GB at
    // 1344x768x362 — the latent and attention workspace are megabytes, not
    // gigabytes. So "shorten your clip" is almost never real advice for H3,
    // and suggesting it would send the user down a dead end.
    //
    // Only offer a frame-count reduction when it genuinely closes the gap;
    // otherwise be honest that the floor itself is the problem.
    let maxFrames = null;
    for (let candidate = frames - 17; candidate >= H3_MIN_FRAMES; candidate -= 17) {
      if (h3VramEstimate({ ...o, width, height, frames: candidate }) <= vramFreeGb) {
        maxFrames = candidate;
        break;
      }
    }

    if (maxFrames !== null) {
      const maxSeconds = Math.max(0.1, Math.floor((maxFrames / fps) * 10) / 10);
      return {
        ok: false,
        requiredGb,
        vramFreeGb,
        maxFrames,
        maxSeconds,
        floorGb: h3VramEstimate({ ...o, width: 32, height: 32, frames: H3_MIN_FRAMES }),
        error: `This MiniMax H3 request needs about ${requiredGb.toFixed(1)} GB but only `
          + `${vramFreeGb.toFixed(1)} GB of VRAM is free. Use ${maxFrames} frames `
          + `(~${maxSeconds}s) or less at ${width} × ${height}.`,
      };
    }

    // The floor itself does not fit. Say so plainly and quantify the shortfall.
    const floorGb = h3VramEstimate({ ...o, width: 32, height: 32, frames: H3_MIN_FRAMES });
    const shortfallGb = floorGb - vramFreeGb;
    return {
      ok: false,
      requiredGb,
      vramFreeGb,
      maxFrames: null,
      maxSeconds: null,
      floorGb,
      shortfallGb,
      error: `MiniMax H3 needs about ${floorGb.toFixed(1)} GB of VRAM before any clip length is `
        + `considered, but only ${vramFreeGb.toFixed(1)} GB is free (about `
        + `${shortfallGb.toFixed(1)} GB short). Shortening the clip will not help — its models are `
        + `the cost. Close other GPU applications to free VRAM, or use a lighter engine `
        + `such as LTX or Wan.`,
    };
  }

  return {
    ok: true,
    requiredGb,
    vramFreeGb,
    needsLowVram: requiredGb > vramFreeGb * 0.85,
  };
}

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
  const intel = intelGenerationProfile(profile);
  const result = {};
  for (const engine of ['ltx', 'ltx25', 'h3', 'ltx-edit', 'eros', 'wan', 'wan-animate2', 'scail', 'ltx25quality']) {
    const reason = apple
      ? APPLE_UNSUPPORTED_ENGINES[engine]
      : (amd ? AMD_UNSUPPORTED_ENGINES[engine]
      : (intel ? INTEL_UNSUPPORTED_ENGINES[engine] : ''));
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
  return Object.fromEntries(['ltx', 'ltx25', 'ltx25quality', 'h3', 'ltx-edit', 'eros', 'wan', 'wan-animate2', 'scail']
    .map((engine) => [engine, configuredVideoEngineCapability(engine, profile, settings)]));
}

function dependencyComponentBlock(componentId, profile = {}) {
  if (appleGenerationProfile(profile)) return APPLE_UNSUPPORTED_COMPONENTS[componentId] || '';
  if (amdGenerationProfile(profile)) return AMD_UNSUPPORTED_COMPONENTS[componentId] || '';
  if (intelGenerationProfile(profile)) return INTEL_UNSUPPORTED_COMPONENTS[componentId] || '';
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
  H3_VRAM_SAFETY_MARGIN_GB,
  INTEL_UNSUPPORTED_COMPONENTS,
  INTEL_UNSUPPORTED_ENGINES,
  appleGenerationProfile,
  amdGenerationProfile,
  configuredVideoEngineCapability,
  configuredVideoEngineCapabilities,
  dependencyComponentBlock,
  fp8Model,
  h3Preflight,
  h3VideoLatentT,
  h3VramEstimate,
  intelGenerationProfile,
  ltxRefinePreflight,
  ltxRefineTokenCount,
  ltxVramEstimate,
  videoEngineCapabilities,
};
