'use strict';

const path = require('path');

/* ------------------------------------------------------------------ */
/* Model catalog for Intel Arc video upscaling                          */
/* ------------------------------------------------------------------ */

const REAL_ESRGAN_MODELS = Object.freeze({
  'realesrgan-x4': { label: 'Real-ESRGAN x4', ext: '.pth', scale: 4, description: 'General purpose 4x upscaler for realistic footage' },
  'realesrgan-anime': { label: 'Real-ESRGAN Anime', ext: '.pth', scale: 4, description: 'Optimized for anime/2D animation with detail enhancement' },
  'realesrgan-x2': { label: 'Real-ESRGAN x2', ext: '.pth', scale: 2, description: 'Fast 2x upscaler for quick previews' },
  'cugan-2x': { label: 'Real-CUGAN 2x', ext: '.pth', scale: 2, description: 'Anime-focused noise reduction and 2x upscale' },
  'cugan-3x': { label: 'Real-CUGAN 3x', ext: '.pth', scale: 3, description: 'Anime-focused 3x upscale with denoising' },
  'cugan-4x': { label: 'Real-CUGAN 4x', ext: '.pth', scale: 4, description: 'Anime-focused 4x upscale with maximum detail recovery' },
});

const ANIME4K_MODELS = Object.freeze({
  'anime4k-uuper': { label: 'Anime4K UUper', scale: 4, description: 'Highest quality anime upscaling with detail preservation' },
  'anime4k-upsc': { label: 'Anime4K Upsc', scale: 4, description: 'Balanced anime upscaling with speed optimization' },
  'anime4k-udnl': { label: 'Anime4K Udnl', scale: 4, description: 'Detail-preserving anime upscaling with denoising' },
  'anime4k-uups': { label: 'Anime4K Uups', scale: 4, description: 'Fast anime upscaling with minimal artifacts' },
});

const MODEL_KEYS = Object.keys(REAL_ESRGAN_MODELS);
const ANIME4K_KEYS = Object.keys(ANIME4K_MODELS);
const EDVR_MODELS = Object.freeze({
  'edvr-v2': { label: 'EDVR v2', ext: '.pth', description: 'Multi-scale multi-level pyramid downscaling for video restoration' },
  'edvr-net': { label: 'EDVR Net', ext: '.pth', description: 'Single-scale EDVR for blur and noise removal' },
});
const ANIMATE_DIFF_MODELS = Object.freeze({
  'animatediff-v3-2': { label: 'AnimateDiff v3.2', ext: '.safetensors', description: 'Temporal attention modules for detail consistency across frames' },
  'animatediff-rw-v2': { label: 'AnimateDiff RW v2', ext: '.safetensors', description: 'Motion module for realistic video upscaling' },
});
const ANIMATE_DIFF_KEYS = Object.keys(ANIMATE_DIFF_MODELS);
const ALL_UPSCALE_MODELS = [...MODEL_KEYS, ...ANIME4K_KEYS, ...Object.keys(EDVR_MODELS), ...ANIMATE_DIFF_KEYS];

/* ------------------------------------------------------------------ */
/* Helpers                                                              */
/* ------------------------------------------------------------------ */

function clampNumber(value, min, max, fallback) {
  const n = Number(value);
  return Number.isFinite(n) ? Math.min(max, Math.max(min, n)) : fallback;
}

function ensureEvenDimensions(width, height) {
  return {
    W: Math.round(width / 2) * 2,
    H: Math.round(height / 2) * 2,
  };
}

function targetResolutionForUpscale(videoWidth, videoHeight, scaleFactor) {
  const scale = clampNumber(scaleFactor, 1.5, 4, 2);
  return ensureEvenDimensions(Math.round(videoWidth * scale), Math.round(videoHeight * scale));
}

/* ------------------------------------------------------------------ */
/* Real-ESRGAN / Real-CUGAN builder                                     */
/* Frame-by-frame CNN upscaler — works on Intel Arc via xpu            */
/* ------------------------------------------------------------------ */

function buildRealEsrGANVideoUpscaleGraph(videoName, opts = {}) {
  const modelKey = String(opts.model || 'realesrgan-x4');
  const config = REAL_ESRGAN_MODELS[modelKey] || REAL_ESRGAN_MODELS['realesrgan-x4'];
  const scaleFactor = clampNumber(opts.scaleFactor, 1.5, 4, config.scale);
  const upscaleModel = opts.upscaleModel || config.ext.replace('.', '') + '.pth';
  const tileSize = clampNumber(opts.tileSize, 256, 1024, 512);

  return {
    load_video: {
      class_type: 'VHS_LoadVideo',
      inputs: {
        video: videoName,
        force_rate: opts.fps || 0,
        custom_width: 0,
        custom_height: 0,
        frame_load_cap: opts.frames || 0,
        skip_first_frames: 0,
        select_every_nth: 1,
        format: 'None',
      },
    },
    upscale_model: {
      class_type: 'UpscaleModelLoader',
      inputs: { model_name: upscaleModel },
    },
    upscale: {
      class_type: 'ImageUpscaleWithModel',
      inputs: {
        image: ['load_video', 0],
        model: ['upscale_model', 0],
        tile: tileSize,
        overlap: opts.tileOverlap || tileSize / 4,
        temporary_decode: true,
      },
    },
    resize: {
      class_type: 'ImageScaleBy',
      inputs: {
        image: ['upscale', 0],
        scale_factor: scaleFactor / config.scale,
        crop: 'disabled',
      },
    },
    video: {
      class_type: 'CreateVideoFromImages',
      inputs: {
        images: ['resize', 0],
        fps: opts.fps || 16,
        crf: opts.crf !== undefined ? clampNumber(opts.crf, 0, 51, 18) : 18,
      },
    },
    save: {
      class_type: 'SaveVideo',
      inputs: {
        video: ['video', 0],
        filename_prefix: opts.filenamePrefix || 'KreaStudio/video_upscale_esrgan',
        format: 'auto',
        codec: 'auto',
      },
    },
  };
}

/* ------------------------------------------------------------------ */
/* BasicVSR++ temporal builder                                          */
/* Multi-frame VSR — processes chunks for temporal consistency          */
/* ------------------------------------------------------------------ */

function buildBasicVSRVideoUpscaleGraph(videoName, opts = {}) {
  const scaleFactor = clampNumber(opts.scaleFactor, 1.5, 4, 2);
  const chunkSize = clampNumber(opts.chunkSize, 8, 32, 16);
  const overlap = clampNumber(opts.overlap || Math.floor(chunkSize / 4), 0, chunkSize - 1, 0);

  return {
    load_video: {
      class_type: 'VHS_LoadVideo',
      inputs: {
        video: videoName,
        force_rate: opts.fps || 0,
        custom_width: 0,
        custom_height: 0,
        frame_load_cap: opts.frames || 0,
        skip_first_frames: 0,
        select_every_nth: 1,
        format: 'None',
      },
    },
    vsr_model: {
      class_type: 'BasicVSRLoader',
      inputs: { model_name: opts.vsrModel || 'RealBasicVSR_v2.pth' },
    },
    vsr_process: {
      class_type: 'BasicVSRProcess',
      inputs: {
        frames: ['load_video', 0],
        model: ['vsr_model', 0],
        scale: scaleFactor,
        chunk_size: chunkSize,
        overlap: overlap,
      },
    },
    video: {
      class_type: 'CreateVideoFromImages',
      inputs: {
        images: ['vsr_process', 0],
        fps: opts.fps || 16,
        crf: opts.crf !== undefined ? clampNumber(opts.crf, 0, 51, 18) : 18,
      },
    },
    save: {
      class_type: 'SaveVideo',
      inputs: {
        video: ['video', 0],
        filename_prefix: opts.filenamePrefix || 'KreaStudio/video_upscale_basicvsr',
        format: 'auto',
        codec: 'auto',
      },
    },
  };
}

/* ------------------------------------------------------------------ */
/* Diffusion-based upscaler (SUPIR / ControlNet Tile)                   */
/* Highest quality, most VRAM-intensive                                 */
/* ------------------------------------------------------------------ */

function buildDiffusionVideoUpscaleGraph(videoName, opts = {}) {
  const scaleFactor = clampNumber(opts.scaleFactor, 1.5, 4, 2);
  const tileSize = clampNumber(opts.tileSize, 256, 1024, 512);
  const denoise = clampNumber(opts.denoise, 0.05, 0.35, 0.22);

  return {
    load_video: {
      class_type: 'VHS_LoadVideo',
      inputs: {
        video: videoName,
        force_rate: opts.fps || 0,
        custom_width: 0,
        custom_height: 0,
        frame_load_cap: opts.frames || 0,
        skip_first_frames: 0,
        select_every_nth: 1,
        format: 'None',
      },
    },
    upscale_model: {
      class_type: 'UpscaleModelLoader',
      inputs: { model_name: opts.upscaleModel || '4x_foolhardy_Remacri.pth' },
    },
    pre_upscale: {
      class_type: 'ImageUpscaleWithModel',
      inputs: {
        image: ['load_video', 0],
        model: ['upscale_model', 0],
        tile: tileSize * 2,
        overlap: (tileSize * 2) / 4,
        temporary_decode: true,
      },
    },
    upscale_tile: {
      class_type: 'UltimateSDUpscale',
      inputs: {
        image: ['pre_upscale', 0],
        model: opts.ditRef || ['dit', 0],
        positive: opts.posRef || ['pos', 0],
        negative: opts.negRef || ['neg', 0],
        vae: opts.vaeRef || ['vae', 0],
        upscale_by: scaleFactor,
        seed: clampNumber(opts.seed, 0, 2147483647, 42),
        steps: 12,
        cfg: 1,
        sampler_name: 'euler',
        scheduler: 'beta',
        denoise,
        upscale_model: ['upscale_model', 0],
        mode_type: 'Chess',
        tile_width: tileSize,
        tile_height: tileSize,
        mask_blur: 8,
        tile_padding: 64,
        seam_fix_mode: 'None',
        seam_fix_denoise: 0.1,
        seam_fix_width: 32,
        seam_fix_mask_blur: 8,
        seam_fix_padding: 16,
        force_uniform_tiles: true,
        tiled_decode: true,
        batch_size: 1,
      },
    },
    video: {
      class_type: 'CreateVideoFromImages',
      inputs: {
        images: ['upscale_tile', 0],
        fps: opts.fps || 16,
        crf: opts.crf !== undefined ? clampNumber(opts.crf, 0, 51, 18) : 18,
      },
    },
    save: {
      class_type: 'SaveVideo',
      inputs: {
        video: ['video', 0],
        filename_prefix: opts.filenamePrefix || 'KreaStudio/video_upscale_diffusion',
        format: 'auto',
        codec: 'auto',
      },
    },
  };
}

/* ------------------------------------------------------------------ */
/* Anime4K shader-based builder                                         */
/* Real-time GLSL/Vulkan upscaling — zero hallucination, extremely fast */
/* ------------------------------------------------------------------ */

function buildAnime4KVideoUpscaleGraph(videoName, opts = {}) {
  const modelKey = String(opts.model || 'anime4k-uuper');
  const config = ANIME4K_MODELS[modelKey] || ANIME4K_MODELS['anime4k-uuper'];
  const scaleFactor = clampNumber(opts.scaleFactor, 1.5, 4, config.scale);

  return {
    load_video: {
      class_type: 'VHS_LoadVideo',
      inputs: {
        video: videoName,
        force_rate: opts.fps || 0,
        custom_width: 0,
        custom_height: 0,
        frame_load_cap: opts.frames || 0,
        skip_first_frames: 0,
        select_every_nth: 1,
        format: 'None',
      },
    },
    anime4k: {
      class_type: 'Anime4K',
      inputs: {
        image: ['load_video', 0],
        model: modelKey,
        scale: scaleFactor,
        glsl: true,
      },
    },
    video: {
      class_type: 'CreateVideoFromImages',
      inputs: {
        images: ['anime4k', 0],
        fps: opts.fps || 16,
        crf: opts.crf !== undefined ? clampNumber(opts.crf, 0, 51, 18) : 18,
      },
    },
    save: {
      class_type: 'SaveVideo',
      inputs: {
        video: ['video', 0],
        filename_prefix: opts.filenamePrefix || 'KreaStudio/video_upscale_anime4k',
        format: 'auto',
        codec: 'auto',
      },
    },
  };
}

/* ------------------------------------------------------------------ */
/* AnimateDiff Video-to-Video Upscale                                   */
/* Temporal attention modules for detail consistency across frames      */
/* ------------------------------------------------------------------ */

function buildAnimateDiffVideoUpscaleGraph(videoName, opts = {}) {
  const scaleFactor = clampNumber(opts.scaleFactor, 1.5, 4, 2);
  const tileSize = clampNumber(opts.tileSize, 256, 1024, 512);
  const denoise = clampNumber(opts.denoise, 0.05, 0.35, 0.22);
  const frames = clampNumber(opts.frames || 0, 1, 512, 0);

  return {
    load_video: {
      class_type: 'VHS_LoadVideo',
      inputs: {
        video: videoName,
        force_rate: opts.fps || 0,
        custom_width: 0,
        custom_height: 0,
        frame_load_cap: frames,
        skip_first_frames: 0,
        select_every_nth: 1,
        format: 'None',
      },
    },
    animate_model: {
      class_type: 'AnimateDiffModelLoader',
      inputs: {
        animate_model: opts.animateModel || 'animatediff-v3-2.safetensors',
      },
    },
    dit_ref: {
      class_type: 'CheckpointLoaderSimple',
      inputs: {
        ckpt_name: opts.ditRef || 'anything_v5.safetensors',
      },
    },
    pos_ref: {
      class_type: 'CLIPTextEncode',
      inputs: {
        text: opts.positivePrompt || 'high quality, detailed, 4k',
        clip: ['dit_ref', 1],
      },
    },
    neg_ref: {
      class_type: 'CLIPTextEncode',
      inputs: {
        text: opts.negativePrompt || 'low quality, blurry, distorted',
        clip: ['dit_ref', 1],
      },
    },
    vae_ref: {
      class_type: 'VAELoader',
      inputs: {
        vae_name: opts.vaeRef || 'vae-ft-ema-840000-ema.safetensors',
      },
    },
    animate_process: {
      class_type: 'AnimateDiffApply',
      inputs: {
        frames: ['load_video', 0],
        model: ['animate_model', 0],
        positive: ['pos_ref', 0],
        negative: ['neg_ref', 0],
        vae: ['vae_ref', 0],
        scale: scaleFactor,
        seed: clampNumber(opts.seed, 0, 2147483647, 42),
        steps: 12,
        cfg: 1,
        sampler_name: 'euler',
        scheduler: 'beta',
        denoise,
        batch_size: 1,
      },
    },
    video: {
      class_type: 'CreateVideoFromImages',
      inputs: {
        images: ['animate_process', 0],
        fps: opts.fps || 16,
        crf: opts.crf !== undefined ? clampNumber(opts.crf, 0, 51, 18) : 18,
      },
    },
    save: {
      class_type: 'SaveVideo',
      inputs: {
        video: ['video', 0],
        filename_prefix: opts.filenamePrefix || 'KreaStudio/video_upscale_animatediff',
        format: 'auto',
        codec: 'auto',
      },
    },
  };
}

/* ------------------------------------------------------------------ */
/* EDVR temporal builder                                                */
/* Multi-frame VSR — processes chunks for temporal consistency          */
/* ------------------------------------------------------------------ */

function buildEdvrVideoUpscaleGraph(videoName, opts = {}) {
  const modelKey = String(opts.model || 'edvr-v2');
  const config = EDVR_MODELS[modelKey] || EDVR_MODELS['edvr-v2'];
  const scaleFactor = clampNumber(opts.scaleFactor, 1.5, 4, 4);
  const chunkSize = clampNumber(opts.chunkSize, 8, 32, 16);
  const overlap = clampNumber(opts.overlap || Math.floor(chunkSize / 4), 0, chunkSize - 1, 0);

  return {
    load_video: {
      class_type: 'VHS_LoadVideo',
      inputs: {
        video: videoName,
        force_rate: opts.fps || 0,
        custom_width: 0,
        custom_height: 0,
        frame_load_cap: opts.frames || 0,
        skip_first_frames: 0,
        select_every_nth: 1,
        format: 'None',
      },
    },
    edvr_model: {
      class_type: 'EDVRLoader',
      inputs: { model_name: opts.edvrModel || 'EDVR_v2.pth' },
    },
    edvr_process: {
      class_type: 'EDVRProcess',
      inputs: {
        frames: ['load_video', 0],
        model: ['edvr_model', 0],
        scale: scaleFactor,
        chunk_size: chunkSize,
        overlap: overlap,
      },
    },
    video: {
      class_type: 'CreateVideoFromImages',
      inputs: {
        images: ['edvr_process', 0],
        fps: opts.fps || 16,
        crf: opts.crf !== undefined ? clampNumber(opts.crf, 0, 51, 18) : 18,
      },
    },
    save: {
      class_type: 'SaveVideo',
      inputs: {
        video: ['video', 0],
        filename_prefix: opts.filenamePrefix || 'KreaStudio/video_upscale_edvr',
        format: 'auto',
        codec: 'auto',
      },
    },
  };
}

/* ------------------------------------------------------------------ */
/* Model selection logic                                                */
/* ------------------------------------------------------------------ */

function selectVideoUpscaleModel(modelKey, availableModels = []) {
  if (!modelKey || !ALL_UPSCALE_MODELS.includes(modelKey)) return null;
  // Check if it's an Anime4K model (no .pth extension)
  const isAnime4K = ANIME4K_KEYS.includes(modelKey);
  if (isAnime4K) {
    const config = ANIME4K_MODELS[modelKey];
    return availableModels.length === 0 ? config : null;
  }
  const config = REAL_ESRGAN_MODELS[modelKey];
  if (availableModels.length === 0) return config;
  // Verify model file is present (partial match on basename)
  const modelName = config.ext.replace('.', '') + '.pth';
  return availableModels.some((m) => String(m).toLowerCase().includes(modelName.toLowerCase())) ? config : null;
}

function selectAnime4KModel(modelKey) {
  if (!ANIME4K_KEYS.includes(modelKey)) return null;
  return ANIME4K_MODELS[modelKey];
}

function selectEdvrModel(modelKey) {
  if (!EDVR_KEYS.includes(modelKey)) return null;
  return EDVR_MODELS[modelKey];
}

function selectAnimateDiffModel(modelKey) {
  if (!ANIMATE_DIFF_KEYS.includes(modelKey)) return null;
  return ANIMATE_DIFF_MODELS[modelKey];
}

function getAvailableUpscaleModels(modelDirs, fsImpl = require('fs')) {
  const models = new Set();
  for (const dir of modelDirs || []) {
    if (!dir) continue;
    let entries = [];
    try {
      entries = fsImpl.readdirSync(dir);
    } catch {
      continue;
    }
    for (const entry of entries) {
      const name = typeof entry === 'string' ? entry : entry && entry.name;
      if (!name || name.endsWith('.download')) continue;
      const ext = path.extname(name).toLowerCase();
      if (ext === '.pth') models.add(name);
    }
  }
  return [...models];
}

/* ------------------------------------------------------------------ */
/* Public API                                                           */
/* ------------------------------------------------------------------ */

module.exports = {
  REAL_ESRGAN_MODELS,
  ANIME4K_MODELS,
  EDVR_MODELS,
  ANIMATE_DIFF_MODELS,
  MODEL_KEYS,
  ANIME4K_KEYS,
  EDVR_KEYS: Object.keys(EDVR_MODELS),
  ANIMATE_DIFF_KEYS,
  ALL_UPSCALE_MODELS,
  ensureEvenDimensions,
  targetResolutionForUpscale,
  buildRealEsrGANVideoUpscaleGraph,
  buildBasicVSRVideoUpscaleGraph,
  buildDiffusionVideoUpscaleGraph,
  buildAnime4KVideoUpscaleGraph,
  buildAnimateDiffVideoUpscaleGraph,
  buildEdvrVideoUpscaleGraph,
  selectVideoUpscaleModel,
  selectAnime4KModel,
  selectEdvrModel,
  selectAnimateDiffModel,
  getAvailableUpscaleModels,
};
