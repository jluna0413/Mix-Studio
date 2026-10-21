'use strict';
/*
 * LoRA intent metadata — turns the flat filename picker into a
 * non-expert-legible, intent-grouped browser.
 *
 * Each entry maps a LoRA filename to:
 *   intent     : grouping bucket key ('Speed', 'Identity', 'Camera', 'Motion', etc.)
 *   intentLabel: friendly group header ('Speed & Turbo', 'Face & Identity', etc.)
 *   label      : friendly human name (replaces raw filenames in picker)
 *   blurb      : one plain-English line explaining what the LoRA does
 *   strength   : suggested default strength (0..1.5)
 *
 * Automatically provides dynamic inference for ANY unlisted or future LoRA.
 */

const LORA_INTENTS = {
  // ---- Identity ----
  'Best_FaceID_v1.0_LoRA.safetensors': {
    intent: 'Identity',
    intentLabel: 'Face & Identity',
    label: 'Best FaceID',
    blurb: 'Locks facial features and likeness across generations and video clips.',
    strength: 0.9,
  },
  'krea2_identity_edit_v1_2.safetensors': {
    intent: 'Identity',
    intentLabel: 'Face & Identity',
    label: 'Krea 2 Identity v1.2',
    blurb: 'Locks face and likeness for Krea 2 edits and image generations.',
    strength: 1.0,
  },
  'krea2_identity_edit_v1_1_r128.safetensors': {
    intent: 'Identity',
    intentLabel: 'Face & Identity',
    label: 'Krea 2 Identity v1.1',
    blurb: 'Locks face and likeness for Krea 2 edits and image generations.',
    strength: 1.0,
  },
  'johnny-moon-poc.safetensors': {
    intent: 'Identity',
    intentLabel: 'Face & Identity',
    label: 'Johnny Moon (POC)',
    blurb: 'Personal identity LoRA — keeps this subject consistent.',
    strength: 1.0,
  },
  'johnny-moon-flux-poc.safetensors': {
    intent: 'Identity',
    intentLabel: 'Face & Identity',
    label: 'Johnny Moon (Klein)',
    blurb: 'Personal identity LoRA for Klein 4B edits.',
    strength: 1.0,
  },
  'johnny-moon-flux-poc-old.safetensors': {
    intent: 'Identity',
    intentLabel: 'Face & Identity',
    label: 'Johnny Moon (Klein, old)',
    blurb: 'Earlier personal identity LoRA for Klein 4B.',
    strength: 1.0,
  },

  // ---- Speed & Turbo ----
  'krea2_turbo_lora_rank_64_bf16.safetensors': {
    intent: 'Speed',
    intentLabel: 'Speed & Turbo',
    label: 'Krea 2 Turbo',
    blurb: 'Accelerates Krea 2 generation with reduced sampling steps.',
    strength: 1.0,
  },
  'Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors': {
    intent: 'Speed',
    intentLabel: 'Speed & Turbo',
    label: 'Qwen Edit Lightning 4-Step',
    blurb: 'Lightning-fast 4-step distilled editing for Qwen-Image-Edit 2511.',
    strength: 1.0,
  },
  'minimax_h3_fl2v_lightx2v_turbo_4step_v0.1_comfy_resized_avg_rank_21_bf16.safetensors': {
    intent: 'Speed',
    intentLabel: 'Speed & Turbo',
    label: 'MiniMax H3 4-Step Turbo',
    blurb: 'Ultra-fast 4-step turbo distillation for MiniMax H3 video.',
    strength: 1.0,
  },
  'minimax_h3_turbo_v4_step600_ema.safetensors': {
    intent: 'Speed',
    intentLabel: 'Speed & Turbo',
    label: 'MiniMax H3 Turbo v4',
    blurb: 'Fast 8-step EMA turbo adapter for high-quality MiniMax H3 video.',
    strength: 1.0,
  },
  'minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors': {
    intent: 'Speed',
    intentLabel: 'Speed & Turbo',
    label: 'MiniMax H3 Official 8-Step Turbo',
    blurb: 'Official Comfy-Org 8-step turbo acceleration LoRA for MiniMax H3.',
    strength: 1.0,
  },
  'Comfy-Org---MiniMax-H3/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors': {
    intent: 'Speed',
    intentLabel: 'Speed & Turbo',
    label: 'MiniMax H3 Official 8-Step Turbo',
    blurb: 'Official Comfy-Org 8-step turbo acceleration LoRA for MiniMax H3.',
    strength: 1.0,
  },
  'wan2.2_i2v_high_noise_lora_rank64_lightx2v_4step_1022.safetensors': {
    intent: 'Speed',
    intentLabel: 'Speed & Turbo',
    label: 'Wan 2.2 Turbo High-Noise',
    blurb: '4-step turbo acceleration for high-noise sampling stage of Wan 2.2.',
    strength: 1.0,
  },
  'wan2.2_i2v_low_noise_lora_rank64_lightx2v_4step_1022.safetensors': {
    intent: 'Speed',
    intentLabel: 'Speed & Turbo',
    label: 'Wan 2.2 Turbo Low-Noise',
    blurb: '4-step turbo acceleration for low-noise refinement stage of Wan 2.2.',
    strength: 1.0,
  },
  'ltx-2.3-22b-distilled-lora-384-1.1.safetensors': {
    intent: 'Speed',
    intentLabel: 'Speed & Turbo',
    label: 'LTX 2.3 Distilled 1.1',
    blurb: 'Fast distilled sampler for LTX 2.3 video generation at 384p.',
    strength: 0.8,
  },
  'ltx-2.3-22b-distilled-lora-384.safetensors': {
    intent: 'Speed',
    intentLabel: 'Speed & Turbo',
    label: 'LTX 2.3 Distilled (Base)',
    blurb: 'Base distilled LoRA for high-speed LTX 2.3 video generation.',
    strength: 0.8,
  },
  'LTX2.3_DMD_reshaped_r256.safetensors': {
    intent: 'Speed',
    intentLabel: 'Speed & Turbo',
    label: 'LTX 2.3 DMD Turbo',
    blurb: 'Distribution matching distillation for ultra-fast LTX 2.3 video.',
    strength: 0.8,
  },

  // ---- Camera & Angles ----
  'qwen_image_edit_2511_multiple-angles-lora.safetensors': {
    intent: 'Camera',
    intentLabel: 'Camera & Angles',
    label: 'Qwen Edit Multi-Angle',
    blurb: 'Enables camera angle shifts (turn around, low angle, overhead) in Qwen Edit.',
    strength: 0.8,
  },
  'LTX2.3-22B_IC-LoRA-Cameraman_v2_14000.safetensors': {
    intent: 'Camera',
    intentLabel: 'Camera & Angles',
    label: 'LTX Cameraman v2',
    blurb: 'Provides smooth, professional camera movements and cinematic framing.',
    strength: 0.7,
  },

  // ---- Video Motion ----
  'ltx-2.3-22b-distilled-1.1_lora-dynamic_fro09_avg_rank_111_bf16.safetensors': {
    intent: 'Motion',
    intentLabel: 'Video Motion',
    label: 'Dynamic Motion',
    blurb: 'Adds lively, energetic movement and action to LTX video clips.',
    strength: 0.7,
  },
  'ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors': {
    intent: 'Motion',
    intentLabel: 'Video Motion',
    label: 'Motion Track Control',
    blurb: 'Controls camera path and trajectory tracking for cinematic motion.',
    strength: 0.6,
  },
  'ltx-2.3-22b-lora-cinemagraph-0.9.safetensors': {
    intent: 'Motion',
    intentLabel: 'Video Motion',
    label: 'Cinemagraph Loop',
    blurb: 'Creates living-photo cinemagraphs with localized looping motion.',
    strength: 0.8,
  },
  'LTX-2.3-Motion-Enhancer-n4w.safetensors': {
    intent: 'Motion',
    intentLabel: 'Video Motion',
    label: 'LTX Motion Enhancer',
    blurb: 'Amplifies realism, fluidity, and natural dynamics of complex motions.',
    strength: 0.7,
  },
  'lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors': {
    intent: 'Motion',
    intentLabel: 'Video Motion',
    label: 'LightX2V I2V',
    blurb: 'Turns a still image into video with realistic camera and subject motion.',
    strength: 0.7,
  },

  // ---- Structure & Depth ----
  'depth-control-lora.safetensors': {
    intent: 'Control',
    intentLabel: 'Structure & Control',
    label: 'Depth Control',
    blurb: 'Guides generation with a depth map to keep shape and layout stable.',
    strength: 1.0,
  },

  // ---- Consistency ----
  'f2k_9B_lcs_consist_20260415.safetensors': {
    intent: 'Consistency',
    intentLabel: 'Subject Consistency',
    label: 'Klein 9B Consistency',
    blurb: 'Keeps character and object appearance consistent across Klein 9B edits.',
    strength: 0.8,
  },
  'f2k_4B_consist_20260314.safetensors': {
    intent: 'Consistency',
    intentLabel: 'Subject Consistency',
    label: 'Klein 4B Consistency',
    blurb: 'Same-subject lock for fast Klein 4B generations and edits.',
    strength: 0.8,
  },
  'ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors': {
    intent: 'Consistency',
    intentLabel: 'Subject Consistency',
    label: 'LTX Ingredients',
    blurb: 'Blends and preserves multiple input elements across video frames.',
    strength: 0.8,
  },

  // ---- Segment & Mask ----
  'edit_anything_v1.1_r256.safetensors': {
    intent: 'Segment',
    intentLabel: 'Select & Mask',
    label: 'Edit Anything',
    blurb: 'Segments objects by description for targeted, localized editing.',
    strength: 1.0,
  },

  // ---- Style & Look ----
  'Caricatures_V2-000007.safetensors': {
    intent: 'Style',
    intentLabel: 'Style & Look',
    label: 'Caricatures',
    blurb: 'Pushes generation toward playful, exaggerated caricature styling.',
    strength: 0.9,
  },

  // ---- Infra / Engine ----
  'gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors': {
    intent: 'Infra',
    intentLabel: 'Engine (auto)',
    label: 'Gemma 3 Text Tuning',
    blurb: 'Engine text-encoder adjustment — applied automatically, no need to add.',
    strength: 1.0,
  },
  'gemma-3-27b_text_encoder_lora.safetensors': {
    intent: 'Infra',
    intentLabel: 'Engine (auto)',
    label: 'Gemma 3 27B Text Encoder',
    blurb: 'Engine text-encoder adjustment — applied automatically, no need to add.',
    strength: 1.0,
  },
};

const LORA_INTENT_ORDER = [
  'Speed & Turbo',
  'Face & Identity',
  'Camera & Angles',
  'Video Motion',
  'Structure & Control',
  'Subject Consistency',
  'Select & Mask',
  'Style & Look',
  'General & Custom',
  'Engine (auto)',
  'Other',
];

function humanizeLoraName(filename) {
  if (!filename) return '';
  let base = String(filename).replace(/\\/g, '/').split('/').pop().trim();
  base = base.replace(/\.(safetensors|pt|bin|ckpt)$/i, '');
  base = base.replace(/_?(rank_?\d+|bf16|fp16|fp8|avg_rank_\d+|reshaped|r\d+|v\d+(\.\d+)?|step\d+|ema)/gi, ' ');
  base = base.replace(/[-_.]+/g, ' ');
  base = base.replace(/([a-z])([A-Z])/g, '$1 $2');
  base = base.replace(/\s+/g, ' ').trim();
  if (!base) return 'Custom LoRA';
  return base.split(' ').map((word) => {
    if (/^(h3|wan|ltx|dmd|ema|poc|kj|ic|i2v|t2v|v2|v3|v4|4b|9b|12b|14b|22b|27b)$/i.test(word)) {
      return word.toUpperCase();
    }
    return word.charAt(0).toUpperCase() + word.slice(1).toLowerCase();
  }).join(' ');
}

function inferLoraMetadata(name) {
  if (!name) return null;
  const lower = String(name).toLowerCase();

  let intent = 'Other';
  let intentLabel = 'General & Custom';
  let blurb = 'Custom LoRA adapter for enhanced generation.';
  let strength = 0.8;

  if (/turbo|lightning|distill|step|fast|speed|lcm|dmd|hyper/.test(lower)) {
    intent = 'Speed';
    intentLabel = 'Speed & Turbo';
    blurb = 'Accelerates generation with fewer sampling steps while preserving detail.';
    strength = 1.0;
  } else if (/face|identity|portrait|person|actor|character|subject|faceid/.test(lower)) {
    intent = 'Identity';
    intentLabel = 'Face & Identity';
    blurb = 'Locks subject identity, facial details, and character likeness.';
    strength = 0.9;
  } else if (/camera|angle|zoom|pan|tilt|cinematic|dolly|orbit|view|cameraman/.test(lower)) {
    intent = 'Camera';
    intentLabel = 'Camera & Angles';
    blurb = 'Controls camera movement, perspective angles, or cinematic framing.';
    strength = 0.75;
  } else if (/motion|dynamic|action|move|flow|cinemagraph|speed/.test(lower)) {
    intent = 'Motion';
    intentLabel = 'Video Motion';
    blurb = 'Controls movement dynamics, flow, and motion intensity in video.';
    strength = 0.7;
  } else if (/depth|lineart|canny|sketch|pose|openpose|control|structure|segment|mask/.test(lower)) {
    intent = 'Control';
    intentLabel = 'Structure & Control';
    blurb = 'Guides structure, composition, or targeted regions with spatial control.';
    strength = 1.0;
  } else if (/style|aesthetic|look|anime|art|realism|illustration|vintage|cyberpunk|pixel|retro|caricature/.test(lower)) {
    intent = 'Style';
    intentLabel = 'Style & Look';
    blurb = 'Applies a distinctive artistic style, color grade, or aesthetic look.';
    strength = 0.85;
  } else if (/consist|outfit|clothing|costume|ingredient/.test(lower)) {
    intent = 'Consistency';
    intentLabel = 'Subject Consistency';
    blurb = 'Maintains consistent character styling and elements across generations.';
    strength = 0.8;
  }

  return {
    intent,
    intentLabel,
    label: humanizeLoraName(name),
    blurb,
    strength,
    isDynamic: true,
  };
}

function normalizeLoraKey(name) {
  if (!name) return '';
  return String(name).replace(/\\/g, '/').split('/').pop().trim();
}

function intentForLora(name) {
  if (!name) return null;
  const direct = LORA_INTENTS[name];
  if (direct) return direct;
  const normalized = normalizeLoraKey(name);
  if (LORA_INTENTS[normalized]) return LORA_INTENTS[normalized];
  return inferLoraMetadata(name);
}

function prettyIntent(name) {
  const it = intentForLora(name);
  if (it && it.label) return it.label;
  return humanizeLoraName(name);
}

function intentBlurb(name) {
  const it = intentForLora(name);
  return it ? it.blurb : '';
}

function intentGroup(name) {
  const it = intentForLora(name);
  return it ? (it.intentLabel || 'Other') : 'Other';
}

function intentSuggestedStrength(name) {
  const it = intentForLora(name);
  return it && typeof it.strength === 'number' ? it.strength : 0.8;
}

const loraIntentsModule = {
  LORA_INTENTS,
  LORA_INTENT_ORDER,
  humanizeLoraName,
  inferLoraMetadata,
  intentForLora,
  prettyIntent,
  intentBlurb,
  intentGroup,
  intentSuggestedStrength,
};

if (typeof module !== 'undefined' && module.exports) {
  module.exports = loraIntentsModule;
}

if (typeof window !== 'undefined') {
  window.LORA_INTENTS = LORA_INTENTS;
  window.LORA_INTENT_ORDER = LORA_INTENT_ORDER;
  window.humanizeLoraName = humanizeLoraName;
  window.inferLoraMetadata = inferLoraMetadata;
  window.intentForLora = intentForLora;
  window.prettyIntent = prettyIntent;
  window.intentBlurb = intentBlurb;
  window.intentGroup = intentGroup;
  window.intentSuggestedStrength = intentSuggestedStrength;
}
