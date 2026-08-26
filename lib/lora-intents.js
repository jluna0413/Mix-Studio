'use strict';
/*
 * LoRA intent metadata — turns the flat filename picker into a
 * non-expert-legible, intent-grouped browser.
 *
 * Each entry maps a LoRA filename to:
 *   intent  : grouping bucket shown as a section header
 *   label   : friendly name (replaces the raw filename in the picker)
 *   blurb   : one plain-English line a beginner understands
 *   strength: suggested default strength (0..1.5)
 *
 * Driven by lib/lora-compat.js classifications already verified on this rig:
 *   qwen-edit, klein9, klein4, krea2, video, h3, unknown
 */

const LORA_INTENTS = {
  // ---- Identity ----
  'Best_FaceID_v1.0_LoRA.safetensors': {
    intent: 'Identity',
    intentLabel: 'Face & Identity',
    label: 'Best FaceID',
    blurb: 'Locks a specific face so edits and videos keep the same person.',
    strength: 0.9,
  },
  'gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors': {
    intent: 'Infra',
    intentLabel: 'Engine (auto)',
    label: 'Gemma 3 Text Tuning',
    blurb: 'Engine text-encoder adjustment — applied for you, no need to add.',
    strength: 1.0,
  },
  'krea2_identity_edit_v1_1_r128.safetensors': {
    intent: 'Identity',
    intentLabel: 'Face & Identity',
    label: 'Krea 2 Identity',
    blurb: 'Locks a face for Krea 2 identity edits.',
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
  // ---- Consistency ----
  'f2k_9B_lcs_consist_20260415.safetensors': {
    intent: 'Consistency',
    intentLabel: 'Subject Consistency',
    label: 'Klein 9B Consistency',
    blurb: 'Keeps a character/object looking the same across multiple Klein 9B edits.',
    strength: 0.8,
  },
  'f2k_4B_consist_20260314.safetensors': {
    intent: 'Consistency',
    intentLabel: 'Subject Consistency',
    label: 'Klein 4B Consistency',
    blurb: 'Same-subject lock for faster Klein 4B edits.',
    strength: 0.8,
  },
  // ---- Depth ----
  'depth-control-lora.safetensors': {
    intent: 'Depth',
    intentLabel: 'Depth & Structure',
    label: 'Depth Control',
    blurb: 'Guides the edit using a depth map — keeps layout/shape stable.',
    strength: 1.0,
  },
  // ---- Motion (video) ----
  'ltx-2.3-22b-distilled-1.1_lora-dynamic_fro09_avg_rank_111_bf16.safetensors': {
    intent: 'Motion',
    intentLabel: 'Video Motion',
    label: 'Dynamic Motion',
    blurb: 'Adds livelier, more energetic movement to LTX video.',
    strength: 0.7,
  },
  'ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors': {
    intent: 'Motion',
    intentLabel: 'Video Motion',
    label: 'Motion Track',
    blurb: 'Controls which way the camera/subject moves in the video.',
    strength: 0.6,
  },
  'ltx-2.3-22b-lora-cinemagraph-0.9.safetensors': {
    intent: 'Motion',
    intentLabel: 'Video Motion',
    label: 'Cinemagraph',
    blurb: 'Loops a small region of motion (living-photo effect).',
    strength: 0.8,
  },
  'lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors': {
    intent: 'Motion',
    intentLabel: 'Video Motion',
    label: 'LightX2V I2V',
    blurb: 'Turns a still image into video with a camera move.',
    strength: 0.7,
  },
  'edit_anything_v1.1_r256.safetensors': {
    intent: 'Segment',
    intentLabel: 'Select & Mask',
    label: 'Edit Anything',
    blurb: 'Segments any object by description so you can edit just that part.',
    strength: 1.0,
  },
  // ---- Style ----
  'Caricatures_V2-000007.safetensors': {
    intent: 'Style',
    intentLabel: 'Style & Look',
    label: 'Caricatures',
    blurb: 'Pushes the result toward a playful, exaggerated caricature.',
    strength: 0.9,
  },
  // ---- Text / infra ----
  'gemma-3-27b_text_encoder_lora.safetensors': {
    intent: 'Infra',
    intentLabel: 'Engine (auto)',
    description: 'Text-encoder tuning used by the engine automatically.',
    blurb: 'Engine text-encoder adjustment — applied for you, no need to add.',
    strength: 1.0,
  },
};

const LORA_INTENT_ORDER = [
  'Identity', 'Consistency', 'Depth', 'Motion', 'Segment', 'Style', 'Infra', 'Other',
];

function intentForLora(name) {
  if (!name) return null;
  return LORA_INTENTS[name] || null;
}

function prettyIntent(name) {
  const it = intentForLora(name);
  if (it) return it.label;
  // Fallback: trim the filename into something readable.
  return name.replace(/\.safetensors$/i, '').replace(/[_-]+/g, ' ').trim();
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
  return it && typeof it.strength === 'number' ? it.strength : null;
}

module.exports = {
  LORA_INTENTS,
  LORA_INTENT_ORDER,
  intentForLora,
  prettyIntent,
  intentBlurb,
  intentGroup,
  intentSuggestedStrength,
};

// Browser global exposure (loaded via <script> before app.js — no module system).
if (typeof window !== 'undefined') {
  window.LORA_INTENTS = LORA_INTENTS;
  window.LORA_INTENT_ORDER = LORA_INTENT_ORDER;
  window.intentForLora = intentForLora;
  window.prettyIntent = prettyIntent;
  window.intentBlurb = intentBlurb;
  window.intentGroup = intentGroup;
  window.intentSuggestedStrength = intentSuggestedStrength;
}
