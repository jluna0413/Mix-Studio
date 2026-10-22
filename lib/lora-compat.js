'use strict';

const { H3_TURBO_LORAS } = require('./h3-turbo-presets');

function words(value) {
  return String(value || '').toLowerCase().replace(/[^a-z0-9]+/g, ' ');
}

function loraBasename(value) {
  return String(value || '').trim().replace(/\\/g, '/').split('/').pop().toLowerCase();
}

function compact(value) {
  return words(value).replace(/\s+/g, '');
}

function classifyLora(info) {
  const name = words(info && info.name);
  const metadata = info && info.metadata ? info.metadata : {};
  const metaText = words(Object.values(metadata).join(' '));
  const joined = `${name} ${metaText}`;
  const compactJoined = compact(joined);
  const keys = (info && Array.isArray(info.keys) ? info.keys : []).map(String);
  const keyText = keys.slice(0, 32).join('\n').toLowerCase();

  // Prefer an explicit MiniMax H3 model marker before generic transformer-key
  // heuristics. Some H3 training exports use key layouts that overlap with
  // other transformer families, while their filename/metadata still records
  // the correct base model.
  if (compactJoined.includes('minimaxh3') || /\bh3\b/.test(joined)
      || keyText.includes('.attn.qkv_proj.')) return 'h3';
  // Video-family markers must win over the generic transformer_blocks key
  // heuristic below: LTX/WAN/lightx2v exports also carry transformer_blocks.*
  // keys but belong to the video engines, not Qwen Edit.
  if (compactJoined.includes('lightx2v')) return 'video';
  if (/(^|\s)(ltx|ltxv|wan\d*|scail|videoboom|cinemagraph)\b/.test(name)
      || compactJoined.includes('ltx2')
      || compactJoined.includes('ltx23')
      || compactJoined.includes('ltx25')) return 'video';
  if (/(qwen).*(edit)|(edit).*(qwen)/.test(joined) ||
      keyText.includes('transformer.transformer_blocks') ||
      keyText.includes('transformer_blocks.')) {
    return 'qwen-edit';
  }
  if (compactJoined.includes('flux2klein9b') || compactJoined.includes('klein9b') || compactJoined.includes('f2k9b')) return 'klein9';
  if (compactJoined.includes('flux2klein4b') || compactJoined.includes('klein4b') || compactJoined.includes('f2k4b')) return 'klein4';
  if (compactJoined.includes('krea2') || keyText.includes('diffusion_model.blocks.0.attn')) return 'krea2';
  if (/\b(ltx|wan|scail|video)\b/.test(joined)) return 'video';
  return 'unknown';
}

function compatibleCategoriesForContext(view, editEngine, videoEngine) {
  if (view === 'create') return ['krea2', 'unknown'];
  if (view === 'video') return videoEngine === 'h3' ? ['h3', 'unknown'] : ['video', 'unknown'];
  if (view === 'edit') {
    if (editEngine === 'qwen') return ['qwen-edit', 'unknown'];
    if (editEngine === 'klein9') return ['klein9', 'unknown'];
    if (['krea2', 'krea2ref', 'krea2remix'].includes(editEngine)) return ['krea2', 'unknown'];
    return ['klein4', 'unknown'];
  }
  return ['unknown'];
}

function loraCompatibilityWarning(loras, loraInfo, view, editEngine, videoEngine) {
  const allowed = new Set(compatibleCategoriesForContext(view, editEngine, videoEngine));
  const bad = [];
  for (const lora of loras || []) {
    if (!lora || !lora.on || !lora.name || lora.managed) continue;
    const category = (loraInfo && loraInfo[lora.name] && loraInfo[lora.name].category) || 'unknown';
    if (!allowed.has(category)) bad.push(lora.name);
  }
  if (!bad.length) return '';
  return `These LoRAs may not affect this mode: ${bad.map((n) => n.replace(/\.safetensors$/i, '').split(/[\\/]/).pop()).join(', ')}`;
}

// H3 Turbo/distill adapters patch the same ~208 modules (attn.qkv_proj,
// out_proj, mlp.fc1/fc2) as the managed turbo LoRA. Stacking more than one
// (proven: 4-step @1.0 + 8-step @0.5 = 1.5x deltas over conflicting training
// targets) corrupts the sampling trajectory and produces heavy artifacts.
// Detection stays in the symbolic layer: deterministic filename patterns plus
// the known managed list, no probabilistic classification involved.
const H3_TURBO_ADAPTER_KNOWN = new Set(Object.values(H3_TURBO_LORAS).map(loraBasename));

function isH3TurboAdapterLora(name) {
  const base = loraBasename(name);
  if (!base) return false;
  if (H3_TURBO_ADAPTER_KNOWN.has(base)) return true;
  if (!/minimax[_-]?h3/.test(base)) return false;
  return /(turbo|lightx2v|distill|step\d+[_-]?(ema|ckpt)|ema[_-]?\d)/.test(base);
}

// Conflicts to reject before a job is created: a user-added H3 turbo adapter
// while the Turbo switch already applies its own (double-distill), and any
// user LoRA whose classified family is not H3/unknown (wrong-model-family,
// e.g. Wan-on-H3 where key overlap is 0/400 — a silent no-op that still
// wastes load time and hides user intent).
function h3TurboStackConflicts(loras, loraInfo, opts) {
  const turboActive = !!(opts && opts.turboActive);
  const managed = new Set(((opts && opts.managedTurboLoras) || []).map(loraBasename));
  const conflicts = [];
  for (const lora of loras || []) {
    if (!lora || !lora.on || !lora.name || lora.managed) continue;
    const base = loraBasename(lora.name);
    // The exact managed-name duplicate keeps its own dedicated error.
    if (managed.has(base)) continue;
    if (turboActive && isH3TurboAdapterLora(base)) {
      conflicts.push({ name: lora.name, reason: 'turbo' });
      continue;
    }
    const category = (loraInfo && loraInfo[lora.name] && loraInfo[lora.name].category) || 'unknown';
    if (category !== 'h3' && category !== 'unknown') {
      conflicts.push({ name: lora.name, reason: 'family', category });
    }
  }
  return conflicts;
}

module.exports = {
  classifyLora,
  compatibleCategoriesForContext,
  loraCompatibilityWarning,
  isH3TurboAdapterLora,
  h3TurboStackConflicts,
};
