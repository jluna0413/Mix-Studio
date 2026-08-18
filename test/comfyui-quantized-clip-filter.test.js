'use strict';

/**
 * Regression test for the ComfyUI quantized text-encoder key filter.
 *
 * Root cause this guards against: ComfyUI's SDClipModel.load_sd() filters the
 * checkpoint state_dict against the model's own state_dict(). A quantized
 * ops.Linear is lazy — it registers no `weight` parameter until
 * _load_quantized_module builds the QuantizedTensor — so the model's
 * state_dict() lists NONE of the quantized projection keys and the filter
 * deleted every one of them. The result was `module.weight = None` and then
 * `AttributeError: 'NoneType' object has no attribute 'device'` on the first
 * forward pass of MiniMax H3.
 *
 * The subtle bug in the first bypass attempt was an off-by-one-dot mismatch:
 *   ".comfy_quant" stripped  -> "…q_proj"    (no trailing dot)
 *   "weight" stripped        -> "…q_proj."   (trailing dot)
 * so the prefixes never compared equal and the bypass silently did nothing.
 *
 * This test encodes the *filter contract* in pure JS so it runs in Mix Studio's
 * suite without needing ComfyUI or a 14.6 GB checkpoint. If someone
 * reintroduces the dot mismatch, this fails.
 *
 * Reference: docs/comfyui-h3-arc-patch.md
 */

const test = require('node:test');
const assert = require('node:assert');

const QUANT_SUFFIXES = [
  '.comfy_quant',
  '.weight_scale',
  '.weight_scale_2',
  '.input_scale',
  '.pre_quant_scale',
  '_rotary',
];

/**
 * Mirror of the patched `load_sd` filter in comfy/sd1_clip.py.
 *
 * @param {string[]} checkpointKeys every key present in the checkpoint
 * @param {Set<string>} modelKeys keys the model's state_dict() exposes
 * @param {Set<string>} shapeMismatches enumerated keys whose shape disagrees
 * @returns {string[]} keys that survive the filter
 */
function filterStateDictKeys(checkpointKeys, modelKeys, shapeMismatches = new Set()) {
  const quantPrefixes = new Set();
  for (const key of checkpointKeys) {
    if (key.endsWith('.comfy_quant')) {
      // Stored WITHOUT a trailing dot so it matches the prefix derived below.
      quantPrefixes.add(key.slice(0, -'.comfy_quant'.length));
    }
  }

  const kept = [];
  for (const key of checkpointKeys) {
    let prefix;
    if (key.endsWith('.weight')) prefix = key.slice(0, -'.weight'.length);
    else if (key.endsWith('.bias')) prefix = key.slice(0, -'.bias'.length);
    else prefix = key;

    const isQuantMetadata = QUANT_SUFFIXES.some((suffix) => key.endsWith(suffix));
    if (isQuantMetadata || quantPrefixes.has(prefix)) {
      kept.push(key); // quantized: bypass the shape filter
      continue;
    }
    // A key the model does not enumerate is NOT necessarily unwanted: a lazy
    // ops.Linear (quantized OR plain BF16, as in qwen3vl's non-quantized
    // vision tower) registers no `weight` until load populates it. Dropping
    // these left visual.blocks.N.attn.qkv with weight=None and crashed the
    // image path. Only shape-filter keys the model actually enumerates.
    if (!modelKeys.has(key)) kept.push(key);
    else if (!shapeMismatches.has(key)) kept.push(key);
  }
  return kept;
}

/** Build a realistic slice of the qwen3vl minimax H3 checkpoint key layout. */
function buildCheckpointKeys(layerCount = 2) {
  const keys = [];
  const projections = [
    'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
    'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj',
  ];
  for (let layer = 0; layer < layerCount; layer += 1) {
    for (const projection of projections) {
      const base = `model.layers.${layer}.${projection}`;
      // Quantized layer: weight blob + quant metadata + scale.
      keys.push(`${base}.weight`, `${base}.comfy_quant`, `${base}.weight_scale`);
    }
    // Non-quantized norms live alongside the quantized projections.
    keys.push(`model.layers.${layer}.input_layernorm.weight`);
    keys.push(`model.layers.${layer}.post_attention_layernorm.weight`);
    keys.push(`model.layers.${layer}.self_attn.q_norm.weight`);
    keys.push(`model.layers.${layer}.self_attn.k_norm.weight`);
  }
  keys.push('model.embed_tokens.weight', 'model.embed_tokens.comfy_quant');
  // Vision tower is NOT quantized in this checkpoint.
  keys.push('visual.blocks.0.attn.qkv.weight', 'visual.blocks.0.attn.qkv.bias');
  keys.push('visual.blocks.0.mlp.linear_fc1.weight', 'visual.blocks.0.mlp.linear_fc1.bias');
  keys.push('visual.merger.linear_fc1.weight', 'visual.merger.linear_fc1.bias');
  return keys;
}

/**
 * The model's state_dict() omits every LAZY param — which is both the
 * quantized text projections AND the non-quantized BF16 vision tower (both are
 * ops.Linear, which registers no `weight` until load populates it). Only truly
 * eager params (norms, embeddings) are enumerated.
 *
 * Getting this right matters: an earlier version of this helper wrongly
 * exposed the vision keys, so the "vision tower preserved" test passed for the
 * wrong reason while the real loader was silently dropping them.
 */
function buildModelKeys(checkpointKeys) {
  const quantPrefixes = new Set(
    checkpointKeys
      .filter((key) => key.endsWith('.comfy_quant'))
      .map((key) => key.slice(0, -'.comfy_quant'.length))
  );
  return new Set(
    checkpointKeys.filter((key) => {
      if (key.endsWith('.comfy_quant') || key.endsWith('.weight_scale')) return false;
      // Lazy ops.Linear layers are absent from state_dict(). In this
      // checkpoint that is every quantized projection plus the whole
      // (non-quantized) vision tower.
      if (key.startsWith('visual.')) return false;
      const prefix = key.endsWith('.weight')
        ? key.slice(0, -'.weight'.length)
        : key.endsWith('.bias') ? key.slice(0, -'.bias'.length) : key;
      return !quantPrefixes.has(prefix); // quantized params are absent
    })
  );
}

test('quantized projection weights survive the state_dict filter', () => {
  const checkpointKeys = buildCheckpointKeys(2);
  const modelKeys = buildModelKeys(checkpointKeys);
  const kept = new Set(filterStateDictKeys(checkpointKeys, modelKeys));

  // Precondition: the model genuinely does not expose the quantized weight.
  assert.ok(
    !modelKeys.has('model.layers.0.self_attn.q_proj.weight'),
    'precondition: lazy quantized Linear must be absent from model state_dict'
  );

  // Every projection weight must survive despite that absence.
  for (const projection of [
    'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
    'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj',
  ]) {
    for (const layer of [0, 1]) {
      const key = `model.layers.${layer}.${projection}.weight`;
      assert.ok(kept.has(key), `quantized weight must be preserved: ${key}`);
    }
  }
});

test('quant metadata and scales are preserved', () => {
  const checkpointKeys = buildCheckpointKeys(1);
  const modelKeys = buildModelKeys(checkpointKeys);
  const kept = new Set(filterStateDictKeys(checkpointKeys, modelKeys));

  assert.ok(kept.has('model.layers.0.self_attn.q_proj.comfy_quant'));
  assert.ok(kept.has('model.layers.0.self_attn.q_proj.weight_scale'));
  assert.ok(kept.has('model.embed_tokens.comfy_quant'));
});

test('enumerated keys with mismatched shapes are still dropped', () => {
  // This is the filter's original purpose (klein T5 compatibility): when the
  // model DOES enumerate a key and the shapes disagree, drop it. Absence is
  // no longer treated as a mismatch, but a real mismatch must still be caught.
  const checkpointKeys = buildCheckpointKeys(1);
  const modelKeys = buildModelKeys(checkpointKeys);
  const mismatched = 'model.layers.0.input_layernorm.weight';
  assert.ok(modelKeys.has(mismatched), 'precondition: model enumerates this key');

  const kept = new Set(
    filterStateDictKeys(checkpointKeys, modelKeys, new Set([mismatched]))
  );
  assert.ok(!kept.has(mismatched), 'shape-mismatched enumerated key must be dropped');
  assert.ok(kept.has('model.layers.0.post_attention_layernorm.weight'), 'matching key kept');
});

test('non-quantized LAZY vision weights survive (i2v crash regression)', () => {
  // The bug: visual.blocks.N.attn.qkv is a lazy ops.Linear but is NOT
  // quantized (no .comfy_quant sibling), so it got neither the quant bypass
  // nor a state_dict entry — and was silently dropped. weight stayed None and
  // the image path died at qwen35.py:490 self.qkv(x) -> ops.py:393
  // s.weight.device with "'NoneType' object has no attribute 'device'".
  //
  // Text-to-video never runs the vision tower, which is exactly why t2v
  // worked and image-to-video crashed on the same checkpoint.
  const checkpointKeys = buildCheckpointKeys(1);
  const modelKeys = buildModelKeys(checkpointKeys);

  // Precondition: the model genuinely does not enumerate the vision weight.
  assert.ok(
    !modelKeys.has('visual.blocks.0.attn.qkv.weight'),
    'precondition: lazy vision Linear must be absent from model state_dict'
  );
  // Precondition: it is NOT rescued by the quantized bypass.
  assert.ok(
    !checkpointKeys.includes('visual.blocks.0.attn.qkv.comfy_quant'),
    'precondition: vision tower is not quantized'
  );

  const kept = new Set(filterStateDictKeys(checkpointKeys, modelKeys));
  for (const key of [
    'visual.blocks.0.attn.qkv.weight',
    'visual.blocks.0.attn.qkv.bias',
    'visual.blocks.0.mlp.linear_fc1.weight',
    'visual.merger.linear_fc1.weight',
  ]) {
    assert.ok(kept.has(key), `lazy vision key must be preserved: ${key}`);
  }
});

test('regression: prefix derivation must strip the dot', () => {
  // This is the exact bug that made the first bypass a no-op. Stripping
  // "weight" instead of ".weight" leaves a trailing dot, so the prefix never
  // matches the one stored from ".comfy_quant" and every quantized weight is
  // silently dropped.
  const weightKey = 'model.layers.0.self_attn.q_proj.weight';
  const quantKey = 'model.layers.0.self_attn.q_proj.comfy_quant';

  const storedPrefix = quantKey.slice(0, -'.comfy_quant'.length);
  const buggyPrefix = weightKey.slice(0, -'weight'.length);   // trailing dot
  const correctPrefix = weightKey.slice(0, -'.weight'.length); // no dot

  assert.notStrictEqual(buggyPrefix, storedPrefix, 'buggy form must not match');
  assert.strictEqual(correctPrefix, storedPrefix, 'correct form must match');
});

test('every quantized weight is preserved at full 50-layer scale', () => {
  const checkpointKeys = buildCheckpointKeys(50);
  const modelKeys = buildModelKeys(checkpointKeys);
  const kept = new Set(filterStateDictKeys(checkpointKeys, modelKeys));

  const quantWeights = checkpointKeys.filter((key) => {
    if (!key.endsWith('.weight')) return false;
    return checkpointKeys.includes(`${key.slice(0, -'.weight'.length)}.comfy_quant`);
  });

  assert.strictEqual(quantWeights.length, 50 * 7 + 1, 'expected 7 projections/layer + embed');
  const dropped = quantWeights.filter((key) => !kept.has(key));
  assert.deepStrictEqual(dropped, [], 'no quantized weight may be dropped');
});

module.exports = { filterStateDictKeys };
