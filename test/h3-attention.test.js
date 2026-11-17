'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');

const {
  H3_ATTENTION_BACKENDS,
  h3AttentionOptions,
  h3AttentionRuntimeGate,
  normalizeH3AttentionBackend,
} = require('../lib/h3-attention');

test('H3 attention backend normalization supports SLA/Triton without changing legacy behavior', () => {
  // 'triton' is the Intel Arc/XPU path; the CUDA-only backends stay available elsewhere.
  assert.deepEqual(H3_ATTENTION_BACKENDS, ['standard', 'sageattention', 'sla', 'kitchen', 'triton']);
  assert.equal(normalizeH3AttentionBackend('sla'), 'sla');
  assert.equal(normalizeH3AttentionBackend('sage'), 'sageattention');
  assert.equal(normalizeH3AttentionBackend('', false), 'standard');
  assert.equal(normalizeH3AttentionBackend('', true), 'sageattention');
  assert.equal(normalizeH3AttentionBackend('triton'), 'triton');
  assert.deepEqual(h3AttentionOptions('sla'), {
    attentionBackend: 'sla', sageAttention: false, slaAttention: true,
    kitchenAttention: false, tritonAttention: false,
  });
});

test('Triton capability reflects the installed ModelAttentionBackend options', () => {
  const { tritonAttentionCapability } = require('../lib/h3-attention');
  assert.equal(tritonAttentionCapability({}).ready, false);
  assert.equal(tritonAttentionCapability({ ModelAttentionBackend: { input: { required: { attention: ['COMBO', { options: ['triton'] }] } } } }).ready, true);
});

test('Kitchen requires the advertised native backend and never stacks attention patches', async () => {
  const { kitchenAttentionCapability } = require('../lib/h3-attention');
  const { buildMiniMaxH3Graph } = require('../lib/video-workflows');
  assert.equal(kitchenAttentionCapability({}).ready, false);
  assert.equal(kitchenAttentionCapability({ ModelAttentionBackend: { input: { required: { attention: ['COMBO', { options: ['pytorch attention'] }] } } } }).ready, false);
  assert.equal(kitchenAttentionCapability({ ModelAttentionBackend: { input: { required: { attention: ['COMBO', { options: ['comfy kitchen attention'] }] } } } }).ready, true);
  const graph = await buildMiniMaxH3Graph({ attentionBackend: 'kitchen', sageAttention: true }, {});
  assert.deepEqual(graph.guider.inputs.model, ['kitchen_attention', 0]);
  assert.equal(graph.kitchen_attention.inputs.attention, 'comfy kitchen attention');
  assert.equal(graph.sage_attention, undefined);
  assert.equal(graph.sla_attention, undefined);
});

test('H3 attention runtime gate only hard-blocks on missing node or inspected not-ready probe', () => {
  assert.deepEqual(h3AttentionRuntimeGate({ ready: false, inspected: false }, true), { blocked: false });
  assert.equal(h3AttentionRuntimeGate({ ready: false, inspected: true, reason: 'no cuda' }, true).blocked, true);
  assert.equal(h3AttentionRuntimeGate({ ready: false, inspected: true, reason: 'no cuda' }, true).code, 'h3_attention_runtime_unavailable');
  assert.equal(h3AttentionRuntimeGate({ ready: true, inspected: true }, false).blocked, true);
  assert.equal(h3AttentionRuntimeGate({ ready: true, inspected: true }, false).code, 'h3_attention_node_missing');
  assert.deepEqual(h3AttentionRuntimeGate({ ready: true, inspected: true }, true), { blocked: false });
});
