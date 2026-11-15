'use strict';

// Backend availability depends on the GPU. On Intel Arc (XPU) the CUDA-only
// backends (sageAttention, SageSLA, Comfy Kitchen) cannot run; standard
// (PyTorch SDPA on XPU) and Triton are the working options there.
const H3_ATTENTION_BACKENDS = Object.freeze([
  'standard', 'sageattention', 'sla', 'kitchen', 'triton',
]);

function normalizeH3AttentionBackend(value, legacySageAttention) {
  const requested = String(value || '').trim().toLowerCase();
  if (requested === 'sage') return 'sageattention';
  if (H3_ATTENTION_BACKENDS.includes(requested)) return requested;
  if (legacySageAttention === false) return 'standard';
  return 'sageattention';
}

function attentionOptions(info) {
  const input = info?.ModelAttentionBackend?.input?.required?.attention;
  const options = Array.isArray(input?.[0]) ? input[0] : input?.[1]?.options;
  return Array.isArray(options) ? options : [];
}

function standardAttentionCapability(info) {
  const options = attentionOptions(info);
  const ready = options.includes('pytorch attention') || options.includes('standard');
  return {
    ready,
    reason: ready ? '' : 'The ComfyUI ModelAttentionBackend has no standard/pytorch attention option.',
  };
}

function tritonAttentionCapability(info) {
  const options = attentionOptions(info);
  const ready = options.includes('triton');
  return {
    ready,
    reason: ready ? '' : 'A Triton attention backend is not exposed by the installed ComfyUI.',
  };
}

function kitchenAttentionCapability(info) {
  const options = attentionOptions(info);
  const ready = options.includes('comfy kitchen attention');
  return {
    ready,
    reason: ready ? '' : 'Update ComfyUI to a version with the native Comfy Kitchen attention backend.',
  };
}

function h3AttentionOptions(value, legacySageAttention) {
  const attentionBackend = normalizeH3AttentionBackend(value, legacySageAttention);
  return {
    attentionBackend,
    sageAttention: attentionBackend === 'sageattention',
    slaAttention: attentionBackend === 'sla',
    kitchenAttention: attentionBackend === 'kitchen',
    tritonAttention: attentionBackend === 'triton',
  };
}

module.exports = {
  H3_ATTENTION_BACKENDS,
  standardAttentionCapability,
  tritonAttentionCapability,
  kitchenAttentionCapability,
  h3AttentionOptions,
  normalizeH3AttentionBackend,
};
