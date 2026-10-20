'use strict';

// Intel Arc B70 (XPU): sageAttention / SageSLA are CUDA-only and cannot run.
// Standard (PyTorch SDPA on XPU) is the working default; 'triton' is an
// optional backend for XPU Triton attention kernels.
const H3_ATTENTION_BACKENDS = Object.freeze(['standard', 'triton']);

function normalizeH3AttentionBackend(value) {
  const requested = String(value || '').trim().toLowerCase();
  if (H3_ATTENTION_BACKENDS.includes(requested)) return requested;
  return 'standard';
}

function standardAttentionCapability(info) {
  const input = info?.ModelAttentionBackend?.input?.required?.attention;
  const options = Array.isArray(input?.[0]) ? input[0] : input?.[1]?.options;
  const ready = Array.isArray(options) && (options.includes('pytorch attention') || options.includes('standard'));
  return { ready, reason: ready ? '' : 'The ComfyUI ModelAttentionBackend has no standard/pytorch attention option.' };
}

function tritonAttentionCapability(info) {
  const input = info?.ModelAttentionBackend?.input?.required?.attention;
  const options = Array.isArray(input?.[0]) ? input[0] : input?.[1]?.options;
  const ready = Array.isArray(options) && options.includes('triton');
  return { ready, reason: ready ? '' : 'A Triton attention backend is not exposed by the installed ComfyUI.' };
}

function h3AttentionOptions(value) {
  const attentionBackend = normalizeH3AttentionBackend(value);
  return {
    attentionBackend,
    sageAttention: false,
    slaAttention: false,
    tritonAttention: attentionBackend === 'triton',
  };
}

module.exports = {
  H3_ATTENTION_BACKENDS,
  standardAttentionCapability,
  tritonAttentionCapability,
  h3AttentionOptions,
  normalizeH3AttentionBackend,
};
