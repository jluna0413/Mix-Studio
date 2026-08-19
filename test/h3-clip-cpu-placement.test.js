'use strict';

// Regression: the H3 text encoder must be placed on CPU, not merely marked
// with an input CLIPLoader does not accept.
//
// MEASURED on Intel Arc Pro B70 32GB (2026-08-18), ComfyUI 0.33.1 + torch
// 2.12.1+xpu, MiniMax H3 t2v:
//
//   CLIPLoader device: 'default'  -> 256x256 x5 frames FAILED at 30.88 GB
//   CLIPLoader device: 'cpu'      -> 512x320 x56 frames OK at 25.15 GB
//
// Mechanism: comfy/sd.py:294 calls load_models_gpu(..., force_full_load=True)
// when the requested device equals the load device. 'default' resolves to the
// GPU and pins the whole ~14.6GB qwen3vl encoder in VRAM for the entire run,
// even though H3 uses it once for conditioning.
//
// The pre-existing offload code set `offload_device: 'cpu'` on CLIPLoader, but
// CLIPLoader exposes NO `offload_device` input (verified against live
// /object_info: required clip_name, type; optional device). That key was
// ignored, so the offload silently never happened.

const test = require('node:test');
const assert = require('node:assert');

const { getOffloadConfig, applyOffloadToGraph } = require('../lib/offload-config');

function h3Graph() {
  return {
    unet: { class_type: 'UNETLoader', inputs: { unet_name: 'minimax_h3_fl2va.safetensors', weight_dtype: 'default' } },
    clip: { class_type: 'CLIPLoader', inputs: { clip_name: 'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors', type: 'minimax', device: 'default' } },
    video_vae: { class_type: 'VAELoader', inputs: { vae_name: 'minimax_h3_video_vae_fp16.safetensors' } },
  };
}

test('Intel: H3 CLIPLoader is moved to CPU via `device`, not `offload_device`', () => {
  const cfg = getOffloadConfig('intel', 31.16);
  const g = applyOffloadToGraph(h3Graph(), cfg);

  assert.strictEqual(g.clip.inputs.device, 'cpu',
    'CLIPLoader.device must be "cpu" — "default" resolves to the GPU and forces ' +
    'a full ~14.6GB load (measured: even 256x256 x5 OOMs at 30.88GB)');

  assert.ok(!('offload_device' in g.clip.inputs),
    'CLIPLoader has no offload_device input; setting it is silently ignored');
});

test('Intel: the DiT still loads on the GPU (only the encoder moves)', () => {
  const cfg = getOffloadConfig('intel', 31.16);
  const g = applyOffloadToGraph(h3Graph(), cfg);

  assert.notStrictEqual(g.unet.inputs.device, 'cpu',
    'moving the DiT to CPU would make sampling unusably slow');
});

test('NVIDIA high-VRAM: encoder stays on the GPU (no needless transfer)', () => {
  const cfg = getOffloadConfig('nvidia', 80);
  const g = applyOffloadToGraph(h3Graph(), cfg);

  assert.notStrictEqual(g.clip.inputs.device, 'cpu',
    'offload is disabled on high-VRAM NVIDIA; the encoder should stay resident');
});

test('NVIDIA <=48GB H3 R2V: encoder moves to CPU as well', () => {
  const cfg = getOffloadConfig('nvidia', 24, { h3R2V: true });
  const g = applyOffloadToGraph(h3Graph(), cfg);

  assert.strictEqual(g.clip.inputs.device, 'cpu',
    'H3 R2V loads DiT + 32B encoder + 2 VAEs; encoder must not stay pinned');
});

test('an explicit non-default device is not overridden', () => {
  const g0 = h3Graph();
  g0.clip.inputs.device = 'cuda:1';
  const g = applyOffloadToGraph(g0, getOffloadConfig('intel', 31.16));

  assert.strictEqual(g.clip.inputs.device, 'cuda:1',
    'an explicitly chosen device must be respected');
});
