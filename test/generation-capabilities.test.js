'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const {
  configuredVideoEngineCapability,
  dependencyComponentBlock,
  ltxRefinePreflight,
  ltxVramEstimate,
  videoEngineCapabilities,
} = require('../lib/generation-capabilities');

const apple = { gpuVendor: 'apple', gpuBackend: 'mps', vramGb: 64 };
const nvidia = { gpuVendor: 'nvidia', gpuBackend: 'cuda', vramGb: 24 };
const amd = { gpuVendor: 'amd', gpuBackend: 'rocm', vramGb: 24 };
const b70 = { gpuVendor: 'intel', gpuBackend: 'xpu', vramGb: 32, vramFreeGb: 18 };

test('Apple Metal exposes BF16 LTX and disables curated FP8 video families', () => {
  const capabilities = videoEngineCapabilities(apple);
  assert.equal(capabilities.ltx.supported, true);
  assert.equal(capabilities.ltx.requiresBf16, true);
  assert.equal(capabilities['ltx-edit'].supported, true);
  assert.equal(capabilities.ltx25.supported, false);
  assert.match(capabilities.ltx25.reason, /INT8 ConvRot|Apple Metal/);
  assert.equal(capabilities.h3.supported, false);
  for (const engine of ['eros', 'wan', 'wan-animate2', 'scail']) {
    assert.equal(capabilities[engine].supported, false);
    assert.match(capabilities[engine].reason, /FP8|Apple Metal/);
  }
  assert.match(dependencyComponentBlock('scailinfinity', apple), /Apple Metal/);
  assert.equal(dependencyComponentBlock('wan', nvidia), '');
  assert.equal(dependencyComponentBlock('wananimate2', nvidia), '');
  assert.match(dependencyComponentBlock('wananimate2', apple), /INT8 ConvRot|Apple Metal/);
  assert.match(dependencyComponentBlock('wananimate2', amd), /INT8 ConvRot|AMD ROCm/);
  assert.match(dependencyComponentBlock('h3r2v', apple), /NVFP4|Apple Metal/);
  assert.match(dependencyComponentBlock('h3turbo', apple), /NVFP4|Apple Metal/);
  assert.match(dependencyComponentBlock('h3', amd), /AMD ROCm/);
  assert.match(dependencyComponentBlock('ltx25', amd), /INT8 ConvRot|AMD ROCm/);
  assert.equal(videoEngineCapabilities(nvidia).ltx25.supported, true);
  assert.equal(videoEngineCapabilities(nvidia).h3.supported, true);
});

test('Apple LTX generation rejects FP8 configuration and accepts BF16', () => {
  assert.equal(configuredVideoEngineCapability('ltx', apple, {
    ltxCkpt: 'ltx-2.3-22b-dev-fp8.safetensors',
  }).supported, false);
  assert.equal(configuredVideoEngineCapability('ltx', apple, {
    ltxCkpt: 'ltx-2.3-22b-dev.safetensors',
  }).supported, true);
});

test('LTX refine preflight catches the reported large Apple request before queueing', () => {
  const unsafe = ltxRefinePreflight({ width: 1600, height: 896, frames: 273, fps: 25, profile: apple });
  assert.equal(unsafe.ok, false);
  assert.ok(unsafe.maxSeconds < 11);
  assert.match(unsafe.error, /too large/);

  assert.equal(ltxRefinePreflight({ width: 1280, height: 1280, frames: 201, fps: 25, profile: apple }).ok, true);
  assert.equal(ltxRefinePreflight({ width: 1600, height: 896, frames: 273, fps: 25, profile: nvidia }).ok, true);
});

test('ltxVramEstimate returns higher stage2 latent for same resolution but stage1 includes text encoder', () => {
  const s1 = ltxVramEstimate(1280, 720, 121, 1);
  const s2 = ltxVramEstimate(1280, 720, 121, 2);
  assert.ok(s2 > ltxVramEstimate(1280, 720, 121, 1) * 0.5, 'stage2 latent is 4x stage1 latent');
  assert.ok(ltxVramEstimate(1920, 1080, 121, 2) > ltxVramEstimate(1280, 720, 121, 2), 'higher res needs more VRAM');
  assert.ok(ltxVramEstimate(1280, 720, 241, 2) > ltxVramEstimate(1280, 720, 121, 2), 'more frames need more VRAM');
});

test('ltxRefinePreflight rejects a 4K 20s refine on a simulated 16GB GPU', () => {
  const result = ltxRefinePreflight({ width: 1920, height: 1080, frames: 501, fps: 25, profile: { gpuVendor: 'nvidia', vramFreeGb: 16 } });
  assert.equal(result.ok, false);
  assert.match(result.error, /exceeds available VRAM/);
});

test('ltxRefinePreflight accepts a 1280x720 10s refine on a simulated 32GB GPU', () => {
  const result = ltxRefinePreflight({ width: 1280, height: 720, frames: 251, fps: 25, profile: { gpuVendor: 'nvidia', vramFreeGb: 32 } });
  assert.equal(result.ok, true);
});

test('ltxRefinePreflight flags needsLowVram when estimate is above 85% threshold', () => {
  const result = ltxRefinePreflight({ width: 1280, height: 720, frames: 121, fps: 25, profile: { gpuVendor: 'intel', vramFreeGb: 28 } });
  assert.equal(result.ok, true);
  assert.equal(result.needsLowVram, true);
  assert.ok(result.requiredGb > result.vramFreeGb * 0.85);
  assert.ok(result.requiredGb <= result.vramFreeGb);
});

test('ltxRefinePreflight skips VRAM gate when vramFreeGb is missing', () => {
  const result = ltxRefinePreflight({ width: 1920, height: 1080, frames: 501, fps: 25, profile: { gpuVendor: 'nvidia' } });
  assert.equal(result.ok, true);
  assert.equal(result.needsLowVram, undefined);
});

test('video UI consumes server capability gating', () => {
  const root = path.join(__dirname, '..');
  const app = fs.readFileSync(path.join(root, 'public', 'app.js'), 'utf8');
  const server = fs.readFileSync(path.join(root, 'server.js'), 'utf8');
  assert.match(app, /state\.videoCapabilities = lastMeta\.capabilities\?\.video \|\| \{\}/);
  assert.match(app, /function supportedVideoEngines\(\)/);
  assert.match(app, /button\.disabled = capability\.supported === false/);
  assert.match(app, /if \(capability\.supported === false\) \{[\s\S]{0,180}toast\(capability\.reason/);
  assert.match(server, /function adoptDeviceCompatibleModelSettings/);
  assert.match(server, /settings\.ltxCkpt = 'ltx-2\.3-22b-dev\.safetensors'/);
});

test('Intel Arc supports LTX 2.5 with INT8 ConvRot text encoder CPU offload', () => {
  // LTX 2.5 uses Gemma 4 12B INT8 ConvRot text encoder (15.4 GB) loaded via
  // ComfyUI's native CLIPLoader (type="ltxv"). Verified empirically:
  // detect_layer_quantization() detects .comfy_quant metadata → mixed_precision_ops()
  // creates proper quantized Linear layers on XPU. The applyOffloadToGraph()
  // config for Intel Arc sets device:"cpu" for CLIPLoader, so the TE loads on
  // CPU and is offloaded after embedding generation. Total VRAM: DiT (21.5 GB)
  // + VAE (2 GB) + working memory < 32 GB.
  const arcB70 = { gpuVendor: 'intel', gpuBackend: 'xpu', vramGb: 32 };
  const caps = videoEngineCapabilities(arcB70);
  assert.equal(caps.ltx25.supported, true);
  assert.equal(caps['ltx25quality'].supported, false);
  assert.match(caps['ltx25quality'].reason, /vRAM|BF16|36 GB/);
  assert.equal(caps.h3.supported, true);
  assert.equal(dependencyComponentBlock('ltx25', arcB70), '');
  assert.equal(dependencyComponentBlock('h3', arcB70), '');
});
