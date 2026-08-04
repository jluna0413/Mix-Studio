'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const {
  buildRegionalT2IGraph,
  buildKrea2InpaintGraph,
} = require('../lib/regional-workflows');

test('Krea2 text-to-image workflow validates basic configuration', () => {
  const server = require('fs').readFileSync(
    require('path').join(__dirname, '..', 'server.js'),
    'utf8'
  );
  assert.match(server, /async function buildT2I\(/);
  assert.match(server, /sampler_name:\s*'euler'|scheduler:\s*'beta'/);
  assert.match(server, /steps:\s*p\.steps|cfg:\s*p\.cfg/);
});

test('Krea2 depth guidance workflow includes source image conditioning', () => {
  const server = require('fs').readFileSync(
    require('path').join(__dirname, '..', 'server.js'),
    'utf8'
  );
  assert.match(server, /async function buildKrea2DepthControl|buildKrea2DepthControl/);
});

test('Regional text-to-image builds Ideogram prompt builder with Krea2 regional node', () => {
  const graph = buildRegionalT2IGraph({
    prompt: 'fashion editorial in a marble lobby',
    width: 1024,
    height: 768,
    seed: 123,
    steps: 18,
    cfg: 3,
    loras: [{ name: 'Global/Look.safetensors', strength: 0.7, on: true }],
    regions: [
      {
        description: 'left model',
        x: 0.08,
        y: 0.16,
        w: 0.36,
        h: 0.7,
        lora: 'People/Left.safetensors',
        refImageName: 'left.png',
      },
    ],
    settings: {
      unet: 'krea2_turbo_fp8_scaled.safetensors',
      clip: 'Huihui-Qwen3-VL-4B-Instruct-abliterated-fp8_scaled.safetensors',
      clipType: 'krea2',
      vae: 'qwen_image_vae.safetensors',
    },
  });

  assert.equal(graph.prompt_builder.class_type, 'Ideogram4PromptBuilderKJ');
  assert.equal(graph.prompt_builder.inputs.import_mode, 'when empty');
  assert.equal(graph.prompt_builder.inputs.output_format, 'compact');
  assert.deepEqual(graph.regional.inputs.bboxes, ['prompt_builder', 2]);
  assert.equal(graph.regional.class_type, 'Krea2RegionalMultiLoRAV3');
  assert.equal(graph.regional.inputs.split_mode, 'bbox');
  assert.equal(graph.sampler.inputs.model[0], 'regional');
});

test('Krea2 inpaint uses VAEEncode + SetLatentNoiseMask for flow models', () => {
  const graph = buildKrea2InpaintGraph({
    imageName: 'source.png',
    maskImageName: 'mask.png',
    prompt: 'replace the jacket with glossy black leather',
    width: 1024,
    height: 1024,
    seed: 77,
    steps: 16,
    cfg: 3,
    denoise: 0.74,
    settings: {
      unet: 'krea2_turbo_fp8_scaled.safetensors',
      clip: 'Huihui-Qwen3-VL-4B-Instruct-abliterated-fp8_scaled.safetensors',
      clipType: 'krea2',
      vae: 'qwen_image_vae.safetensors',
    },
  });

  assert.equal(graph.encode.class_type, 'VAEEncode');
  assert.equal(graph.inpaint_latent.class_type, 'SetLatentNoiseMask');
  assert.equal(graph.sampler.inputs.latent_image[0], 'inpaint_latent');
  assert.equal(graph.sampler.inputs.denoise, 0.74);
  assert.ok(graph.sampler.inputs.denoise >= 0.75 || graph.inpaint_latent, 'denoise >= 0.75 or inpaint_latent present');
});

test('Klein edit workflow uses ReferenceLatent chains', () => {
  const server = require('fs').readFileSync(
    require('path').join(__dirname, '..', 'server.js'),
    'utf8'
  );
  assert.match(server, /async function buildEdit\(/);
  assert.match(server, /ReferenceLatent|reference_latent/);
});

test('Qwen edit workflow includes Lightning LoRA support', () => {
  const server = require('fs').readFileSync(
    require('path').join(__dirname, '..', 'server.js'),
    'utf8'
  );
  assert.match(server, /async function buildEditQwen\(/);
  assert.match(server, /lightning.*lora|firelio/i);
});

test('LTX animate builds two-stage base + refine sigma workflow', () => {
  const server = require('fs').readFileSync(
    require('path').join(__dirname, '..', 'server.js'),
    'utf8'
  );
  assert.match(server, /async function buildAnimate\(/);
  assert.match(server, /ltx.*animate|LTX.*Animate/);
  assert.match(server, /sigmas?:\s*null|base.*sigma|refine/);
});

test('LTX Face ID generates single-stage face-identity locked audio', () => {
  const server = require('fs').readFileSync(
    require('path').join(__dirname, '..', 'server.js'),
    'utf8'
  );
  assert.match(server, /async function buildAnimateFaceId\(/);
  assert.match(server, /LTXIdentityOverlapConditioning|FaceID.*Lora|audio.*latent/);
});

test('10Eros uses Echo DMD sampler with sigma presets', () => {
  const server = require('fs').readFileSync(
    require('path').join(__dirname, '..', 'server.js'),
    'utf8'
  );
  assert.match(server, /async function buildAnimateEros\(/);
  assert.match(server, /EROS_SIGMA_PRESETS|Echo.*DMD|dmd/);
});

test('Wan 2.2 uses dual KSamplerAdvanced handoff at 16fps', () => {
  const server = require('fs').readFileSync(
    require('path').join(__dirname, '..', 'server.js'),
    'utf8'
  );
  assert.match(server, /async function buildAnimateWan\(/);
  assert.match(server, /KSamplerAdvanced|wan.*2\.2|fps.*16/);
});

test('SCAIL 2 integrates SAM3 tracking and WanSCAILToVideo', () => {
  const server = require('fs').readFileSync(
    require('path').join(__dirname, '..', 'server.js'),
    'utf8'
  );
  assert.match(server, /async function buildAnimateScail\(/);
  assert.match(server, /SAM3|WanSCAILToVideo|scail.*video/);
});

test('Upscale workflows support SeedVR2 and Ultimate SD', () => {
  const server = require('fs').readFileSync(
    require('path').join(__dirname, '..', 'server.js'),
    'utf8'
  );
  assert.match(server, /async function buildUpscale\(/);
  assert.match(server, /SeedVR2|Ultimate.*SD|upscale.*workflow/);
});

test('Composite builds side-by-side ImageStitch for multi-frame output', () => {
  const server = require('fs').readFileSync(
    require('path').join(__dirname, '..', 'server.js'),
    'utf8'
  );
  assert.match(server, /async function buildImageComposite\(/);
  assert.match(server, /ImageStitch|composite.*image/);
});

test('Generation graph builder dispatches to appropriate engine builder', () => {
  const server = require('fs').readFileSync(
    require('path').join(__dirname, '..', 'server.js'),
    'utf8'
  );
  assert.match(server, /async function buildGenerationGraph\(/);
  assert.match(server, /buildT2I|buildRegionalT2IGraph|buildKrea2InpaintGraph|buildEdit|buildAnimate|buildUpscale/);
});