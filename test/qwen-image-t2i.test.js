'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

// ---------------------------------------------------------------------------
// Path setup
// ---------------------------------------------------------------------------
const ROOT = path.join(__dirname, '..');
const serverSource = fs.readFileSync(path.join(ROOT, 'server.js'), 'utf8');

// ---------------------------------------------------------------------------
// Helpers: extract a named async function declaration from source code.
// ---------------------------------------------------------------------------
function extractFunction(source, name) {
  const regex = new RegExp(`async function ${name}\\(`, 'm');
  const match = source.match(regex);
  assert.ok(match, `Function "${name}" not found in server.js`);
  const startIndex = match.index;
  const braceIndex = source.indexOf('{', startIndex);
  assert.ok(braceIndex > 0, `Opening brace not found for "${name}"`);
  let depth = 0;
  for (let i = braceIndex; i < source.length; i++) {
    if (source[i] === '{') depth++;
    if (source[i] === '}') depth--;
    if (depth === 0) return source.substring(startIndex, i + 1);
  }
  assert.fail(`Unbalanced braces in function "${name}"`);
}

// ---------------------------------------------------------------------------
// Source code assertions — verify server.js contains the expected patterns.
// These follow the existing test conventions in this project.
// ---------------------------------------------------------------------------

test('server.js contains buildQwenT2I function declaration', () => {
  assert.match(serverSource, /async function buildQwenT2I\(p\)/);
});

test('buildQwenT2I uses QwenImageTextToImageApi node class', () => {
  assert.match(serverSource, /class_type:\s*'QwenImageTextToImageApi'/);
});

test('buildQwenT2I produces SaveImage node with KreaStudio/qwen prefix', () => {
  assert.match(serverSource, /filename_prefix:\s*'KreaStudio\/qwen'/);
  assert.match(serverSource, /images:\s*\[['"]api['"]\s*,\s*0\]/);
});

test('buildQwenT2I serializes model params as flat dot-keys', () => {
  assert.match(serverSource, /'model\.prompt'/);
  assert.match(serverSource, /'model\.negative_prompt'/);
  assert.match(serverSource, /'model\.width'/);
  assert.match(serverSource, /'model\.height'/);
});

test('buildQwenT2I sets prompt_extend to true by default', () => {
  assert.match(serverSource, /prompt_extend:\s*true/);
});

test('buildQwenT2I sets watermark to (Powered by Qwen-Image)', () => {
  assert.match(serverSource, /watermark:\s*'\(Powered by Qwen-Image\)'/);
});

test('buildQwenT2I falls back to qwen-image-3.0-pro when settings.qwenImageModel is falsy', () => {
  assert.match(serverSource, /settings\.qwenImageModel\s*\|\|\s*['"]qwen-image-3\.0-pro['"]/);
});

test('buildGenerationGraph dispatches qwen engine to buildQwenT2I', () => {
  assert.match(serverSource, /p\.engine\s*===\s*['"]qwen['"]\s*\?\s*buildQwenT2I\s*:\s*buildT2I/);
});

test('T2I_ENGINES is defined as ["qwen"]', () => {
  assert.match(serverSource, /const T2I_ENGINES\s*=\s*\[['"]qwen['"]\]/);
});

// ---------------------------------------------------------------------------
// Functional tests — extract functions into a sandboxed VM context with
// mock dependencies, then call them and assert on the returned graph.
// ---------------------------------------------------------------------------

// --- Minimal mock context for buildQwenT2I ---
// buildQwenT2I needs: settings (object), filterInputs (function).
// We provide a mock settings object and an identity filterInputs.
const qwenContext = vm.createContext({
  settings: { qwenImageModel: undefined },
  filterInputs: (graph) => graph,
});
vm.runInContext(extractFunction(serverSource, 'buildQwenT2I'), qwenContext);
const { buildQwenT2I } = qwenContext;

// --- Minimal mock context for buildGenerationGraph ---
// buildGenerationGraph needs: buildQwenT2I, buildT2I, hasActiveRegions,
// buildRegionalT2I, filterInputs, settings, T2I_ENGINES.
// We extract the real buildQwenT2I and buildGenerationGraph, and mock the rest.
const genContext = vm.createContext({
  settings: { qwenImageModel: undefined },
  T2I_ENGINES: ['qwen'],
  filterInputs: (graph) => graph,
  hasActiveRegions: () => false,
  buildRegionalT2I: async () => { throw new Error('buildRegionalT2I should not be called in these tests'); },
  buildT2I: async (p) => ({
    unet: { class_type: 'MockUNet', inputs: { ckpt_name: p.model } },
    clip: { class_type: 'CLIPLoader' },
    vae: { class_type: 'VAELoader' },
    save: { class_type: 'SaveImage', inputs: { images: ['decode', 0], filename_prefix: 'KreaStudio/gen' } },
  }),
});
// Inject the real buildQwenT2I and buildGenerationGraph into the generation context
vm.runInContext(extractFunction(serverSource, 'buildQwenT2I'), genContext);
vm.runInContext(extractFunction(serverSource, 'buildGenerationGraph'), genContext);
const { buildGenerationGraph } = genContext;

function spy(ctx, name, implementation) {
  const original = ctx[name];
  const spyFn = async (...args) => {
    spyFn.calledWith = args[0];
    return implementation(...args);
  };
  spyFn.calledWith = null;
  ctx[name] = spyFn;
  return spyFn;
}

// ---------------------------------------------------------------------------
// Tests: buildQwenT2I graph structure
// ---------------------------------------------------------------------------

test('buildQwenT2I produces graph with QwenImageTextToImageApi + SaveImage', async () => {
  const graph = await buildQwenT2I({
    prompt: 'a cat wearing sunglasses',
    width: 1024,
    height: 1024,
    seed: 42,
    steps: 20,
    cfg: 5,
  });

  assert.ok(graph.api, 'graph has api node');
  assert.ok(graph.save, 'graph has save node');
  assert.equal(graph.api.class_type, 'QwenImageTextToImageApi');
  assert.equal(graph.save.class_type, 'SaveImage');
});

test('buildQwenT2I SaveImage uses KreaStudio/qwen filename prefix', async () => {
  const graph = await buildQwenT2I({ prompt: 'test', width: 1024, height: 1024, seed: 1, steps: 10, cfg: 1 });
  assert.equal(graph.save.inputs.filename_prefix, 'KreaStudio/qwen');
  assert.equal(graph.save.inputs.images[0], 'api');
  assert.equal(graph.save.inputs.images[1], 0);
});

test('buildQwenT2I sets model to qwen-image-3.0-pro by default', async () => {
  const graph = await buildQwenT2I({ prompt: 'test', width: 1024, height: 1024, seed: 1, steps: 10, cfg: 1 });
  assert.equal(graph.api.inputs.model, 'qwen-image-3.0-pro');
});

test('buildQwenT2I uses custom model from settings.qwenImageModel', async () => {
  qwenContext.settings.qwenImageModel = 'qwen-image-3.0';
  const graph = await buildQwenT2I({ prompt: 'test', width: 1024, height: 1024, seed: 1, steps: 10, cfg: 1 });
  assert.equal(graph.api.inputs.model, 'qwen-image-3.0');
  // Restore
  qwenContext.settings.qwenImageModel = undefined;
});

test('buildQwenT2I falls back to qwen-image-3.0-pro when settings.qwenImageModel is undefined', async () => {
  qwenContext.settings.qwenImageModel = undefined;
  const graph = await buildQwenT2I({ prompt: 'test', width: 1024, height: 1024, seed: 1, steps: 10, cfg: 1 });
  assert.equal(graph.api.inputs.model, 'qwen-image-3.0-pro');
});

test('buildQwenT2I falls back to qwen-image-3.0-pro when settings.qwenImageModel is null', async () => {
  qwenContext.settings.qwenImageModel = null;
  const graph = await buildQwenT2I({ prompt: 'test', width: 1024, height: 1024, seed: 1, steps: 10, cfg: 1 });
  assert.equal(graph.api.inputs.model, 'qwen-image-3.0-pro');
});

// ---------------------------------------------------------------------------
// Tests: Flat DynamicCombo V3 dot-key serialization
// ---------------------------------------------------------------------------

test('buildQwenT2I serializes prompt as model.prompt dot-key', async () => {
  const graph = await buildQwenT2I({ prompt: 'a futuristic city', width: 1024, height: 1024, seed: 1, steps: 10, cfg: 1 });
  assert.equal(graph.api.inputs['model.prompt'], 'a futuristic city');
});

test('buildQwenT2I serializes negativePrompt as model.negative_prompt dot-key', async () => {
  const graph = await buildQwenT2I({
    prompt: 'a cat',
    negativePrompt: 'blurry, low quality',
    width: 1024,
    height: 1024,
    seed: 1,
    steps: 10,
    cfg: 1,
  });
  assert.equal(graph.api.inputs['model.negative_prompt'], 'blurry, low quality');
});

test('buildQwenT2I serializes width as model.width dot-key', async () => {
  const graph = await buildQwenT2I({ prompt: 'test', width: 832, height: 1216, seed: 1, steps: 10, cfg: 1 });
  assert.equal(graph.api.inputs['model.width'], 832);
});

test('buildQwenT2I serializes height as model.height dot-key', async () => {
  const graph = await buildQwenT2I({ prompt: 'test', width: 832, height: 1216, seed: 1, steps: 10, cfg: 1 });
  assert.equal(graph.api.inputs['model.height'], 1216);
});

test('buildQwenT2I includes all four dot-keys together', async () => {
  const graph = await buildQwenT2I({
    prompt: 'dot-key test',
    negativePrompt: 'ugly',
    width: 1024,
    height: 768,
    seed: 1,
    steps: 10,
    cfg: 1,
  });
  assert.equal(graph.api.inputs['model.prompt'], 'dot-key test');
  assert.equal(graph.api.inputs['model.negative_prompt'], 'ugly');
  assert.equal(graph.api.inputs['model.width'], 1024);
  assert.equal(graph.api.inputs['model.height'], 768);
});

test('buildQwenT2I also includes non-dot-key inputs: model, seed, steps, cfg', async () => {
  const graph = await buildQwenT2I({ prompt: 'test', width: 1024, height: 1024, seed: 42, steps: 20, cfg: 7 });
  assert.equal(graph.api.inputs.model, 'qwen-image-3.0-pro');
  assert.equal(graph.api.inputs.seed, 42);
  assert.equal(graph.api.inputs.steps, 20);
  assert.equal(graph.api.inputs.cfg, 7);
});

// ---------------------------------------------------------------------------
// Tests: prompt_extend defaults to true
// ---------------------------------------------------------------------------

test('buildQwenT2I prompt_extend defaults to true', async () => {
  const graph = await buildQwenT2I({ prompt: 'test', width: 1024, height: 1024, seed: 1, steps: 10, cfg: 1 });
  assert.equal(graph.api.inputs.prompt_extend, true);
});

// ---------------------------------------------------------------------------
// Tests: watermark defaults to (Powered by Qwen-Image)
// ---------------------------------------------------------------------------

test('buildQwenT2I watermark defaults to (Powered by Qwen-Image)', async () => {
  const graph = await buildQwenT2I({ prompt: 'test', width: 1024, height: 1024, seed: 1, steps: 10, cfg: 1 });
  assert.equal(graph.api.inputs.watermark, '(Powered by Qwen-Image)');
});

// ---------------------------------------------------------------------------
// Tests: engine dispatch — qwen routes to buildQwenT2I, not buildT2I
// ---------------------------------------------------------------------------

test('buildGenerationGraph routes qwen engine to buildQwenT2I', async () => {
  const qwenSpy = spy(genContext, 'buildQwenT2I', async (p) => ({
    api: { class_type: 'QwenImageTextToImageApi', inputs: { model: 'qwen-image-3.0-pro' } },
    save: { class_type: 'SaveImage' },
  }));
  const defaultSpy = spy(genContext, 'buildT2I', async (p) => ({
    unet: { class_type: 'MockUNet' },
    save: { class_type: 'SaveImage' },
  }));

  const graph = await buildGenerationGraph(
    { mode: 't2i', engine: 'qwen', prompt: 'test', seed: 1, steps: 10, cfg: 1, width: 1024, height: 1024 },
    []
  );

  assert.ok(qwenSpy.calledWith !== null, 'buildQwenT2I was called');
  assert.deepEqual(qwenSpy.calledWith, {
    mode: 't2i', engine: 'qwen', prompt: 'test', seed: 1, steps: 10, cfg: 1, width: 1024, height: 1024,
  });
  assert.equal(defaultSpy.calledWith, null, 'buildT2I was NOT called when engine is qwen');
  assert.equal(graph.api.class_type, 'QwenImageTextToImageApi');
});

test('buildGenerationGraph routes non-qwen engine to buildT2I', async () => {
  const qwenSpy = spy(genContext, 'buildQwenT2I', async (p) => ({
    api: { class_type: 'QwenImageTextToImageApi' },
    save: { class_type: 'SaveImage' },
  }));
  const defaultSpy = spy(genContext, 'buildT2I', async (p) => ({
    unet: { class_type: 'MockUNet' },
    save: { class_type: 'SaveImage' },
  }));

  const graph = await buildGenerationGraph(
    { mode: 't2i', engine: undefined, prompt: 'test', seed: 1, steps: 10, cfg: 1, width: 1024, height: 1024 },
    []
  );

  assert.equal(qwenSpy.calledWith, null, 'buildQwenT2I was NOT called when engine is not qwen');
  assert.ok(defaultSpy.calledWith !== null, 'buildT2I was called');
  assert.equal(graph.unet.class_type, 'MockUNet');
});

test('buildGenerationGraph dispatches even when engine is the string "qwen"', async () => {
  // Verify the dispatch works for the exact string 'qwen' (case-sensitive)
  const qwenSpy = spy(genContext, 'buildQwenT2I', async (p) => ({
    api: { class_type: 'QwenImageTextToImageApi' },
    save: { class_type: 'SaveImage' },
  }));
  const defaultSpy = spy(genContext, 'buildT2I', async (p) => ({
    unet: { class_type: 'MockUNet' },
    save: { class_type: 'SaveImage' },
  }));

  const graph = await buildGenerationGraph(
    { mode: 't2i', engine: 'qwen', prompt: 'dispatch test', seed: 99, steps: 5, cfg: 3, width: 512, height: 512 },
    []
  );

  assert.ok(qwenSpy.calledWith !== null, 'buildQwenT2I called for engine="qwen"');
  assert.equal(defaultSpy.calledWith, null, 'buildT2I NOT called');
  assert.equal(graph.api.class_type, 'QwenImageTextToImageApi');
});
