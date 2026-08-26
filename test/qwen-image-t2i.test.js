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
const appSource = fs.readFileSync(path.join(ROOT, 'public', 'app.js'), 'utf8');

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

test('buildQwenT2I uses the GGUF unet loader, not the diffusers pipeline node', () => {
  const fn = extractFunction(serverSource, 'buildQwenT2I');
  assert.match(fn, /class_type:\s*'UnetLoaderGGUF'/);
  // The bf16 diffusers pipeline needs 57.70GB resident vs 33.46GB of VRAM.
  assert.doesNotMatch(fn, /QwenImageTextToImageLocal/);
});

test('buildQwenT2I loads the scaled-fp8 text encoder via the REGULAR CLIPLoader', () => {
  const fn = extractFunction(serverSource, 'buildQwenT2I');
  assert.match(fn, /class_type:\s*'CLIPLoader'/);
  // CLIPLoaderGGUF rejects scaled fp8: "Mixing scaled FP8 with GGUF is not supported".
  assert.doesNotMatch(fn, /CLIPLoaderGGUF/);
});

test('buildQwenT2I produces SaveImage node with KreaStudio/qwen prefix', () => {
  assert.match(serverSource, /filename_prefix:\s*'KreaStudio\/qwen'/);
  const fn = extractFunction(serverSource, 'buildQwenT2I');
  assert.match(fn, /images:\s*\[['"]decode['"]\s*,\s*0\]/);
});

test('buildQwenT2I defaults the transformer to the Q4_K_M GGUF quant', () => {
  assert.match(serverSource, /settings\.qwenImageGgufModel\s*\|\|\s*'qwen-image-2512-Q4_K_M\.gguf'/);
});

test('settings default qwenImageGgufModel / text encoder / vae', () => {
  assert.match(serverSource, /qwenImageGgufModel:\s*'qwen-image-2512-Q4_K_M\.gguf'/);
  assert.match(serverSource, /qwenImageTextEncoder:\s*'qwen_2\.5_vl_7b_fp8_scaled\.safetensors'/);
  assert.match(serverSource, /qwenImageVae:\s*'qwen_image_vae\.safetensors'/);
});

test('buildQwenT2I passes steps and cfg to the sampler', () => {
  assert.match(serverSource, /steps:\s*p\.steps/);
  assert.match(serverSource, /cfg:\s*p\.cfg/);
});

test('buildQwenT2I clamps batch via the latent batch_size', () => {
  assert.match(serverSource, /batch_size:\s*Math\.min\(p\.batch\s*\|\|\s*1,\s*4\)/);
});

test('buildGenerationGraph dispatches qwen engine to buildQwenT2I', () => {
  assert.match(serverSource, /p\.engine\s*===\s*['"]qwen['"]\s*\??\s*buildQwenT2I\s*:\s*buildT2I/);
});

test('T2I_ENGINES is defined as ["qwen"]', () => {
  assert.match(serverSource, /const T2I_ENGINES\s*=\s*\['qwen'\]/);
});

test('the GGUF transformer, text encoder, and VAE all exist on disk', () => {
  const install = JSON.parse(fs.readFileSync(path.join(ROOT, 'install.json'), 'utf8'));
  const comfyModels = (install.comfy && install.comfy.modelsPath) || '';
  assert.ok(comfyModels, 'install.json must declare comfy.modelsPath');

  const roots = [comfyModels];
  const extraPathsFile = path.join(path.dirname(comfyModels), 'extra_model_paths.yaml');
  if (fs.existsSync(extraPathsFile)) {
    const yaml = fs.readFileSync(extraPathsFile, 'utf8');
    for (const m of yaml.matchAll(/^\s*base_path:\s*(.+?)\s*$/gm)) {
      const base = m[1].replace(/^["']|["']$/g, '');
      if (base) roots.push(base);
    }
  }

  const wanted = [
    ['diffusion_models', 'qwen-image-2512-Q4_K_M.gguf'],
    ['text_encoders', 'qwen_2.5_vl_7b_fp8_scaled.safetensors'],
    ['vae', 'qwen_image_vae.safetensors'],
  ];
  const missing = [];
  for (const [dir, file] of wanted) {
    const found = roots.some((r) => fs.existsSync(path.join(r, dir, file)));
    if (!found) missing.push(`${file} in ${dir}`);
  }
  if (missing.length > 0) {
    console.warn(`Qwen Image T2I GGUF models not present: ${missing.join(', ')}. ` +
      'Skip this assertion on machines without the models downloaded. ' +
      'The server still builds the graph; it fails at ComfyUI model load time if a file is absent.');
    return;
  }
  for (const [dir, file] of wanted) {
    const found = roots.some((r) => fs.existsSync(path.join(r, dir, file)));
    assert.ok(found, `${file} not found in ${dir} under any configured model root: ${roots.join(' | ')}`);
  }
});

test('the live data dir comes from config, not the repo ./data folder', () => {
  // Guard against a whole class of false diagnostics. This install is
  // portable: install.json sets dataDir to E:\MixStudioData, and the repo's
  // ./data is a STALE pre-migration copy. Reading ./data/db.json shows a
  // frozen gallery and makes working generations look like failures.
  const { resolveRuntimeConfig } = require('../lib/runtime-config.js');
  const runtime = resolveRuntimeConfig(ROOT);
  const repoData = path.join(ROOT, 'data');

  if (runtime.installMode === 'portable' && fs.existsSync(path.join(repoData, 'db.json'))) {
    assert.notEqual(
      path.resolve(runtime.dataDir),
      path.resolve(repoData),
      'portable install still points at the repo ./data folder'
    );
    // If the fossil is present, the live db must be the bigger/newer one.
    const liveDb = path.join(runtime.dataDir, 'db.json');
    assert.ok(fs.existsSync(liveDb), `live db.json missing at ${liveDb}`);
  }
});

// ---------------------------------------------------------------------------
// Functional tests — extract functions into a sandboxed VM context with
// mock dependencies, then call them and assert on the returned graph.
// ---------------------------------------------------------------------------

// --- Minimal mock context for buildQwenT2I ---
// buildQwenT2I needs: settings (object), filterInputs (function).
// We provide a mock settings object and an identity filterInputs.
const qwenContext = vm.createContext({
  settings: {
    qwenImageModel: undefined,
    qwenImageLightningLora: 'Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors',
  },
  filterInputs: (graph) => graph,
  // buildQwenT2I clamps Lightning steps/strength; mirror the real helpers.
  clampInt: (v, min, max, dflt) => {
    const n = Math.floor(Number(v));
    return Number.isFinite(n) ? Math.min(max, Math.max(min, n)) : dflt;
  },
  clampNum: (v, min, max, dflt) => {
    const n = Number(v);
    return Number.isFinite(n) ? Math.min(max, Math.max(min, n)) : dflt;
  },
});
vm.runInContext(extractFunction(serverSource, 'buildQwenT2I'), qwenContext);
const { buildQwenT2I } = qwenContext;

// --- Minimal mock context for buildGenerationGraph ---
// buildGenerationGraph needs: buildQwenT2I, buildT2I, hasActiveRegions,
// buildRegionalT2I, filterInputs, settings, T2I_ENGINES.
const genContext = vm.createContext({
  settings: {
    qwenImageModel: undefined,
    qwenImageLightningLora: 'Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors',
  },
  T2I_ENGINES: ['qwen'],
  filterInputs: (graph) => graph,
  clampInt: (v, min, max, dflt) => {
    const n = Math.floor(Number(v));
    return Number.isFinite(n) ? Math.min(max, Math.max(min, n)) : dflt;
  },
  clampNum: (v, min, max, dflt) => {
    const n = Number(v);
    return Number.isFinite(n) ? Math.min(max, Math.max(min, n)) : dflt;
  },
  hasActiveRegions: () => false,
  buildRegionalT2I: async () => { throw new Error('buildRegionalT2I should not be called'); },
  buildT2I: async (p) => ({
    unet: { class_type: 'MockUNet', inputs: { ckpt_name: p.model } },
    clip: { class_type: 'CLIPLoader' },
    vae: { class_type: 'VAELoader' },
    save: { class_type: 'SaveImage', inputs: { images: ['decode', 0], filename_prefix: 'KreaStudio/gen' } },
  }),
});
vm.runInContext(extractFunction(serverSource, 'buildQwenT2I'), genContext);
vm.runInContext(extractFunction(serverSource, 'buildGenerationGraph'), genContext);
const { buildGenerationGraph } = genContext;

function spy(ctx, name, implementation) {
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

test('buildQwenT2I produces a full native GGUF graph', async () => {
  const graph = await buildQwenT2I({
    prompt: 'a cat wearing sunglasses',
    width: 1024,
    height: 1024,
    seed: 42,
    steps: 20,
    cfg: 2.5,
  });

  assert.equal(graph.unet.class_type, 'UnetLoaderGGUF');
  assert.equal(graph.clip.class_type, 'CLIPLoader');
  assert.equal(graph.vae.class_type, 'VAELoader');
  assert.equal(graph.pos.class_type, 'CLIPTextEncode');
  assert.equal(graph.neg.class_type, 'CLIPTextEncode');
  assert.equal(graph.latent.class_type, 'EmptySD3LatentImage');
  assert.equal(graph.sampler.class_type, 'KSampler');
  assert.equal(graph.decode.class_type, 'VAEDecode');
  assert.equal(graph.save.class_type, 'SaveImage');
  assert.equal(graph.api, undefined, 'no diffusers pipeline node remains');
});

test('buildQwenT2I wires the graph edges correctly', async () => {
  // Opt out of Lightning so this asserts the plain unet -> sampler wiring.
  const graph = await buildQwenT2I({
    prompt: 'test', width: 1024, height: 1024, seed: 1, steps: 20, cfg: 2.5, qwenLightning: false,
  });
  // Compare element-wise: arrays built inside the VM context are not
  // reference-equal to arrays created out here, so deepEqual/deepStrictEqual
  // fails on otherwise identical [node, slot] link pairs.
  const link = (v) => [v[0], v[1]];
  assert.deepEqual(link(graph.pos.inputs.clip), ['clip', 0]);
  assert.deepEqual(link(graph.neg.inputs.clip), ['clip', 0]);
  assert.deepEqual(link(graph.sampler.inputs.model), ['unet', 0]);
  assert.deepEqual(link(graph.sampler.inputs.positive), ['pos', 0]);
  assert.deepEqual(link(graph.sampler.inputs.negative), ['neg', 0]);
  assert.deepEqual(link(graph.sampler.inputs.latent_image), ['latent', 0]);
  assert.deepEqual(link(graph.decode.inputs.samples), ['sampler', 0]);
  assert.deepEqual(link(graph.decode.inputs.vae), ['vae', 0]);
  assert.deepEqual(link(graph.save.inputs.images), ['decode', 0]);
});

test('buildQwenT2I SaveImage uses KreaStudio/qwen filename prefix', async () => {
  const graph = await buildQwenT2I({ prompt: 'test', width: 1024, height: 1024, seed: 1, steps: 20, cfg: 2.5 });
  assert.equal(graph.save.inputs.filename_prefix, 'KreaStudio/qwen');
});

test('buildQwenT2I defaults to the Q4_K_M GGUF quant that fits 33.46GB VRAM', async () => {
  const graph = await buildQwenT2I({ prompt: 'test', width: 1024, height: 1024, seed: 1, steps: 20, cfg: 2.5 });
  assert.equal(graph.unet.inputs.unet_name, 'qwen-image-2512-Q4_K_M.gguf');
});

test('buildQwenT2I loads the qwen_image-type text encoder and VAE', async () => {
  const graph = await buildQwenT2I({ prompt: 'test', width: 1024, height: 1024, seed: 1, steps: 20, cfg: 2.5 });
  assert.equal(graph.clip.inputs.clip_name, 'qwen_2.5_vl_7b_fp8_scaled.safetensors');
  assert.equal(graph.clip.inputs.type, 'qwen_image');
  assert.equal(graph.vae.inputs.vae_name, 'qwen_image_vae.safetensors');
});

test('buildQwenT2I honours settings overrides for unet/clip/vae', async () => {
  qwenContext.settings.qwenImageGgufModel = 'qwen-image-edit-Q4_K_M.gguf';
  qwenContext.settings.qwenImageTextEncoder = 'custom_encoder.safetensors';
  qwenContext.settings.qwenImageVae = 'custom_vae.safetensors';
  const graph = await buildQwenT2I({ prompt: 'test', width: 1024, height: 1024, seed: 1, steps: 20, cfg: 2.5 });
  assert.equal(graph.unet.inputs.unet_name, 'qwen-image-edit-Q4_K_M.gguf');
  assert.equal(graph.clip.inputs.clip_name, 'custom_encoder.safetensors');
  assert.equal(graph.vae.inputs.vae_name, 'custom_vae.safetensors');
  qwenContext.settings.qwenImageGgufModel = undefined;
  qwenContext.settings.qwenImageTextEncoder = undefined;
  qwenContext.settings.qwenImageVae = undefined;
});

// ---------------------------------------------------------------------------
// Tests: prompt / dimension / sampler plumbing
// ---------------------------------------------------------------------------

test('buildQwenT2I puts the prompt on the positive encoder', async () => {
  const graph = await buildQwenT2I({ prompt: 'a futuristic city', width: 1024, height: 1024, seed: 1, steps: 20, cfg: 2.5 });
  assert.equal(graph.pos.inputs.text, 'a futuristic city');
});

test('buildQwenT2I puts negativePrompt on the negative encoder', async () => {
  const graph = await buildQwenT2I({
    prompt: 'a cat',
    negativePrompt: 'blurry, low quality',
    width: 1024,
    height: 1024,
    seed: 1,
    steps: 20,
    cfg: 2.5,
  });
  assert.equal(graph.neg.inputs.text, 'blurry, low quality');
});

test('buildQwenT2I defaults the negative prompt to empty string', async () => {
  const graph = await buildQwenT2I({ prompt: 'test', width: 1024, height: 1024, seed: 1, steps: 20, cfg: 2.5 });
  assert.equal(graph.neg.inputs.text, '');
});

test('buildQwenT2I passes exact non-square dims to the latent (no bucket snapping)', async () => {
  // These three ratios were verified end-to-end on the Arc B70.
  for (const [w, h] of [[1024, 1024], [1024, 512], [768, 1152]]) {
    const graph = await buildQwenT2I({ prompt: 'test', width: w, height: h, seed: 1, steps: 20, cfg: 2.5 });
    assert.equal(graph.latent.inputs.width, w);
    assert.equal(graph.latent.inputs.height, h);
  }
});

test('buildQwenT2I passes seed, steps, cfg and sampler settings to KSampler', async () => {
  // Lightning off: the caller's steps/cfg must pass through untouched.
  const graph = await buildQwenT2I({
    prompt: 'test', width: 1024, height: 1024, seed: 12345, steps: 20, cfg: 2.5, qwenLightning: false,
  });
  assert.equal(graph.sampler.inputs.seed, 12345);
  assert.equal(graph.sampler.inputs.steps, 20);
  assert.equal(graph.sampler.inputs.cfg, 2.5);
  assert.equal(graph.sampler.inputs.sampler_name, 'euler');
  assert.equal(graph.sampler.inputs.scheduler, 'simple');
  assert.equal(graph.sampler.inputs.denoise, 1.0);
});

test('buildQwenT2I clamps batch to max 4 and defaults to 1', async () => {
  const many = await buildQwenT2I({ prompt: 'test', width: 1024, height: 1024, seed: 1, steps: 20, cfg: 2.5, batch: 10 });
  assert.equal(many.latent.inputs.batch_size, 4);
  const one = await buildQwenT2I({ prompt: 'test', width: 1024, height: 1024, seed: 1, steps: 20, cfg: 2.5 });
  assert.equal(one.latent.inputs.batch_size, 1);
});

test('buildQwenT2I uses enhancedText when available', async () => {
  const graph = await buildQwenT2I({
    prompt: 'original prompt',
    enhancedText: 'enhanced prompt with details',
    width: 1024,
    height: 1024,
    seed: 1,
    steps: 20,
    cfg: 2.5,
  });
  assert.equal(graph.pos.inputs.text, 'enhanced prompt with details');
});

// ---------------------------------------------------------------------------
// Tests: Lightning 4-step acceleration (measured 281.3s -> 45.1s, 6.24x)
// ---------------------------------------------------------------------------

test('buildQwenT2I inserts the Lightning LoRA by default and routes the model through it', async () => {
  const graph = await buildQwenT2I({ prompt: 'test', width: 1024, height: 1024, seed: 1, steps: 4, cfg: 1 });
  assert.ok(graph.lightning, 'lightning LoRA node present by default');
  assert.equal(graph.lightning.class_type, 'LoraLoaderModelOnly');
  assert.match(graph.lightning.inputs.lora_name, /Lightning-4steps/);
  // LoraLoaderModelOnly patches the model only — the CLIP must stay direct.
  assert.deepEqual([graph.lightning.inputs.model[0], graph.lightning.inputs.model[1]], ['unet', 0]);
  assert.deepEqual([graph.sampler.inputs.model[0], graph.sampler.inputs.model[1]], ['lightning', 0]);
  assert.deepEqual([graph.pos.inputs.clip[0], graph.pos.inputs.clip[1]], ['clip', 0]);
});

test('buildQwenT2I forces cfg 1 with Lightning (it is CFG-distilled)', async () => {
  // A stale form submission carrying cfg 2.5 must not blow out the image.
  const graph = await buildQwenT2I({ prompt: 'test', width: 1024, height: 1024, seed: 1, steps: 4, cfg: 2.5 });
  assert.equal(graph.sampler.inputs.cfg, 1);
});

test('buildQwenT2I clamps Lightning steps into the trained 4-8 range', async () => {
  const tooMany = await buildQwenT2I({ prompt: 'test', width: 1024, height: 1024, seed: 1, steps: 20, cfg: 1 });
  assert.equal(tooMany.sampler.inputs.steps, 8);
  const tooFew = await buildQwenT2I({ prompt: 'test', width: 1024, height: 1024, seed: 1, steps: 1, cfg: 1 });
  assert.equal(tooFew.sampler.inputs.steps, 4);
  const inRange = await buildQwenT2I({ prompt: 'test', width: 1024, height: 1024, seed: 1, steps: 6, cfg: 1 });
  assert.equal(inRange.sampler.inputs.steps, 6);
});

test('buildQwenT2I honours qwenLightning:false — no LoRA, caller steps/cfg respected', async () => {
  const graph = await buildQwenT2I({
    prompt: 'test', width: 1024, height: 1024, seed: 1, steps: 20, cfg: 2.5, qwenLightning: false,
  });
  assert.equal(graph.lightning, undefined, 'no lightning node when opted out');
  assert.deepEqual([graph.sampler.inputs.model[0], graph.sampler.inputs.model[1]], ['unet', 0]);
  assert.equal(graph.sampler.inputs.steps, 20);
  assert.equal(graph.sampler.inputs.cfg, 2.5);
});

test('buildQwenT2I clamps Lightning LoRA strength to 0-2, default 1', async () => {
  const dflt = await buildQwenT2I({ prompt: 'test', width: 1024, height: 1024, seed: 1, steps: 4, cfg: 1 });
  assert.equal(dflt.lightning.inputs.strength_model, 1);
  const high = await buildQwenT2I({ prompt: 'test', width: 1024, height: 1024, seed: 1, steps: 4, cfg: 1, qwenLightningStrength: 9 });
  assert.equal(high.lightning.inputs.strength_model, 2);
  const low = await buildQwenT2I({ prompt: 'test', width: 1024, height: 1024, seed: 1, steps: 4, cfg: 1, qwenLightningStrength: -3 });
  assert.equal(low.lightning.inputs.strength_model, 0);
});

test('server normalizes qwenLightning to a boolean and clamps its strength', () => {
  assert.match(serverSource, /p\.qwenLightning\s*=\s*p\.qwenLightning\s*!==\s*false/);
  assert.match(serverSource, /p\.qwenLightningStrength\s*=\s*clampNum\(p\.qwenLightningStrength,\s*0,\s*2,\s*1\)/);
});

test('settings default qwenImageLightningLora to the 4-step Lightning file', () => {
  assert.match(serverSource, /qwenImageLightningLora:\s*'Qwen-Image-Edit-2511-Lightning-4steps-V1\.0-bf16\.safetensors'/);
});

// ---------------------------------------------------------------------------
// Tests: Qwen EDIT engine must also fit in VRAM (bf16 was 51.35GB of 33.46GB)
// ---------------------------------------------------------------------------

test('qwenEditUnet uses the GGUF quant, not the 39GB bf16 transformer', () => {
  assert.match(serverSource, /qwenEditUnet:\s*'qwen-image-edit-Q4_K_M\.gguf'/);
  assert.doesNotMatch(serverSource, /qwenEditUnet:\s*'qwen_image_edit_2511_bf16\.safetensors'/);
});

test('PERSISTED settings do not pin an over-budget model over the code default', () => {
  // Stored settings.json is merged OVER DEFAULT_SETTINGS (server.js ~line 671),
  // so an old persisted value silently wins and a code-only fix is inert. This
  // actually happened: qwenEditUnet stayed on the 39GB bf16 file after the
  // default was changed, and the running graph kept loading UNETLoader with it
  // until the persisted value was patched through /api/settings.
  const { resolveRuntimeConfig } = require('../lib/runtime-config.js');
  const runtime = resolveRuntimeConfig(ROOT);
  const settingsFile = path.join(runtime.dataDir, 'settings.json');
  if (!fs.existsSync(settingsFile)) return; // fresh install: nothing persisted yet

  const stored = JSON.parse(fs.readFileSync(settingsFile, 'utf8'));
  // Models measured as exceeding this box's 33.46GB when fully resident.
  const OVER_BUDGET = [
    'qwen_image_edit_2511_bf16.safetensors',   // 39GB transformer -> 51.35GB total
  ];
  for (const [key, value] of Object.entries(stored)) {
    if (typeof value !== 'string') continue;
    const base = value.split(/[\\/]/).pop();
    assert.ok(
      !OVER_BUDGET.includes(base),
      `persisted settings.${key} pins ${base}, which does not fit in VRAM — patch it via POST /api/settings`
    );
  }
});

test('the Qwen edit GGUF transformer exists on disk', () => {
  const install = JSON.parse(fs.readFileSync(path.join(ROOT, 'install.json'), 'utf8'));
  const comfyModels = (install.comfy && install.comfy.modelsPath) || '';
  const roots = [comfyModels];
  const extraPathsFile = path.join(path.dirname(comfyModels), 'extra_model_paths.yaml');
  if (fs.existsSync(extraPathsFile)) {
    const yaml = fs.readFileSync(extraPathsFile, 'utf8');
    for (const m of yaml.matchAll(/^\s*base_path:\s*(.+?)\s*$/gm)) {
      const base = m[1].replace(/^["']|["']$/g, '');
      if (base) roots.push(base);
    }
  }
  if (!roots.some((r) => fs.existsSync(path.join(r, 'diffusion_models', 'qwen-image-edit-Q4_K_M.gguf')))) {
    console.warn('Qwen Image Edit GGUF (qwen-image-edit-Q4_K_M.gguf) not present under any configured model root. ' +
      'Skip this assertion on machines without the model downloaded. The server still builds the graph; ' +
      'it fails at ComfyUI model load time if the file is absent.');
    return;
  }
  const found = roots.some((r) => fs.existsSync(path.join(r, 'diffusion_models', 'qwen-image-edit-Q4_K_M.gguf')));
  assert.ok(found, 'qwen-image-edit-Q4_K_M.gguf not found under any configured model root');
});

// ---------------------------------------------------------------------------
// Tests: engine dispatch — qwen routes to buildQwenT2I, not buildT2I
// ---------------------------------------------------------------------------

test('buildGenerationGraph routes qwen engine to buildQwenT2I', async () => {
  const qwenSpy = spy(genContext, 'buildQwenT2I', async (p) => ({
    unet: { class_type: 'UnetLoaderGGUF', inputs: { unet_name: 'qwen-image-2512-Q4_K_M.gguf' } },
    save: { class_type: 'SaveImage' },
  }));
  const defaultSpy = spy(genContext, 'buildT2I', async (p) => ({
    unet: { class_type: 'MockUNet' },
    save: { class_type: 'SaveImage' },
  }));

  const graph = await buildGenerationGraph(
    { mode: 't2i', engine: 'qwen', prompt: 'test', seed: 1, steps: 4, cfg: 1, width: 1024, height: 1024 },
    []
  );

  assert.ok(qwenSpy.calledWith !== null, 'buildQwenT2I was called');
  assert.deepEqual(qwenSpy.calledWith, {
    mode: 't2i', engine: 'qwen', prompt: 'test', seed: 1, steps: 4, cfg: 1, width: 1024, height: 1024,
  });
  assert.equal(defaultSpy.calledWith, null, 'buildT2I was NOT called when engine is qwen');
  assert.equal(graph.unet.class_type, 'UnetLoaderGGUF');
});

test('buildGenerationGraph routes non-qwen engine to buildT2I', async () => {
  const qwenSpy = spy(genContext, 'buildQwenT2I', async (p) => ({
    unet: { class_type: 'UnetLoaderGGUF' },
    save: { class_type: 'SaveImage' },
  }));
  const defaultSpy = spy(genContext, 'buildT2I', async (p) => ({
    unet: { class_type: 'MockUNet' },
    save: { class_type: 'SaveImage' },
  }));

  const graph = await buildGenerationGraph(
    { mode: 't2i', engine: undefined, prompt: 'test', seed: 1, steps: 4, cfg: 1, width: 1024, height: 1024 },
    []
  );

  assert.equal(qwenSpy.calledWith, null, 'buildQwenT2I was NOT called when engine is not qwen');
  assert.ok(defaultSpy.calledWith !== null, 'buildT2I was called');
  assert.equal(graph.unet.class_type, 'MockUNet');
});

test('buildGenerationGraph dispatches even when engine is the string "qwen"', async () => {
  const qwenSpy = spy(genContext, 'buildQwenT2I', async (p) => ({
    unet: { class_type: 'UnetLoaderGGUF' },
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
  assert.equal(graph.unet.class_type, 'UnetLoaderGGUF');
});

// ---------------------------------------------------------------------------
// Tests: Frontend (app.js) — Qwen Image engine UI
// ---------------------------------------------------------------------------

test('app.js defines CREATE_ENGINES with krea2 and qwen_image', () => {
  assert.match(appSource, /const CREATE_ENGINES = Object\.freeze\(\['krea2', 'qwen_image'\]\)/);
});

test('app.js defines CREATE_ENGINES_FEATURES for both engines', () => {
  assert.match(appSource, /CREATE_ENGINES_FEATURES.*krea2.*qwen_image/);
});

test('app.js defines Qwen Image model variants (3.0-pro and 3.0)', () => {
  assert.match(appSource, /qwen-image-3\.0-pro.*Qwen Image 3\.0 Pro/);
  assert.match(appSource, /qwen-image-3\.0.*Qwen Image 3\.0/);
});

test('app.js defines Qwen Image precision options (bf16 and fp8_lightning)', () => {
  assert.match(appSource, /value: 'bf16'.*label: 'BF16'/);
  assert.match(appSource, /value: 'fp8_lightning'.*label: 'FP8 Lightning'/);
});

test('app.js sets createEngine state default to krea2 with qwen_image alternative', () => {
  assert.match(appSource, /createEngine: 'krea2'.*krea2 \| qwen_image/);
});

test('app.js defines Qwen Image canvas constraints (256-2560, step 16)', () => {
  assert.match(appSource, /QWEN_IMAGE_MAX_TOTAL_PIXELS = 2560 \* 2560/);
  assert.match(appSource, /minDim: 256/);
  assert.match(appSource, /maxDim: 2560/);
  assert.match(appSource, /step: 16/);
});

test('app.js renders Qwen Image label in engine picker', () => {
  assert.match(appSource, /qwen_image.*Qwen Image|'qwen_image'.*'Qwen Image'/);
});

test('app.js applies Qwen Image canvas constraints when engine is qwen_image', () => {
  assert.match(appSource, /state\.createEngine === 'qwen_image'/);
  assert.match(appSource, /qwenConstraints\.clampWidth/);
  assert.match(appSource, /qwenConstraints\.clampHeight/);
});

test('app.js clamps width/height to step 16 for Qwen Image', () => {
  assert.match(appSource, /Math\.round\(w \/ 16\) \* 16/);
  assert.match(appSource, /Math\.max\(256, Math\.min\(2560/);
});

// ---------------------------------------------------------------------------
// Tests: Documentation
// ---------------------------------------------------------------------------

test('docs/qwen-image.md exists and contains key sections', () => {
  const docsPath = path.join(ROOT, 'docs', 'qwen-image.md');
  assert.ok(fs.existsSync(docsPath), 'docs/qwen-image.md exists');
  const docs = fs.readFileSync(docsPath, 'utf8');
  assert.match(docs, /Qwen Image/i);
  assert.match(docs, /VRAM|memory/i);
  assert.match(docs, /canvas|resolution/i);
  assert.match(docs, /troubleshoot/i);
});
