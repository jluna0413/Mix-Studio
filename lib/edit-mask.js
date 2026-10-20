'use strict';

const EDIT_MASK_ENGINES = new Set(['klein4', 'klein9', 'qwen', 'krea2']);
const DEFAULT_MASK_EXPAND = 14;
const DEFAULT_MASK_INFLUENCE = 78;
const SAM3_MASK_CLASSES = [
  'LoadImage',
  'SAM3Grounding',
  'SaveImage',
];

// The deployed ComfyUI-SAM3 node (classic API, registers on ComfyUI 0.30.0)
// exposes only a text-prompted segmenter. Point-based selection is not
// available in that node, so a points-only request degrades to a generic
// text prompt rather than failing the whole Smart Select call.
const SAM3_DEFAULT_TEXT_PROMPT = 'the main subject';

function supportsEditMask(engine) {
  return EDIT_MASK_ENGINES.has(String(engine || ''));
}

function maskExpand(value) {
  const pixels = Math.round(Number(value));
  return Math.max(6, Math.min(32, Number.isFinite(pixels) ? pixels : DEFAULT_MASK_EXPAND));
}

function maskInfluence(value) {
  const percent = Math.round(Number(value));
  return Math.max(25, Math.min(100, Number.isFinite(percent) ? percent : DEFAULT_MASK_INFLUENCE));
}

function maskInfluenceDenoise(value) {
  return Number((maskInfluence(value) / 100).toFixed(2));
}

function localizedEditPrompt(prompt) {
  const request = String(prompt || '').trim();
  const context = 'Localized edit only. Preserve the exact surrounding scene, camera perspective, depth, lighting direction, shadows, reflections, material texture, and color balance. Seamlessly integrate the requested change with the adjacent pixels.';
  return request ? `${request}. ${context}` : context;
}

function appendEditMaskNodes(graph, options = {}) {
  const prefix = String(options.prefix || 'edit_mask');
  const maskImageName = String(options.maskImageName || '').trim();
  if (!graph || !maskImageName || !Array.isArray(options.samples)) return null;

  const loadKey = `${prefix}_load`;
  const channelKey = `${prefix}_channel`;
  const growKey = `${prefix}_grow`;
  const latentKey = `${prefix}_latent`;
  graph[loadKey] = { class_type: 'LoadImage', inputs: { image: maskImageName } };
  graph[channelKey] = { class_type: 'ImageToMask', inputs: { image: [loadKey, 0], channel: 'red' } };
  graph[growKey] = {
    class_type: 'GrowMask',
    inputs: { mask: [channelKey, 0], expand: maskExpand(options.expand), tapered_corners: true },
  };
  graph[latentKey] = {
    class_type: 'SetLatentNoiseMask',
    inputs: { samples: options.samples, mask: [growKey, 0] },
  };
  return { latent: [latentKey, 0], mask: [growKey, 0] };
}

function appendEditMaskComposite(graph, options = {}) {
  if (!graph || !Array.isArray(options.original) || !Array.isArray(options.generated) || !Array.isArray(options.mask)) return null;
  const key = String(options.key || 'edit_mask_composite');
  graph[key] = {
    class_type: 'ImageCompositeMasked',
    inputs: {
      destination: options.original,
      source: options.generated,
      x: 0,
      y: 0,
      resize_source: true,
      mask: options.mask,
    },
  };
  return [key, 0];
}

function normalizeMaskPoints(points) {
  return (Array.isArray(points) ? points : []).slice(0, 10).map((point) => ({
    x: Math.max(0, Math.min(1, Number(point && point.x) || 0)),
    y: Math.max(0, Math.min(1, Number(point && point.y) || 0)),
    foreground: !point || point.foreground !== false,
  }));
}

/**
 * Build a transient SAM3 graph that saves a white-on-black mask only.
 * The deployed ComfyUI-SAM3 node exposes SAM3Grounding, which takes the
 * LoadSAM3Model config and returns (masks, visualization, boxes, scores).
 * Slot 0 is a MASK, so MaskToImage bridges it into the saved preview and
 * the rest of the pipeline only sees the cutout.
 */
function buildSam3MaskGraph(options = {}) {
  const imageName = String(options.imageName || '').trim();
  let prompt = String(options.prompt || '').trim();
  const points = normalizeMaskPoints(options.points);
  const prefix = String(options.prefix || 'sam3').replace(/[^a-zA-Z0-9_]/g, '').slice(0, 24) || 'sam3';
  if (!imageName) throw new Error('A source image is required');
  if (!prompt && !points.length) throw new Error('Describe an object or tap the image');
  if (!prompt) prompt = SAM3_DEFAULT_TEXT_PROMPT;

  const graph = {
    [`${prefix}_model`]: {
      class_type: 'LoadSAM3Model',
      inputs: { precision: 'auto', compile: false },
    },
    [`${prefix}_source`]: { class_type: 'LoadImage', inputs: { image: imageName } },
    [`${prefix}_segment`]: {
      class_type: 'SAM3Grounding',
      inputs: {
        sam3_model_config: [`${prefix}_model`, 0],
        image: [`${prefix}_source`, 0],
        confidence_threshold: 0.2,
        text_prompt: prompt.slice(0, 240),
      },
    },
  };
  graph[`${prefix}_mask`] = {
    class_type: 'MaskToImage',
    inputs: { mask: [`${prefix}_segment`, 0] },
  };
  graph[`${prefix}_save_mask`] = {
    class_type: 'SaveImage',
    inputs: { images: [`${prefix}_mask`, 0], filename_prefix: `mixbox_${prefix}_mask` },
  };
  return graph;
}

/**
 * Build one self-contained SAM3 mask graph per text prompt. Each graph uses a
 * unique key prefix so several can be queued sequentially without node-name
 * collisions. Returns an array of { prompt, graph } (one entry per prompt).
 * The first prompt with no points falls back to SAM3_DEFAULT_TEXT_PROMPT so a
 * bare "character" request still segments something.
 */
function buildSam3PartsGraphs({ imageName, prompts = [] } = {}) {
  const cleanImage = String(imageName || '').trim();
  if (!cleanImage) throw new Error('A source image is required');
  const list = (Array.isArray(prompts) ? prompts : [])
    .map((p) => String(p || '').trim().slice(0, 240))
    .filter(Boolean)
    .slice(0, 12);
  if (!list.length) list.push(SAM3_DEFAULT_TEXT_PROMPT);
  return list.map((prompt, index) => ({
    prompt,
    graph: buildSam3MaskGraph({ imageName: cleanImage, prompt, prefix: `p${index}` }),
  }));
}

module.exports = {
  EDIT_MASK_ENGINES,
  DEFAULT_MASK_EXPAND,
  DEFAULT_MASK_INFLUENCE,
  SAM3_MASK_CLASSES,
  localizedEditPrompt,
  maskExpand,
  maskInfluence,
  maskInfluenceDenoise,
  supportsEditMask,
  appendEditMaskNodes,
  appendEditMaskComposite,
  buildSam3MaskGraph,
  buildSam3PartsGraphs,
  normalizeMaskPoints,
};
