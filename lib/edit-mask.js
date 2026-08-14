'use strict';

const EDIT_MASK_ENGINES = new Set(['klein4', 'klein9', 'qwen', 'krea2']);
const DEFAULT_MASK_EXPAND = 14;
const DEFAULT_MASK_INFLUENCE = 78;
const SAM3_MASK_CLASSES = [
  'LoadImage',
  'SAM3TextSegment',
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
 * The deployed ComfyUI-SAM3 node exposes a single text-prompted segmenter
 * (SAM3TextSegment) which returns (segmented_image, mask_preview). We save
 * the mask_preview so the rest of the pipeline only sees the cutout.
 */
function buildSam3MaskGraph(options = {}) {
  const imageName = String(options.imageName || '').trim();
  let prompt = String(options.prompt || '').trim();
  const points = normalizeMaskPoints(options.points);
  if (!imageName) throw new Error('A source image is required');
  if (!prompt && !points.length) throw new Error('Describe an object or tap the image');
  if (!prompt) prompt = SAM3_DEFAULT_TEXT_PROMPT;

  const graph = {
    source: { class_type: 'LoadImage', inputs: { image: imageName } },
    sam3_segment: {
      class_type: 'SAM3TextSegment',
      inputs: {
        image: ['source', 0],
        text_prompt: prompt.slice(0, 240),
      },
    },
  };
  graph.save_mask = {
    class_type: 'SaveImage',
    inputs: { images: ['sam3_segment', 1], filename_prefix: 'mixbox_sam3_mask' },
  };
  return graph;
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
  normalizeMaskPoints,
};
