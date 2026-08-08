'use strict';

const EDIT_IMAGE_INPUT_ENGINES = new Set(['klein4', 'klein9', 'qwen', 'krea2ref', 'krea2remix']);

function supportsImageInput(engine) {
  return EDIT_IMAGE_INPUT_ENGINES.has(String(engine || ''));
}

function imageInputNotAcceptedMessage(imageName) {
  const name = String(imageName || 'the reference image');
  return `The model does not accept image input. The image '${name}' was provided but this model type only supports text prompts. Switch to an edit engine that supports image reference (Krea 2 Edit, Klein 4B/9B, or Qwen Edit) or remove the reference image.`;
}

const COMFY_IMAGE_INPUT_PATTERNS = [
  /does not support.*image/i,
  /image.*(not\s+|un)?support/i,
  /does not accept.*image/i,
  /image.*not.*accept/i,
  /not.*support.*image/i,
  /invalid.*image.*input/i,
  /model.*does not.*accept.*image/i,
];

function matchesImageInputPattern(text) {
  return COMFY_IMAGE_INPUT_PATTERNS.some((pattern) => pattern.test(String(text || '')));
}

function isComfyImageInputError(error) {
  if (!error) return false;
  const message = String(error.message || error);
  if (message.indexOf('ComfyUI') === -1) return false;
  return matchesImageInputPattern(message);
}

function nodeErrorsHasImageInput(nodeErrorsStr) {
  return matchesImageInputPattern(nodeErrorsStr);
}

module.exports = {
  EDIT_IMAGE_INPUT_ENGINES,
  supportsImageInput,
  imageInputNotAcceptedMessage,
  isComfyImageInputError,
  nodeErrorsHasImageInput,
  matchesImageInputPattern,
};
