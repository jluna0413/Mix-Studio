'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const {
  EDIT_IMAGE_INPUT_ENGINES,
  supportsImageInput,
  imageInputNotAcceptedMessage,
  isComfyImageInputError,
  nodeErrorsHasImageInput,
  matchesImageInputPattern,
} = require('../lib/edit-image-input');

const serverJs = fs.readFileSync(path.join(__dirname, '..', 'server.js'), 'utf8');

test('edit engines with image-reference graph builders are listed as supporting image input', () => {
  for (const engine of ['klein4', 'klein9', 'qwen', 'krea2ref', 'krea2remix']) {
    assert.equal(supportsImageInput(engine), true, `${engine} should support image input`);
  }
});

test('Krea 2 without mask falls through to T2I and does not support image reference', () => {
  assert.equal(supportsImageInput('krea2'), false);
  assert.equal(supportsImageInput(''), false);
  assert.equal(supportsImageInput(null), false);
  assert.equal(supportsImageInput(undefined), false);
  assert.equal(supportsImageInput('t2i'), false);
});

test('the EDIT_IMAGE_INPUT_ENGINES set excludes krea2 (T2I fallback)', () => {
  assert.ok(EDIT_IMAGE_INPUT_ENGINES.has('klein4'));
  assert.ok(EDIT_IMAGE_INPUT_ENGINES.has('krea2remix'));
  assert.ok(!EDIT_IMAGE_INPUT_ENGINES.has('krea2'));
});

test('imageInputNotAcceptedMessage names the image and lists replacement engines', () => {
  const msg = imageInputNotAcceptedMessage('ks_ref_12345.png');
  assert.match(msg, /The model does not accept image input/);
  assert.match(msg, /ks_ref_12345\.png/);
  assert.match(msg, /Krea 2 Edit, Klein 4B\/9B, or Qwen Edit/);
  assert.match(msg, /remove the reference image/);
});

test('imageInputNotAcceptedMessage falls back to a generic label when name is empty', () => {
  const msg = imageInputNotAcceptedMessage('');
  assert.match(msg, /The model does not accept image input/);
  assert.match(msg, /reference image/);
  assert.match(msg, /remove the reference image/);
});

test('isComfyImageInputError detects ComfyUI validation errors about image input', () => {
  assert.equal(isComfyImageInputError(
    new Error('ComfyUI validation: {"5":{"errors":[{"message":"does not support image input"}]}}')
  ), true);
  assert.equal(isComfyImageInputError(
    new Error('ComfyUI validation: {"3":{"errors":[{"message":"image input not supported by this model"}]}}')
  ), true);
  assert.equal(isComfyImageInputError(
    new Error('ComfyUI validation: {"7":{"errors":[{"message":"Model does not accept image"}]}}')
  ), true);
});

test('isComfyImageInputError detects ComfyUI fetch 400 errors about image input', () => {
  assert.equal(isComfyImageInputError(
    new Error('ComfyUI /prompt -> 400 {"error":{"type":"prompt_outputs_failed_validation","message":"Prompt outputs failed validation"},"node_errors":{"5":{"errors":[{"message":"does not support image input"}]}}}')
  ), true);
  assert.equal(isComfyImageInputError(
    new Error('ComfyUI /prompt -> 400 {"node_errors":{"3":{"errors":[{"message":"image input not supported"}]}}}')
  ), true);
});

test('isComfyImageInputError rejects unrelated ComfyUI validation errors', () => {
  assert.equal(isComfyImageInputError(
    new Error('ComfyUI validation: {"5":{"errors":[{"message":"required field missing"}]}}')
  ), false);
  assert.equal(isComfyImageInputError(
    new Error('ComfyUI validation: {"5":{"errors":[{"message":"model not found"}]}}')
  ), false);
  assert.equal(isComfyImageInputError(
    new Error('ComfyUI /prompt -> 400 {"error":"Some other error"}')
  ), false);
});

test('isComfyImageInputError rejects non-validation errors and non-errors', () => {
  assert.equal(isComfyImageInputError(new Error('ComfyUI connection failed')), false);
  assert.equal(isComfyImageInputError(new Error('something else')), false);
  assert.equal(isComfyImageInputError(null), false);
  assert.equal(isComfyImageInputError(undefined), false);
  assert.equal(isComfyImageInputError({ message: 'no error here' }), false);
  assert.equal(isComfyImageInputError(new Error('image does not support this')), false);
});

test('nodeErrorsHasImageInput detects image input patterns in node_errors JSON', () => {
  assert.equal(nodeErrorsHasImageInput('{"5":{"errors":[{"message":"does not support image input"}]}}'), true);
  assert.equal(nodeErrorsHasImageInput('{"3":{"errors":[{"message":"image input not supported by this model"}]}}'), true);
  assert.equal(nodeErrorsHasImageInput('{"7":{"errors":[{"message":"Model does not accept image"}]}}'), true);
});

test('nodeErrorsHasImageInput rejects unrelated errors', () => {
  assert.equal(nodeErrorsHasImageInput('{"5":{"errors":[{"message":"required field missing"}]}}'), false);
  assert.equal(nodeErrorsHasImageInput('{"5":{"errors":[{"message":"model not found"}]}}'), false);
});

test('matchesImageInputPattern detects image input errors without ComfyUI prefix', () => {
  assert.equal(matchesImageInputPattern('does not support image input'), true);
  assert.equal(matchesImageInputPattern('image input not supported by this model'), true);
  assert.equal(matchesImageInputPattern('Model does not accept image'), true);
  assert.equal(matchesImageInputPattern('This CLIP model does not support image input'), true);
  assert.equal(matchesImageInputPattern('required field missing'), false);
  assert.equal(matchesImageInputPattern('model not found'), false);
  assert.equal(matchesImageInputPattern(''), false);
});

test('server imports edit-image-input helpers and guards the generate route', () => {
  assert.match(serverJs, /require\(['"]\.\/lib\/edit-image-input['"]\)/);
  assert.match(serverJs, /supportsImageInput\(p\.editEngine\)/);
  assert.match(serverJs, /isComfyImageInputError\(error\)/);
  assert.match(serverJs, /imageInputNotAcceptedMessage\(refNames\[0\]\)/);
  assert.match(serverJs, /code: 'comfy_image_input_not_supported'/);
  assert.match(serverJs, /nodeErrorsHasImageInput\(/);
  assert.match(serverJs, /matchesImageInputPattern\(/);
});

test('server proactively rejects reference images for engines that lack image support', () => {
  assert.match(serverJs, /refNames\.length && !supportsImageInput\(p\.editEngine\) && !p\.editOutpaint && !p\.maskImageName && !hasActiveRegions\(p\.regions\)/);
});

test('server intercepts comfyFetch errors in queuePrompt for image input support', () => {
  assert.match(serverJs, /isComfyImageInputError\(/);
  assert.match(serverJs, /comfy_image_input_not_supported/);
  assert.match(serverJs, /const promptUrl = settings\.comfyUrl/);
  assert.match(serverJs, /nodeErrorsHasImageInput\(nodeErrorsStr\)/);
});

test('server intercepts execution errors for image input support', () => {
  assert.match(serverJs, /matchesImageInputPattern\(exceptionMessage\)/);
  assert.match(serverJs, /matchesImageInputPattern\(histError\)/);
});

test('server applies offload config to edit/T2I generation jobs', () => {
  assert.match(serverJs, /applyOffloadToGraph\(graph, getOffloadConfig\(p\.gpuVendor, p\.vramGb\)\)/);
});

test('server sets gpuVendor and vramGb on params in the generate route', () => {
  assert.match(serverJs, /p\.gpuVendor = hardwareProfile\.gpuVendor/);
  assert.match(serverJs, /p\.vramGb = hardwareProfile\.vramGb/);
});
