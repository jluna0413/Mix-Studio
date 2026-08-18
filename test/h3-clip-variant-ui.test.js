'use strict';

/**
 * UI wiring tests for the MiniMax H3 text-encoder precision control.
 *
 * Why this control exists: the int8_convrot qwen3vl encoder depends on
 * comfy_kitchen's ConvRot kernels, which ship CUDA-only. On Intel Arc (XPU)
 * those kernels are absent and H3 generation dies with
 * `AttributeError: 'NoneType' object has no attribute 'device'`. The nvfp4_awq
 * encoder uses the eager backend and works on both vendors.
 *
 * `h3ClipVariant` previously sat in settings.json with zero readers anywhere in
 * the codebase and no UI, so it could silently disagree with `h3Clip`. These
 * tests lock down the full wiring path:
 *   index.html select -> app.js save/load -> autosave registry ->
 *   install request modelVariants.h3 -> server normalize + shared-settings gate
 *
 * Reference: docs/comfyui-h3-arc-patch.md
 */

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');

const root = path.join(__dirname, '..');
const html = fs.readFileSync(path.join(root, 'public', 'index.html'), 'utf8');
const app = fs.readFileSync(path.join(root, 'public', 'app.js'), 'utf8');
const server = fs.readFileSync(path.join(root, 'server.js'), 'utf8');

test('Preferences exposes a text encoder precision selector with all variants', () => {
  assert.match(html, /id="setH3ClipVariant"/, 'select element must exist');
  assert.match(
    html,
    /id="setH3ClipVariant"[\s\S]*?value="auto"[\s\S]*?value="nvfp4-awq"[\s\S]*?value="int8-convrot"[\s\S]*?value="int8-convrot-uncensored"/,
    'all four variant options must be offered in order'
  );
  assert.match(html, /<label for="setH3ClipVariant">/, 'must have an associated label');
});

test('the selector warns that INT8 ConvRot is NVIDIA-only', () => {
  // Users must be able to see why the choice matters before picking a broken one.
  assert.match(html, /INT8 ConvRot[\s\S]{0,120}NVIDIA only/i);
  assert.match(html, /id="setH3ClipVariantHelp"/, 'help text element must exist');
  assert.match(
    html,
    /fails on Intel Arc/i,
    'help text must name the failing hardware explicitly'
  );
});

test('app.js persists and restores h3ClipVariant', () => {
  assert.match(
    app,
    /h3ClipVariant: \$\('#setH3ClipVariant'\)\.value/,
    'save path must read the select'
  );
  assert.match(
    app,
    /\$\('#setH3ClipVariant'\)\.value = s\.h3ClipVariant \|\| 'auto'/,
    "load path must restore the select and default to 'auto'"
  );
});

test('h3ClipVariant is registered for settings autosave', () => {
  // Without this the dropdown silently fails to persist on change.
  assert.match(
    app,
    /'setH3ClipVariant', 'setH3FrameModelVariant'/,
    'must be in the autosave id registry'
  );
});

test('install requests forward the selected variant as modelVariants.h3', () => {
  // The installer resolves the encoder per-vendor from modelVariants.h3.
  // Previously only krea2 was sent, so h3Variant was permanently 'auto'.
  const matches = app.match(/h3: \$\('#setH3ClipVariant'\)\?\.value \|\| 'auto'/g) || [];
  assert.ok(
    matches.length >= 2,
    `both install and repair calls must forward h3 (found ${matches.length})`
  );
});

test('server defaults, normalizes, and gates h3ClipVariant', () => {
  assert.match(
    server,
    /h3ClipVariant: 'auto'/,
    'DEFAULT_SETTINGS must define h3ClipVariant'
  );
  assert.match(
    server,
    /s\.h3ClipVariant = normalizeH3ClipVariant\(s\.h3ClipVariant\)/,
    'normalizeSettings must normalize it so junk cannot persist'
  );
  assert.match(
    server,
    /'h3ClipVariant', 'h3FrameModelVariant'/,
    'must be covered by the owner-only shared H3 settings gate'
  );
  assert.match(
    server,
    /modelVariants = \{ krea2: krea2Variant, h3: h3Variant \}/,
    'install endpoint must pass h3 through to the installer'
  );
});

test('the control sits inside the MiniMax H3 settings section', () => {
  // Guard against the field drifting into an unrelated model group.
  const h3SectionAt = html.indexOf('data-settings-model-section="h3"');
  const controlAt = html.indexOf('id="setH3ClipVariant"');
  const nextSectionAt = html.indexOf('data-settings-model-section="wan22"');

  assert.ok(h3SectionAt !== -1, 'H3 settings section must exist');
  assert.ok(controlAt !== -1, 'control must exist');
  assert.ok(
    controlAt > h3SectionAt && (nextSectionAt === -1 || controlAt < nextSectionAt),
    'control must live between the H3 section start and the next model section'
  );
});
