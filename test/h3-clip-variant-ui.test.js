'use strict';

/**
 * UI wiring tests for the MiniMax H3 text-encoder precision control.
 *
 * The dropdown exposes three GPU-portable variants:
 *   nvfp4-awq (default), lightweight, distilled.
 * NVIDIA-only INT8 ConvRot options were removed.
 *
 * `h3ClipVariant` persists via app.js save/load -> autosave registry ->
 * install request modelVariants.h3 -> server normalize + shared-settings gate
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
    /id="setH3ClipVariant"[\s\S]*?value="nvfp4-awq"[\s\S]*?value="lightweight"[\s\S]*?value="distilled"/,
    'nvfp4-awq, lightweight, and distilled options must be offered in order'
  );
  assert.match(html, /<label for="setH3ClipVariant">/, 'must have an associated label');
});

test('NVIDIA-only INT8 ConvRot clip options are gone', () => {
  const variantSelect = html.slice(
    html.indexOf('id="setH3ClipVariant"'),
    html.indexOf('id="setH3FrameModelVariant"')
  );
  assert.doesNotMatch(
    variantSelect,
    /value="int8-convrot"/,
    'int8-convrot option must be removed from the H3 clip variant dropdown'
  );
  assert.doesNotMatch(
    variantSelect,
    /value="int8-convrot-uncensored"/,
    'int8-convrot-uncensored option must be removed from the H3 clip variant dropdown'
  );
  assert.doesNotMatch(
    variantSelect,
    /value="auto"/,
    'auto option must be removed from the H3 clip variant dropdown'
  );
});

test('app.js persists and restores h3ClipVariant', () => {
  assert.match(app, /h3ClipVariant: ['"]?\$\('#setH3ClipVariant'\)/, 'save path must read the select');
  assert.match(
    app,
    /\$\('#setH3ClipVariant'\)\.value = s\.h3ClipVariant \|\| 'nvfp4-awq'/,
    "load path must restore the select and default to 'nvfp4-awq'"
  );
});

test('h3ClipVariant is registered for settings autosave', () => {
  assert.match(
    app,
    /'setH3ClipVariant', 'setH3FrameModelVariant'/,
    'must be in the autosave id registry'
  );
});

test('install requests forward the selected variant as modelVariants.h3', () => {
  // Both install and repair calls must pass h3 from the select with a nvfp4-awq default.
  const matches = app.match(/h3:.*setH3ClipVariant.*nvfp4-awq/g) || [];
  assert.ok(
    matches.length >= 2,
    `both install and repair calls must forward h3 (found ${matches.length})`
  );
});

test('server defaults, normalizes, and gates h3ClipVariant', () => {
  assert.match(
    server,
    /h3ClipVariant: 'nvfp4-awq'/,
    'DEFAULT_SETTINGS must define h3ClipVariant as nvfp4-awq'
  );
  assert.match(
    server,
    /normalizeH3ClipVariant\(s\.h3ClipVariant\)/,
    'normalizeSettings must normalize it so junk cannot persist'
  );
  assert.match(
    server,
    /'h3ClipVariant', 'h3FrameModelVariant'/,
    'must be covered by the owner-only shared H3 settings gate'
  );
  assert.match(
    server,
    /const modelVariants = \{[\s\S]{0,200}krea2: krea2Variant,[\s\S]{0,120}h3: h3Variant,/,
    'install endpoint must pass h3 through to the installer'
  );
});

test('the control sits inside the MiniMax H3 settings section', () => {
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
