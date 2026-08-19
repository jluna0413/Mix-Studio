'use strict';

/**
 * Wiring tests for the MiniMax H3 Turbo low-VRAM control.
 *
 * The bug
 * -------
 * `MiniMaxH3TurboLoRA` accepts a `low_vram` flag that trades speed for a
 * smaller memory footprint. The builder honours it
 * (lib/video-workflows.js: `low_vram: opts.turboLowVram === true`) and the
 * route reads it (`server.js: turboLowVram: body.h3TurboLowVram === true`),
 * but NOTHING IN public/ EVER SENDS IT — verified by grep, 0 occurrences.
 *
 * So the flag was permanently false and the low-memory path unreachable, while
 * 608x384 x124 consumed 28.41 GB of 31.16 GB and then lost the device. This is
 * the third instance of the same defect class in this engine (see
 * h3-offload-callsites.test.js for h3R2V, and h3ClipVariant before it):
 * correct code that no caller reaches.
 */

const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');

const root = path.join(__dirname, '..');
const appJs = fs.readFileSync(path.join(root, 'public', 'app.js'), 'utf8');
const indexHtml = fs.readFileSync(path.join(root, 'public', 'index.html'), 'utf8');
const serverJs = fs.readFileSync(path.join(root, 'server.js'), 'utf8');
const videoWorkflows = fs.readFileSync(
  path.join(root, 'lib', 'video-workflows.js'), 'utf8'
);

test('the builder still honours the low_vram flag', () => {
  assert.match(
    videoWorkflows, /low_vram:\s*opts\.turboLowVram === true/,
    'MiniMaxH3TurboLoRA must pass through opts.turboLowVram'
  );
});

test('the route still reads h3TurboLowVram from the request body', () => {
  assert.match(
    serverJs, /turboLowVram:\s*body\.h3TurboLowVram === true/,
    'the animate route must read h3TurboLowVram'
  );
});

test('the frontend HAS a low-VRAM control (THE BUG: it had none)', () => {
  assert.match(
    indexHtml, /id="vidH3TurboLowVram"/,
    'index.html must expose a Turbo low-VRAM control'
  );
});

test('the control follows the app-native switch convention, not a raw checkbox', () => {
  // This app uses role="switch" buttons with aria-checked, not <input
  // type="checkbox"> — enforced elsewhere by ltx-director-ui.test.js. A raw
  // checkbox here broke that test, correctly.
  const index = indexHtml.indexOf('id="vidH3TurboLowVram"');
  assert.ok(index > 0, 'control not found');
  const tagStart = indexHtml.lastIndexOf('<', index);
  const tag = indexHtml.slice(tagStart, indexHtml.indexOf('>', index) + 1);
  assert.match(tag, /<button\b/, 'must be a button, not an input');
  assert.match(tag, /role="switch"/, 'must declare role="switch"');
  assert.match(tag, /aria-checked="(?:true|false)"/, 'must declare aria-checked');
  assert.doesNotMatch(tag, /type="checkbox"/, 'must not be a raw checkbox');
});

test('app.js drives the switch via aria-checked', () => {
  assert.match(
    appJs, /vidH3TurboLowVram'\)[\s\S]{0,400}aria-checked/,
    'the switch state must be reflected through aria-checked'
  );
});

test('the frontend SENDS h3TurboLowVram (otherwise the flag is dead config)', () => {
  assert.match(
    appJs, /h3TurboLowVram/,
    'app.js must send h3TurboLowVram in the generate payload'
  );
});

test('the control is read from the checkbox, not hardcoded', () => {
  const index = appJs.indexOf('h3TurboLowVram');
  assert.ok(index > 0, 'h3TurboLowVram not found in app.js');
  const context = appJs.slice(index - 120, index + 200);
  assert.ok(
    /vidH3TurboLowVram/.test(context),
    `h3TurboLowVram must derive from the #vidH3TurboLowVram control; got: ${context.replace(/\s+/g, ' ').slice(0, 200)}`
  );
});

test('the low-VRAM state persists with the other H3 turbo form state', () => {
  // Form state is saved per profile; a control the user must re-tick every time
  // is a control they will forget, and then wonder why generation OOMs.
  assert.match(
    appJs, /h3TurboLowVram/,
    'the flag must participate in saved form state'
  );
});
