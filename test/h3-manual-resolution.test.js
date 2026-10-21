'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');

const H3Resolution = require('../public/h3-resolution.js');

test('a manually entered H3 canvas snaps onto the 32 pixel latent grid', () => {
  const dimensions = H3Resolution.manualDimensions(383, 575);
  assert.deepEqual(dimensions, { width: 384, height: 576 });
});

test('a manually entered H3 canvas keeps an already aligned size', () => {
  const dimensions = H3Resolution.manualDimensions(768, 1344);
  assert.deepEqual(dimensions, { width: 768, height: 1344 });
});

test('a manual H3 canvas below the usable minimum is rejected', () => {
  assert.equal(H3Resolution.manualDimensions(32, 576), null);
  assert.equal(H3Resolution.manualDimensions(384, 32), null);
  assert.equal(H3Resolution.manualDimensions(0, 0), null);
  assert.equal(H3Resolution.manualDimensions('', ''), null);
  assert.equal(H3Resolution.manualDimensions('abc', 'def'), null);
  assert.equal(H3Resolution.manualDimensions(undefined, undefined), null);
});

test('a manual H3 canvas never collapses below the latent grid', () => {
  const dimensions = H3Resolution.manualDimensions(1, 1);
  assert.equal(dimensions, null, 'anything under 64px is rejected outright');
  const rounded = H3Resolution.manualDimensions(64, 64);
  assert.deepEqual(rounded, { width: 64, height: 64 });
});

test('the manual marker is exposed for the UI to branch on', () => {
  assert.equal(H3Resolution.MANUAL_SIZE, 'manual');
});

test('the tier dimensions keep working so a manual canvas is an opt-in', () => {
  const tier = H3Resolution.dimensions(9 / 16, 1, 1);
  assert.ok(tier.width > 0 && tier.height > 0);
  assert.equal(tier.width % H3Resolution.MULTIPLE, 0);
  assert.equal(tier.height % H3Resolution.MULTIPLE, 0);
});
