'use strict';

/**
 * Guards that the H3 VRAM preflight is actually WIRED INTO the animate route.
 *
 * h3-vram-preflight.test.js proves the estimator and guard behave correctly.
 * It cannot prove server.js calls them — and an unwired guard protects nobody.
 * That failure mode is exactly what happened with `h3R2V`, which was correct
 * code that no caller ever reached (see test/h3-offload-callsites.test.js).
 */

const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');

const serverSource = fs.readFileSync(path.join(__dirname, '..', 'server.js'), 'utf8');

test('server.js imports h3Preflight from generation-capabilities', () => {
  assert.match(
    serverSource, /h3Preflight/,
    'server.js must reference h3Preflight'
  );
  // It must come from the shared capabilities module, not be re-implemented.
  const importBlock = serverSource.slice(
    serverSource.indexOf('require(\'./lib/generation-capabilities\')') - 900,
    serverSource.indexOf('require(\'./lib/generation-capabilities\')') + 60
  );
  assert.match(
    importBlock, /h3Preflight/,
    'h3Preflight must be imported from ./lib/generation-capabilities'
  );
});

test('the animate route calls h3Preflight for the h3 engine', () => {
  assert.match(
    serverSource, /engine === 'h3'[\s\S]{0,400}h3Preflight\(/,
    'the h3 branch must invoke h3Preflight'
  );
});

test('a failed preflight returns 409 with an actionable code', () => {
  assert.match(
    serverSource, /code: 'h3_insufficient_vram'/,
    'a blocked H3 job must carry a machine-readable code'
  );
  // Find the guard and confirm it short-circuits with 409 rather than warning.
  const guardIndex = serverSource.indexOf("code: 'h3_insufficient_vram'");
  assert.ok(guardIndex > 0, 'guard not found');
  const context = serverSource.slice(guardIndex - 400, guardIndex + 200);
  assert.match(context, /return json\(res, 409/, 'must return HTTP 409, not continue');
  assert.match(context, /!h3Check\.ok/, 'must branch on the preflight result');
});

test('the preflight passes the real request geometry, not placeholders', () => {
  const guardIndex = serverSource.indexOf('h3Preflight({');
  assert.ok(guardIndex > 0, 'h3Preflight call not found');
  const call = serverSource.slice(guardIndex, guardIndex + 400);

  for (const field of ['width', 'height', 'frames', 'profile']) {
    // Accept both `frames: frames` and ES6 shorthand `frames,`.
    assert.match(
      call, new RegExp(`${field}\\s*[:,]`),
      `preflight must receive ${field}`
    );
  }
  // Mode matters: reference mode loads an extra DiT and the audio VAE.
  assert.match(call, /mode:/, 'preflight must receive the H3 mode');
  assert.match(
    call, /profile: generationHardwareProfile/,
    'preflight must use the live hardware profile, not a literal'
  );
});

test('the H3 guard sits alongside the LTX guard (same pattern)', () => {
  // Both preflights should live in the same pre-submit region so future engines
  // follow one convention rather than scattering checks.
  const ltxIndex = serverSource.indexOf('ltxRefinePreflight({');
  const h3Index = serverSource.indexOf('h3Preflight({');
  assert.ok(ltxIndex > 0 && h3Index > 0, 'both guards must exist');
  assert.ok(
    Math.abs(h3Index - ltxIndex) < 4000,
    'the H3 guard should sit near the LTX guard in the same pre-submit block'
  );
});
