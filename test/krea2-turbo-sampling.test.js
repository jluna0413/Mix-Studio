'use strict';

// Regression coverage for upstream issue #82: switching from Qwen Image 2.1 back
// to Krea 2 Turbo must not leave Qwen's 40-step / CFG 2.5 sampling on the Turbo
// model. The route clamps these server-side so no client can bypass them.

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const serverJs = fs.readFileSync(path.join(__dirname, '..', 'server.js'), 'utf8');

function normalize(params) {
  const clampInt = (v, min, max, dflt) => {
    const n = Math.floor(Number(v));
    return Number.isFinite(n) ? Math.min(max, Math.max(min, n)) : dflt;
  };
  const clampNum = (v, min, max, dflt) => {
    const n = Number(v);
    return Number.isFinite(n) ? Math.min(max, Math.max(min, n)) : dflt;
  };
  const start = serverJs.indexOf('p.krea2Turbo = p.mode === \'edit\' ? true : p.krea2Turbo !== false;');
  assert.ok(start > -1, 'generate normalization block present');
  const end = serverJs.indexOf('p.krea2RawTurboLora =', start);
  const body = serverJs.slice(start, end)
    .replace(/p\.krea2RawTurboLora[\s\S]*$/,'');
  const context = vm.createContext({ clampInt, clampNum, p: Object.assign({}, params) });
  vm.runInContext(`${body}\nglobalThis.out = { steps: p.steps, cfg: p.cfg };`, context);
  return JSON.parse(JSON.stringify(context.out));
}

test('Krea 2 Turbo text-to-image pins CFG 1 and caps steps at the Turbo envelope', () => {
  assert.deepEqual(normalize({ mode: 't2i', imageEngine: 'krea2', krea2Turbo: true, steps: 40, cfg: 2.5 }), { steps: 16, cfg: 1 });
  assert.deepEqual(normalize({ mode: 't2i', imageEngine: 'krea2', krea2Turbo: true, steps: 99, cfg: 5 }), { steps: 16, cfg: 1 });
});

test('Krea 2 Turbo keeps an explicit in-envelope step count and still forces CFG 1', () => {
  assert.deepEqual(normalize({ mode: 't2i', imageEngine: 'krea2', krea2Turbo: true, steps: 10, cfg: 1 }), { steps: 10, cfg: 1 });
});

test('Qwen Image 2.1 and Raw Krea 2 are unaffected by the Turbo pin', () => {
  assert.deepEqual(normalize({ mode: 't2i', imageEngine: 'qwen21', krea2Turbo: false, steps: 40, cfg: 2.5 }), { steps: 40, cfg: 2.5 });
  assert.deepEqual(normalize({ mode: 't2i', imageEngine: 'krea2', krea2Turbo: false, steps: 12, cfg: 1 }), { steps: 12, cfg: 1 });
});
