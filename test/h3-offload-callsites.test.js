'use strict';

/**
 * Guards the getOffloadConfig CALL SITES in server.js.
 *
 * The unit tests in h3-offload-escalation.test.js prove the function honours
 * `{ h3R2V: true }`. They cannot prove server.js actually PASSES it — and that
 * was the entire bug: correct function, wrong callers, for every caller.
 *
 * These tests read server.js as text and assert the call sites are wired
 * correctly. Text assertions are brittle by nature, so they check the
 * load-bearing semantics (an options object reaches the H3 call; no bare
 * vramFreeGb is passed positionally) rather than exact formatting.
 */

const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');

const serverSource = fs.readFileSync(path.join(__dirname, '..', 'server.js'), 'utf8');

/** Extract each getOffloadConfig(...) call with balanced-paren scanning. */
function extractCalls(source) {
  const calls = [];
  const needle = 'getOffloadConfig(';
  let index = source.indexOf(needle);
  while (index !== -1) {
    let depth = 0;
    let end = index + needle.length - 1;
    for (let i = index + needle.length - 1; i < source.length; i += 1) {
      const ch = source[i];
      if (ch === '(') depth += 1;
      else if (ch === ')') {
        depth -= 1;
        if (depth === 0) { end = i; break; }
      }
    }
    calls.push(source.slice(index, end + 1));
    index = source.indexOf(needle, end);
  }
  return calls;
}

test('every getOffloadConfig call site was found', () => {
  const calls = extractCalls(serverSource);
  assert.ok(calls.length >= 4, `expected at least 4 call sites, found ${calls.length}`);
});

test('no call site passes vramFreeGb in the options position (THE BUG)', () => {
  // p.vramFreeGb is a NUMBER. Passing it third silently disabled the H3
  // escalation because a number has no .h3R2V property.
  for (const call of extractCalls(serverSource)) {
    assert.ok(
      !/vramFreeGb\s*\)/.test(call),
      `getOffloadConfig must not receive vramFreeGb as its options argument: ${call.replace(/\s+/g, ' ')}`
    );
  }
});

test('the H3 video path passes an options object with h3R2V', () => {
  const calls = extractCalls(serverSource);
  const withH3 = calls.filter((call) => /h3R2V/.test(call));
  assert.strictEqual(
    withH3.length, 1,
    'exactly one call site (the video/animate path) should signal h3R2V'
  );
  // It must be derived from the engine, not hardcoded true.
  assert.match(
    withH3[0].replace(/\s+/g, ' '),
    /h3R2V:\s*engine === 'h3'/,
    'h3R2V must be derived from the engine so non-H3 engines are unaffected'
  );
});

test('h3R2V is passed inside braces, i.e. an actual object', () => {
  const call = extractCalls(serverSource).find((c) => /h3R2V/.test(c));
  assert.ok(call, 'no h3R2V call site found');
  assert.match(
    call.replace(/\s+/g, ' '),
    /\{\s*h3R2V:/,
    'h3R2V must be a property of an object literal, not a positional argument'
  );
});
