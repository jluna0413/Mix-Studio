'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const server = fs.readFileSync(path.join(__dirname, '..', 'server.js'), 'utf8');

test('ComfyUI object metadata gets a shared long timeout and a distinct error', () => {
  assert.match(server, /const COMFY_OBJECT_INFO_TIMEOUT_MS = 60_000/);
  assert.match(server, /AbortSignal\.timeout\(COMFY_OBJECT_INFO_TIMEOUT_MS\)/);
  assert.match(server, /wrapped\.code = 'comfy_object_info_timeout'/);
  assert.doesNotMatch(server, /getObjectInfo\([^\n]*AbortSignal\.timeout\((?:4000|6000)\)/);
});

test('transient ComfyUI reload failures are retried with backoff before erroring', () => {
  // After a restart, /object_info can refuse connections or time out while
  // custom nodes load. The route must retry instead of returning ok:false,
  // so the frontend polls to "installed" rather than showing a hard error.
  assert.match(server, /async function getObjectInfoWithRetry\(/);
   assert.match(server, /for \(let attempt = 0; attempt <= retries; attempt \+= 1\)/);
   assert.match(server, /await sleep\(backoffMs \* \(attempt \+ 1\)\)/);
  assert.match(server, /isComfyConnectionError/);
  assert.match(server, /comfy_object_info_unavailable/);
});

test('connection-refused during reload is treated as retryable', () => {
  // Mirrors the post-restart window: ComfyUI not yet accepting sockets.
  assert.match(server, /ECONNREFUSED|ENOTFOUND|ECONNRESET/);
  assert.match(server, /socket.*hang|network/i);
});
