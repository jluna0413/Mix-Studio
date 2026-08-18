'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const root = path.join(__dirname, '..');
const app = fs.readFileSync(path.join(root, 'public', 'app.js'), 'utf8');
const server = fs.readFileSync(path.join(root, 'server.js'), 'utf8');
const { resolveScratchDiskDir } = require('../lib/comfy-restart');
const { normalizeH3References } = require('../lib/video-workflows');

test('/api/animate is wrapped in try/catch for clean JSON error responses', () => {
  assert.match(server, /if \(route === '\/api\/animate' && req\.method === 'POST'\) \{\s*try \{/);
  assert.match(server, /\} catch \(error\) \{\s*const status = error\.statusCode/);
  assert.match(server, /code: error\.code \|\| 'animate_failed'/);
});

test('offloadDir and scratchDiskDir settings default to fast NVMe directory', () => {
  assert.match(server, /offloadDir:\s*'E:\\\\tmp\\\\pytorch'/);
  assert.match(server, /scratchDiskDir:\s*'E:\\\\tmp\\\\pytorch'/);
  assert.equal(resolveScratchDiskDir({ settings: { offloadDir: 'E:\\custom\\tmp' } }), 'E:\\custom\\tmp');
  assert.equal(resolveScratchDiskDir({}), 'E:\\tmp\\pytorch');
});

test('smart VRAM unloading triggers when queue is idle', () => {
  assert.match(server, /function notifyQueueIdle\(delayMs = 3000\)/);
  assert.match(server, /console\.log\('\[queue\]', 'Queue is idle\. Triggering ComfyUI model unload to free VRAM\.'\)/);
  assert.match(server, /comfyFetch\(freeReq\.path, freeReq\.init\)/);
});

test('song and audio references work across video engines and map to H3 references', () => {
  assert.match(app, /\$\('#vidAudioChip'\)\.hidden = false;/);
  assert.match(app, /audioTitle\.textContent = hasAudio \? 'Audio added' : 'Audio';/);
  assert.match(app, /let vidAudioName;[\s\S]*?if \(state\.vidAudio\) \{[\s\S]*?ensureAudioUploaded\(state\.vidAudio\)/);

  const refs = normalizeH3References({ audios: [] }, 'track.mp3');
  assert.equal(refs.audios.length, 1);
  assert.equal(refs.audios[0].name, 'track.mp3');
});
