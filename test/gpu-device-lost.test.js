'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const fs = require('node:fs');
const path = require('node:path');

const {
  isGpuDeviceLostError,
  extractComfyErrorMessage,
  gpuDeviceLostUserMessage,
} = require('../lib/gpu-device-lost');

const serverJs = fs.readFileSync(path.join(__dirname, '..', 'server.js'), 'utf8');

test('isGpuDeviceLostError detects UR_RESULT_ERROR_DEVICE_LOST', () => {
  assert.equal(isGpuDeviceLostError('level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)'), true);
  assert.equal(isGpuDeviceLostError('RuntimeError: level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)'), true);
});

test('isGpuDeviceLostError detects DEVICE_LOST variants', () => {
  assert.equal(isGpuDeviceLostError('DEVICE_LOST'), true);
  assert.equal(isGpuDeviceLostError('gpu device lost'), true);
  assert.equal(isGpuDeviceLostError('device has been lost'), true);
  assert.equal(isGpuDeviceLostError('device lost'), true);
});

test('isGpuDeviceLostError detects level_zero error 20', () => {
  assert.equal(isGpuDeviceLostError('level_zero backend failed with error: 20'), true);
});

test('isGpuDeviceLostError rejects unrelated errors', () => {
  assert.equal(isGpuDeviceLostError(''), false);
  assert.equal(isGpuDeviceLostError(null), false);
  assert.equal(isGpuDeviceLostError(undefined), false);
  assert.equal(isGpuDeviceLostError('required field missing'), false);
  assert.equal(isGpuDeviceLostError('model not found'), false);
  assert.equal(isGpuDeviceLostError('[Errno 22] Invalid argument'), false);
  assert.equal(isGpuDeviceLostError('execution error'), false);
});

test('extractComfyErrorMessage extracts from error object', () => {
  const status = {
    status_str: 'error',
    error: {
      exception_type: 'RuntimeError',
      exception_message: 'level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)',
    },
  };
  assert.equal(extractComfyErrorMessage(status), 'level_zero backend failed with error: 20 (UR_RESULT_ERROR_DEVICE_LOST)');
});

test('extractComfyErrorMessage extracts from message field', () => {
  const status = {
    status_str: 'error',
    error: { message: 'GPU crashed' },
  };
  assert.equal(extractComfyErrorMessage(status), 'GPU crashed');
});

test('extractComfyErrorMessage falls back to exception_type', () => {
  const status = {
    status_str: 'error',
    error: { exception_type: 'RuntimeError' },
  };
  assert.equal(extractComfyErrorMessage(status), 'RuntimeError');
});

test('extractComfyErrorMessage handles string error', () => {
  const status = { status_str: 'error', error: 'something broke' };
  assert.equal(extractComfyErrorMessage(status), 'something broke');
});

test('extractComfyErrorMessage handles messages array', () => {
  const status = {
    status_str: 'error',
    messages: [
      ['executing', { node: '5', prompt_id: 'abc' }],
      ['execution_error', { exception_message: 'device lost' }],
    ],
  };
  assert.equal(extractComfyErrorMessage(status), 'device lost');
});

test('extractComfyErrorMessage returns empty for null/undefined', () => {
  assert.equal(extractComfyErrorMessage(null), '');
  assert.equal(extractComfyErrorMessage(undefined), '');
  assert.equal(extractComfyErrorMessage({}), '');
});

test('gpuDeviceLostUserMessage returns clear message', () => {
  const msg = gpuDeviceLostUserMessage();
  assert.match(msg, /GPU device lost/);
  assert.match(msg, /ComfyUI crashed/);
  assert.match(msg, /restart ComfyUI/);
});

test('server imports gpu-device-lost helpers', () => {
  assert.match(serverJs, /require\(['"]\.\/lib\/gpu-device-lost['"]\)/);
});

test('server uses isGpuDeviceLostError in execution_error handler', () => {
  assert.match(serverJs, /isGpuDeviceLostError\(/);
});

test('server uses extractComfyErrorMessage in polling fallback', () => {
  assert.match(serverJs, /extractComfyErrorMessage\(/);
});

test('server defines restartComfyOnGpuDeviceLost', () => {
  assert.match(serverJs, /function restartComfyOnGpuDeviceLost/);
});
