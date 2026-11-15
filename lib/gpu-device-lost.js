'use strict';

/**
 * GPU device-lost detection for ComfyUI errors.
 *
 * When ComfyUI's level_zero backend (Intel Arc) loses the GPU device
 * mid-computation, it raises RuntimeError: "level_zero backend failed with
 * error: 20 (UR_RESULT_ERROR_DEVICE_LOST)".  This module provides helpers to
 * detect that condition from various error representations (WebSocket
 * execution_error messages, ComfyUI /history status objects, raw strings).
 */

const GPU_DEVICE_LOST_PATTERNS = [
  /UR_RESULT_ERROR_DEVICE_LOST/i,
  /level_zero backend failed with error:\s*20/i,
  /DEVICE_LOST/i,
  /device lost/i,
  /gpu device lost/i,
  /device has been lost/i,
];

/**
 * Returns true when the given text indicates a GPU device-lost condition.
 *
 * @param {string} text - Error message or serialized error object.
 * @returns {boolean}
 */
function isGpuDeviceLostError(text) {
  if (!text) return false;
  const str = String(text);
  return GPU_DEVICE_LOST_PATTERNS.some((pattern) => pattern.test(str));
}

/**
 * Extracts a human-readable error message from a ComfyUI /history status object.
 *
 * ComfyUI's history status shape:
 *   { status_str: 'error', completed: false, messages: [...],
 *     error: { exception_type, exception_message, traceback, ... } }
 *
 * @param {object} histStatus - The `status` field from a ComfyUI history entry.
 * @returns {string} A readable error message, or '' if none found.
 */
function extractComfyErrorMessage(histStatus) {
  if (!histStatus || typeof histStatus !== 'object') return '';
  const err = histStatus.error;
  if (err && typeof err === 'object') {
    if (typeof err.exception_message === 'string' && err.exception_message.trim()) {
      return err.exception_message.trim();
    }
    if (typeof err.message === 'string' && err.message.trim()) {
      return err.message.trim();
    }
    if (typeof err.exception_type === 'string' && err.exception_type.trim()) {
      return err.exception_type.trim();
    }
  }
  if (typeof err === 'string' && err.trim()) return err.trim();
  if (typeof histStatus.exception_message === 'string' && histStatus.exception_message.trim()) {
    return histStatus.exception_message.trim();
  }
  if (Array.isArray(histStatus.messages)) {
    for (const entry of histStatus.messages) {
      if (Array.isArray(entry) && entry.length >= 2) {
        const msg = entry[1];
        if (msg && typeof msg === 'object' && typeof msg.exception_message === 'string') {
          return msg.exception_message.trim();
        }
        if (typeof msg === 'string' && msg.trim()) return msg.trim();
      }
    }
  }
  return '';
}

/**
 * Returns a user-facing message for a GPU device-lost error.
 *
 * @returns {string}
 */
function gpuDeviceLostUserMessage() {
  return 'GPU device lost — ComfyUI crashed. Please restart ComfyUI and try again.';
}

module.exports = {
  isGpuDeviceLostError,
  extractComfyErrorMessage,
  gpuDeviceLostUserMessage,
  GPU_DEVICE_LOST_PATTERNS,
};
