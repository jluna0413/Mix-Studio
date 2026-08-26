'use strict';

/*
 * Structured JSON logging + request-context propagation for Mix Studio.
 *
 * Goals (from the observability audit):
 *   - Every log line is a single JSON object with: ts, level, requestId,
 *     profileId, route, plus arbitrary context.
 *   - A requestId is generated (or accepted from a Correlation-Id header)
 *     at the edge and propagates through async job completion, SSE events,
 *     and into the generated response so a failure can be traced end-to-end
 *     (frontend POST -> server -> ComfyUI prompt_id -> /history -> download
 *     -> SSE).
 *   - The logger replaces ad-hoc console.log/console.warn/console.error
 *     call-sites with structured, queryable output written to logs/app.log
 *     (JSON lines) while still echoing a one-line human summary to stdout.
 */

const fs = require('fs');
const path = require('path');
const { AsyncLocalStorage } = require('async_hooks');

const LOG_DIR = path.join(__dirname, '..', 'logs');
if (!fs.existsSync(LOG_DIR)) {
  try { fs.mkdirSync(LOG_DIR, { recursive: true }); } catch { /* best effort */ }
}

const APP_LOG = path.join(LOG_DIR, 'app.log');
const METRICS_LOG = path.join(LOG_DIR, 'metrics.log');

const LEVELS = { debug: 10, info: 20, warn: 30, error: 40 };

// Active request context for the current async call stack. Captures
// requestId / profileId / route at the HTTP edge so any log emitted deeper
// in the call tree (enhancement, graph build, queue) inherits it.
const requestContext = new AsyncLocalStorage();

function generateRequestId() {
  return 'req-' + Date.now().toString(36) + '-' + Math.random().toString(36).slice(2, 9);
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function ctx() {
  return requestContext.getStore() || {};
}

function currentRequestId() {
  return ctx().requestId || '';
}

/*
 * Run `fn` with the given request context active. Used by the HTTP handler
 * to bind requestId/profileId/route to everything executed during the
 * request, including trackJob (which copies requestId onto the job so it
 * survives into async WS completion / SSE).
 */
function withContext(opts, fn) {
  const prev = ctx();
  const merged = Object.assign({}, prev, opts);
  return requestContext.run(merged, fn);
}

function writeLine(file, obj) {
  try {
    fs.appendFileSync(file, JSON.stringify(obj) + '\n');
  } catch { /* never block the request path on a log write failure */ }
}

function emit(level, message, context) {
  const c = ctx();
  const entry = {
    ts: new Date().toISOString(),
    level,
    requestId: (context && context.requestId) || c.requestId || '',
    profileId: (context && context.profileId) || c.profileId || '',
    route: (context && context.route) || c.route || '',
  };
  if (message !== undefined) entry.message = message;
  // Merge additional structured context last so caller keys win.
  Object.assign(entry, context || {});
  delete entry.requestId; // already hoisted above
  delete entry.profileId;
  delete entry.route;
  entry.requestId = (context && context.requestId) || c.requestId || '';
  entry.profileId = (context && context.profileId) || c.profileId || '';
  entry.route = (context && context.route) || c.route || '';
  if (level === 'error' && (context && context.error)) {
    const e = context.error;
    entry.error = { name: e && e.name, message: e && e.message, code: e && e.code, stack: e && e.stack };
  }
  writeLine(APP_LOG, entry);
  // Human-readable stdout mirror (one line).
  const who = [entry.requestId, entry.profileId, entry.route].filter(Boolean).join(' | ');
  const tag = `[${level.toUpperCase()}]${who ? ' ' + who : ''}`;
  if (level === 'error') console.error(tag, message || (context && context.error && context.error.message) || '');
  else if (level === 'warn') console.warn(tag, message || '');
  else console.log(tag, message || '');
  return entry;
}

const log = {
  debug: (message, context) => emit('debug', message, context),
  info: (message, context) => emit('info', message, context),
  warn: (message, context) => emit('warn', message, context),
  error: (message, context) => emit('error', message, context),

  // Attach a child logger pre-bound with extra static context (e.g. {engine:'qwen', jobId}).
  child(extra) {
    const base = Object.assign({}, ctx(), extra);
    return {
      debug: (m, c) => emit('debug', m, Object.assign({}, base, c)),
      info: (m, c) => emit('info', m, Object.assign({}, base, c)),
      warn: (m, c) => emit('warn', m, Object.assign({}, base, c)),
      error: (m, c) => emit('error', m, Object.assign({}, base, c)),
    };
  },

  // Record a metrics datapoint as a JSON line in metrics.log.
  metric(name, value, tags) {
    const c = ctx();
    const entry = {
      ts: new Date().toISOString(),
      name,
      value,
      requestId: (tags && tags.requestId) || c.requestId || '',
      profileId: (tags && tags.profileId) || c.profileId || '',
      route: (tags && tags.route) || c.route || '',
    };
    Object.assign(entry, tags || {});
    writeLine(METRICS_LOG, entry);
    return entry;
  },

  generateRequestId,
  sleep,
  withContext,
  requestContext,
  currentRequestId,
  LEVELS,
};

module.exports = log;
