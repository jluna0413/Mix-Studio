'use strict';

const fs = require('fs');
const path = require('path');
const { performance } = require('perf_hooks');

// Ensure log directory exists
const LOG_DIR = path.join(__dirname, '..', 'logs');
if (!fs.existsSync(LOG_DIR)) {
  fs.mkdirSync(LOG_DIR, { recursive: true });
}

const ERROR_LOG = path.join(LOG_DIR, 'errors.log');
const METRICS_LOG = path.join(LOG_DIR, 'metrics.log');

/**
 * Generate a unique request ID for tracing
 */
let requestIdCounter = 0;
function generateRequestId() {
  return 'req-' + (++requestIdCounter) + '-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
}

/**
 * Structured error logger with context
 * @param {Error} error - The error object
 * @param {Object} context - Additional context (requestId, jobId, route, engine, etc.)
 * @returns {Object} - The logged error entry
 */
function error(error, context = {}) {
  const timestamp = new Date().toISOString();
  const requestId = context.requestId || generateRequestId();
  const jobId = context.jobId || '';
  const route = context.route || '';
  const engine = context.engine || '';
  const gpuInfo = context.gpuInfo || {};
  const duration = context.duration || 0;
  
  const errorEntry = {
    timestamp,
    requestId,
    jobId,
    route,
    engine,
    gpuInfo,
    duration,
    error: {
      name: error.name || 'Error',
      message: error.message || error.toString(),
      stack: error.stack || '',
    }
  };
  
  // Write to error log
  fs.appendFileSync(ERROR_LOG, JSON.stringify(errorEntry) + '\n');
  
  // Also console.error for immediate visibility
  console.error(`[ERROR] ${timestamp} | ${requestId} | ${route} | ${engine} | ${jobId}`);
  console.error(`  Message: ${error.message}`);
  if (error.stack) {
    console.error(`  Stack: ${error.stack.split('\n').slice(0, 5).join('\n')}`);
  }
  
  return errorEntry;
}

/**
 * Telemetry metrics logger
 * @param {string} name - Metric name
 * @param {number|string} value - Metric value
 * @param {Object} tags - Additional tags (requestId, route, etc.)
 */
function metric(name, value, tags = {}) {
  const timestamp = new Date().toISOString();
  const requestId = tags.requestId || '';
  
  const metricEntry = {
    timestamp,
    name,
    value,
    requestId,
    ...tags
  };
  
  fs.appendFileSync(METRICS_LOG, JSON.stringify(metricEntry) + '\n');
}

/**
 * Performance timer wrapper
 * @param {string} name - Operation name for metrics
 * @param {Function} fn - Async function to time
 * @returns {Promise} - Result of the function
 */
function timeOperation(name, fn) {
  const start = performance();
  return fn().then(result => {
    const duration = performance() - start;
    metric(name, duration, { requestId: generateRequestId() });
    return result;
  }).catch(err => {
    const duration = performance() - start;
    metric(name, duration, { requestId: generateRequestId() });
    throw err;
  });
}

module.exports = {
  error,
  metric,
  timeOperation,
  generateRequestId
};