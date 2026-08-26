'use strict';

const fs = require('fs');
const path = require('path');

const STORAGE_THRESHOLDS = { WARNING: 0.80, CRITICAL: 0.90, EMERGENCY: 0.95 };
const CLEANUP_POLICIES = { ARCHIVE_AFTER_DAYS: 30, MAX_ACTIVE_ITEMS: 100 };

class StorageManager {
  constructor(dataDir, comfyOutputDir) {
    // NEVER hardcode a data directory. This install is portable
    // (install.json -> dataDir), so the repo's ./data is a stale
    // pre-migration copy, not the live gallery. Resolve dataDir from
    // RUNTIME.dataDir (see lib/runtime-config.js) and pass it in.
    if (!dataDir) throw new Error('StorageManager requires dataDir (use RUNTIME.dataDir)');
    this.dataDir = dataDir;
    this.comfyOutputDir = comfyOutputDir || path.join(this.dataDir, 'comfy-output');
    this.imagesDir = path.join(this.dataDir, 'images');
    this.videosDir = path.join(this.dataDir, 'videos');
    this.archiveDir = path.join(this.dataDir, 'archive');
  }

  ensureDirectories() {
    const dirs = [this.dataDir, this.imagesDir, this.videosDir, this.archiveDir];
    for (const dir of dirs) {
      if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    }
  }

  // ... (rest of the class remains unchanged)
}

module.exports = { StorageManager, STORAGE_THRESHOLDS, CLEANUP_POLICIES };