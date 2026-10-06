"use strict";
// Shared, dependency-free policy; exercised by Node tests and the browser viewer.
(() => {
  const SOURCE_WIDTH = 2592, SOURCE_HEIGHT = 1944;
  function fps(value) {
    const rate = Number(value);
    if (!Number.isFinite(rate) || rate < 0.2 || rate > 30) {
      throw new Error("Choose a processing FPS cap between 0.2 and 30.");
    }
    return rate;
  }
  function sourceSize(width, height) {
    if (![width, height].every(n => Number.isInteger(n) && n > 0)) {
      throw new Error("Source image dimensions are unavailable.");
    }
    const scale = Math.min(1, SOURCE_WIDTH / width, SOURCE_HEIGHT / height);
    return { width: Math.max(1, Math.floor(width * scale)), height: Math.max(1, Math.floor(height * scale)) };
  }
  function letterbox(width, height, targetWidth = 320, targetHeight = 240) {
    const scale = Math.min(targetWidth / width, targetHeight / height);
    const w = width * scale, h = height * scale;
    return { x: (targetWidth - w) / 2, y: (targetHeight - h) / 2, width: w, height: h };
  }
  function delay(rate, elapsedMs) {
    // No catch-up batches: the next tick always samples the current source frame.
    return Math.max(0, 1000 / fps(rate) - elapsedMs);
  }
  const policy = { SOURCE_WIDTH, SOURCE_HEIGHT, fps, sourceSize, letterbox, delay };
  if (typeof module !== "undefined" && module.exports) module.exports = policy;
  else globalThis.DTRFramePolicy = policy;
})();
