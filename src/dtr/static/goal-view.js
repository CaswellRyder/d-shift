"use strict";
// Presentation only: never changes model acceptance or authorizes a target.
(() => {
  function describe(row) {
    const evidence = row.goal_evidence;
    const shapeKnown = evidence && ["circle", "square", "triangle"].includes(evidence.shape)
      && Number.isFinite(evidence.shape_score) && evidence.shape_score >= (evidence.threshold ?? 0.8);
    const current = row.tracking_valid !== false &&
      (row.classification_age_ms == null || (Number.isFinite(row.classification_age_ms)
        && row.classification_age_ms >= 0 && row.classification_age_ms <= 1000));
    const tentative = Boolean(!row.accepted && !row.suppressed && current && shapeKnown
      && evidence.status === "shape_only" && evidence.color === "unknown");
    const status = row.suppressed ? "Duplicate suppressed" : row.accepted ? "Accepted" :
      tentative ? "Tentative — color unknown" : row.label === "background" ? "Background" : "Below threshold";
    const overlay = row.accepted ? `#${row.track_id} ${row.label} ${(row.score * 100).toFixed(0)}%` :
      tentative ? `#${row.track_id} ${evidence.shape} — color ?` : `#${row.track_id}`;
    const shape = !evidence ? "N/A" : shapeKnown ?
      `${evidence.shape} ${(evidence.shape_score * 100).toFixed(1)}%` : "Uncertain";
    const color = !evidence ? "N/A" : evidence.color === "unknown" ? "Unknown" : evidence.color;
    return { tentative, status, overlay, shape, color,
      stroke: row.accepted ? "#17db60" : tentative ? "#c084fc" : "#b7c0ce",
      priority: row.accepted ? 2 : tentative ? 1 : 0 };
  }
  if (typeof module !== "undefined" && module.exports) module.exports = { describe };
  else globalThis.DTRGoalView = { describe };
})();
