const { test } = require('node:test');
const assert = require('node:assert/strict');
const { describe } = require('../src/dtr/static/goal-view.js');
const row = { track_id: 7, accepted: false, label: 'orange_square', score: 0.475,
  goal_evidence: { shape: 'square', shape_score: 0.95, threshold: 0.8, color: 'unknown', status: 'shape_only' } };

test('shape-only overlay is tentative and never rewrites model acceptance or color', () => {
  const before = JSON.stringify(row), info = describe(row);
  assert.equal(info.tentative, true);
  assert.equal(info.overlay, '#7 square — color ?');
  assert.equal(info.color, 'Unknown');
  assert.equal(info.shape, 'square 95.0%');
  assert.equal(info.status, 'Tentative — color unknown');
  assert.equal(JSON.stringify(row), before);
});
test('suppression, stale/lost tracks and insufficient evidence cannot become tentative', () => {
  for (const extra of [{suppressed:true}, {tracking_valid:false}, {classification_age_ms:1001},
    {classification_age_ms:NaN}, {goal_evidence:{...row.goal_evidence,shape_score:0.79}},
    {goal_evidence:{...row.goal_evidence,shape:'unknown'}}, {goal_evidence:null}]) {
    assert.equal(describe({...row,...extra}).tentative, false);
  }
});
test('accepted and balloon results preserve their existing labels', () => {
  const accepted = describe({...row,accepted:true,score:0.95});
  assert.equal(accepted.overlay, '#7 orange_square 95%');
  assert.equal(accepted.tentative, false);
  assert.equal(describe({...row,goal_evidence:undefined}).shape, 'N/A');
  assert.equal(describe({...row,suppressed:true}).status, 'Duplicate suppressed');
});
