const { test } = require('node:test');
const assert = require('node:assert/strict');
const policy = require('../src/dtr/static/frame-policy.js');

test('large 4:3 input is downscaled to exactly 2592 x 1944', () => {
  assert.deepEqual(policy.sourceSize(5184, 3888), { width: 2592, height: 1944 });
  assert.deepEqual(policy.sourceSize(2592, 1944), { width: 2592, height: 1944 });
});
test('low-resolution, wide and portrait sources retain geometry and never upscale', () => {
  assert.deepEqual(policy.sourceSize(640, 480), { width: 640, height: 480 });
  assert.deepEqual(policy.sourceSize(3840, 2160), { width: 2592, height: 1458 });
  assert.deepEqual(policy.sourceSize(3000, 4000), { width: 1458, height: 1944 });
  assert.deepEqual(policy.letterbox(2592, 1944), { x: 0, y: 0, width: 320, height: 240 });
  assert.deepEqual(policy.letterbox(1920, 1080), { x: 0, y: 30, width: 320, height: 180 });
  assert.deepEqual(policy.letterbox(640, 640), { x: 40, y: 0, width: 240, height: 240 });
});
test('pacing spends only remaining budget, never catches up or queues frames', () => {
  assert.equal(policy.delay(5, 70), 130);
  assert.equal(policy.delay(5, 350), 0);
  assert.equal(policy.delay(2, 70), 430);
  assert.equal(policy.delay(0.2, 50), 4950);
});
test('invalid rates and dimensions fail explicitly', () => {
  for (const value of ['', 0, -1, 31, Infinity, NaN, 'abc']) assert.throws(() => policy.fps(value));
  for (const size of [[0, 240], [320, -1], [NaN, 240], [320.5, 240]]) {
    assert.throws(() => policy.sourceSize(...size));
  }
});
