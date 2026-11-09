import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import vm from 'node:vm';

const manual = await readFile(new URL('./manual.js', import.meta.url), 'utf8');
const {ManualStream} = await import('data:text/javascript;base64,' + Buffer.from(manual).toString('base64'));
const source = (await readFile(new URL('./app.js', import.meta.url), 'utf8')).replace(/^import .*;\n/, '');
const flush = () => new Promise(resolve => setImmediate(resolve));

test('a service generation change clears held input before using the new token', async () => {
  const nodes = new Map();
  function element() {
    return {listeners: {}, children: [], checked: false, disabled: false,
      classList: {remove() {}, toggle() {}},
      addEventListener(name, fn) {this.listeners[name] = fn;},
      append(...children) {this.children.push(...children);},
      replaceChildren(...children) {this.children = children;},
      elements: {namedItem: id => get('form-' + id)}};
  }
  function get(id) {if (!nodes.has(id)) nodes.set(id, element()); return nodes.get(id);}
  const motions = ['forward', 'back', 'left', 'right', 'up', 'down'].map(direction => {
    const node = element(); node.dataset = {direction}; return node;
  });
  let generation = '100';
  const requests = [];
  const context = vm.createContext({ManualStream, Date,
    document: {getElementById: get, querySelectorAll: () => motions,
      createElement: element, addEventListener() {}},
    window: {addEventListener() {}}, setTimeout() {}, setInterval() {},
    fetch: async (path, options) => {
      requests.push([path, options]);
      const body = path === '/api/state'
        ? {input_generation: generation, connected: true, age_ms: 0,
          input_inhibited: false, pause_requested: false, estop_requested: false}
        : path === '/api/captures' ? {configured: false} : {receipt: {status: 'QUEUED_BENCH'}};
      return {ok: true, json: async () => body};
    }});
  vm.runInContext(source + '\nglobalThis.poll = pollState; globalThis.inspect = () => ({enabled,current,axes:stream.axes,inputGeneration});', context);
  await flush();
  get('enable').checked = true;
  get('enable').listeners.change();
  vm.runInContext("hold('forward')", context);
  await flush();
  assert.equal(context.inspect().enabled, true);
  assert.equal(context.inspect().axes.forward, 0.2);
  await context.poll();
  assert.equal(context.inspect().enabled, true, 'ordinary polls must preserve a current hold');
  generation = '200';
  await context.poll();
  assert.equal(context.inspect().inputGeneration, '200');
  assert.equal(context.inspect().enabled, false);
  assert.equal(context.inspect().current, null);
  assert.equal(context.inspect().axes, null);
  assert.equal(get('enable').checked, false);
  assert.ok(motions.every(node => node.disabled));
  const manualCount = requests.filter(([path]) => path === '/api/manual').length;
  await vm.runInContext('stream.tick()', context);
  assert.equal(requests.filter(([path]) => path === '/api/manual').length, manualCount);
});
