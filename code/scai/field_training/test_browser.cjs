const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { resolve } = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

function browser() {
  const pixels = new Uint8ClampedArray(224 * 224 * 4);
  const drawing = { drawImage() {}, getImageData() { return { data: pixels }; } };
  const context = vm.createContext({
    document: {
      createElement() { return { getContext() { return drawing; } }; },
      getElementById() { return {}; },
    },
    ort: { Tensor: class {} },
  });
  const path = resolve(__dirname, "../../../js/classifier.js");
  vm.runInContext(readFileSync(path, "utf8"), context);
  return context;
}

test("sky window avoids a foreground building", () => {
  const { skyRegion } = browser();
  const sky = new Float32Array(28 * 28);
  assert.equal(skyRegion(sky), null);
  sky.fill(1, 0, 28 * 12);
  const [left, top, right, bottom] = skyRegion(sky);
  assert.ok(right > left && bottom > top);
  assert.ok((Math.min(bottom, 12) - top) / (bottom - top) >= 0.8);
  sky.fill(1);
  assert.deepEqual(Array.from(skyRegion(sky)), [0, 0, 28, 28]);
});

test("a background score asks for a different view", async () => {
  const context = browser();
  context.loadModels = async () => [
    { run: async () => ({ sky: { data: new Float32Array(784).fill(1) } }) },
    { run: async () => ({ scores: { data: [0.05, 0.30, 0.65] } }) },
    { threshold: 0.25, override: 0.6 },
  ];
  assert.match(await context.classify({ width: 224, height: 224 }), /Aim at/);
});
