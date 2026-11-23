// Run after js/classifier.js. Coordinates use the photo model's 28 x 28 sky grid.
function propose(mask) {
  const boxes = [];
  for (const [width, height] of [[20, 16], [12, 10], [8, 6]]) {
    let best;
    let bestQuality = -1;
    for (let y = 0; y <= 28 - height; y += 2) {
      for (let x = 0; x <= 28 - width; x += 2) {
        let sum = 0;
        for (let j = y; j < y + height; j++) {
          for (let i = x; i < x + width; i++) sum += mask[j * 28 + i];
        }
        const area = width * height;
        const sky = sum / area;
        if (sky < 0.55) continue;
        const overlap = boxes.reduce((max, [left, top, right, bottom]) => {
          const w = Math.max(0, Math.min(x + width, right) - Math.max(x, left));
          const h = Math.max(0, Math.min(y + height, bottom) - Math.max(y, top));
          return Math.max(max, w * h / area);
        }, 0);
        const quality = sky - 0.15 * overlap;
        if (quality > bestQuality) {
          bestQuality = quality;
          best = [x, y, x + width, y + height];
        }
      }
    }
    if (best) boxes.push(best);
  }
  return boxes;
}

async function measureRegions(image) {
  const [sky, clouds, settings] = await loadModels("photo");
  const width = image.naturalWidth;
  const height = image.naturalHeight;
  async function score(box) {
    const [x, y, right, bottom] = box.map((v, i) =>
      Math.floor(v * (i % 2 ? height : width) / 28));
    const input = tensor(image, x, y, Math.max(1, right - x), Math.max(1, bottom - y));
    const mask = (await sky.run({input})).sky;
    const result = await clouds.run({pixel_values: input, sky_mask: mask});
    const [other, sc, background] = result.scores.data;
    const abstain = background > Math.max(sc, other) ||
      (!skyRegion(mask.data) && sc < settings.override);
    return {value: abstain ? 0 : sc, mask};
  }
  const full = await score([0, 0, 28, 28]);
  const boxes = propose(full.mask.data);
  const values = [];
  for (const box of boxes) values.push((await score(box)).value);
  const top = Math.max(0, ...values);
  return {full: full.value, multi: Math.max(full.value, (full.value + top) / 2), boxes, values};
}
