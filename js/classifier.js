const modelDirectory = "code/scai/browser/";
const mean = [0.485, 0.456, 0.406];
const std = [0.229, 0.224, 0.225];
const size = 224;
const grid = 28;
const canvas = document.createElement("canvas");
canvas.width = canvas.height = size;
const ctx = canvas.getContext("2d", { willReadFrequently: true });
ctx.imageSmoothingQuality = "high";
const byId = (id) => document.getElementById(id);
const aim = "Too little usable sky. Aim at a wider patch of sky.";
let activeStream;
let timer;
let generation = 0;
const sessions = new Map();

function loadModels(mode) {
  if (!sessions.has(mode)) {
    const folder = modelDirectory + mode + "/";
    sessions.set(mode, Promise.all([
      ort.InferenceSession.create(folder + "sky.onnx"),
      ort.InferenceSession.create(folder + "clouds.onnx"),
      fetch(folder + "settings.json").then(response => {
        if (!response.ok) throw Error("Missing model settings");
        return response.json();
      }),
    ]).catch(error => { sessions.delete(mode); throw error; }));
  }
  return sessions.get(mode);
}

function tensor(image, x, y, w, h) {
  ctx.drawImage(image, x, y, w, h, 0, 0, size, size);
  const rgba = ctx.getImageData(0, 0, size, size).data;
  const data = new Float32Array(size * size * 3);
  for (let i = 0; i < size * size; i++) {
    for (let c = 0; c < 3; c++) {
      data[c * size * size + i] = (rgba[4 * i + c] / 255 - mean[c]) / std[c];
    }
  }
  return new ort.Tensor("float32", data, [1, 3, size, size]);
}

// Pick the largest supported rectangle containing at least 80% predicted sky.
function skyRegion(mask) {
  const stride = grid + 1;
  const sums = new Int32Array(stride * stride);
  for (let y = 0; y < grid; y++) {
    for (let x = 0; x < grid; x++) {
      const i = (y + 1) * stride + x + 1;
      sums[i] = Number(mask[y * grid + x] >= 0.5) +
        sums[i - 1] + sums[i - stride] - sums[i - stride - 1];
    }
  }
  if (sums[sums.length - 1] >= 0.65 * grid * grid) return [0, 0, grid, grid];
  for (const [w, h] of [[28,20], [24,16], [20,14], [16,12], [12,10], [10,8]]) {
    let largest = -1, box;
    for (let y = 0; y <= grid - h; y++) {
      for (let x = 0; x <= grid - w; x++) {
        const i = y * stride + x;
        const count = sums[i + h * stride + w] - sums[i + h * stride] -
          sums[i + w] + sums[i];
        if (count > largest) {
          largest = count;
          box = [x, y, x + w, y + h];
        }
      }
    }
    if (largest >= 0.8 * w * h) return box;
  }
  return null;
}

async function liveScores(image, w, h, input, box, clouds, head) {
  const full = (await clouds.run({input})).embedding.data;
  let detail = full;
  if (box && (box[2] - box[0] !== grid || box[3] - box[1] !== grid)) {
    const [x, y, right, bottom] = box.map((v, i) => Math.floor(v * (i % 2 ? h : w) / grid));
    const patch = tensor(image, x, y, Math.max(1, right - x), Math.max(1, bottom - y));
    detail = (await clouds.run({input: patch})).embedding.data;
  }
  const logits = head.weights.map((weights, c) => {
    let sum = head.bias[c];
    for (let i = 0; i < full.length; i++) {
      sum += weights[i] * full[i] + weights[i + full.length] * detail[i];
    }
    return sum;
  });
  const largest = Math.max(...logits);
  const exp = logits.map(v => Math.exp(v - largest));
  const sc = exp[1] / (exp[0] + exp[1] + exp[2]);
  return [sc, logits[2] > Math.max(logits[0], logits[1])];
}

async function classify(image, mode = "photo") {
  if (globalThis.classifyFieldScene) {
    const result = await globalThis.classifyFieldScene(image);
    return result.positive ? "Possible stratocumulus" : "No strong stratocumulus signal in this view.";
  }
  const [sky, clouds, configuration] = await loadModels(mode);
  let settings = configuration;
  const w = image.videoWidth || image.naturalWidth || image.width;
  const h = image.videoHeight || image.naturalHeight || image.height;
  const input = tensor(image, 0, 0, w, h);
  const mask = (await sky.run({input})).sky;
  const box = skyRegion(mask.data);
  let sc, background;
  if (mode === "video") {
    const skyPixels = mask.data.reduce((count, value) => count + Number(value >= 0.5), 0);
    if (skyPixels < configuration.obstructedBelow * mask.data.length) {
      settings = configuration.obstructed;
    }
    [sc, background] = await liveScores(image, w, h, input, box, clouds, settings);
  } else {
    const result = await clouds.run({pixel_values: input, sky_mask: mask});
    const [other, score, noSky] = result.scores.data;
    sc = score;
    background = noSky > Math.max(other, score);
  }
  if (background || (!box && sc < settings.override)) return aim;
  if (settings === configuration.obstructed && sc < settings.threshold) return aim;
  return sc >= settings.threshold ? "Possible stratocumulus" : "No strong stratocumulus signal in this view.";
}

function show(id, text) {
  const element = byId(id);
  element.textContent = text;
  element.className = text === "Possible stratocumulus" ? "green" : "";
  element.style.display = "block";
}

function stopWebcam() {
  generation++;
  clearTimeout(timer);
  activeStream?.getTracks().forEach(track => track.stop());
  activeStream = null;
  byId("videoPreview").srcObject = null;
  for (const id of ["videoPreview", "stopWebcamButton", "liveResult", "loading"]) {
    byId(id).style.display = "none";
  }
  byId("webcamButton").style.display = "inline-block";
}

async function handleImage(file) {
  stopWebcam();
  const current = generation;
  const img = byId("imagePreview");
  const url = URL.createObjectURL(file);
  byId("result").style.display = "none";
  byId("loading").style.display = "block";
  try {
    img.src = url;
    await img.decode();
    if (current !== generation) return;
    img.style.display = "block";
    const text = await classify(img);
    if (current === generation) show("result", text);
  } catch {
    if (current === generation) show("result", "Could not analyze this image. Try again.");
  } finally {
    URL.revokeObjectURL(url);
    if (current === generation) byId("loading").style.display = "none";
  }
}

async function startWebcam() {
  stopWebcam();
  const current = generation;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({video: {facingMode: "environment"}});
    if (current !== generation) {
      stream.getTracks().forEach(track => track.stop());
      return;
    }
    activeStream = stream;
    const video = byId("videoPreview");
    video.srcObject = stream;
    await video.play();
    if (current !== generation) return;
    for (const id of ["imagePreview", "result", "webcamButton"]) byId(id).style.display = "none";
    video.style.display = "block";
    byId("stopWebcamButton").style.display = "inline-block";
    const frameCanvas = document.createElement("canvas");
    const frameContext = frameCanvas.getContext("2d");
    let previous;
    let displayed;
    async function frame() {
      if (current !== generation) return;
      try {
        // Both the sky mask and crop must come from the same camera frame.
        const scale = Math.min(1, 640 / Math.max(video.videoWidth, video.videoHeight));
        const width = Math.max(1, Math.round(video.videoWidth * scale));
        const height = Math.max(1, Math.round(video.videoHeight * scale));
        if (frameCanvas.width !== width || frameCanvas.height !== height) {
          frameCanvas.width = width;
          frameCanvas.height = height;
          frameContext.imageSmoothingQuality = "high";
        }
        frameContext.drawImage(video, 0, 0, width, height);
        const text = await classify(frameCanvas, "video");
        if (current === generation) {
          if (text === aim || text === previous) displayed = text;
          if (displayed && displayed !== text) displayed = undefined;
          show("liveResult", displayed || "Checking the sky…");
          previous = text;
        }
      } catch {
        if (current === generation) show("liveResult", "Could not analyze this frame. Retrying…");
      }
      if (current === generation) timer = setTimeout(frame, 800);
    }
    frame();
  } catch {
    if (current === generation) {
      stopWebcam();
      show("result", "Could not start the camera. Check camera permissions or upload an image.");
    }
  }
}

byId("uploadButton").onclick = () => byId("fileInput").click();
byId("fileInput").onchange = event => {
  const file = event.target.files[0];
  event.target.value = "";
  if (file) handleImage(file);
};
byId("webcamButton").onclick = startWebcam;
byId("stopWebcamButton").onclick = stopWebcam;
