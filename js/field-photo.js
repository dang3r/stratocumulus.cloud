// Opt-in photo classifier for mixed scenes. Calibration is specific to this model.
if (new URLSearchParams(location.search).get("model") === "cloud-gaze") {
  const switchLink = document.getElementById("modelSwitch");
  switchLink.href = location.pathname;
  switchLink.textContent = "Mixed-scene model active · Switch to the standard model";
  document.getElementById("sensitivityControl").hidden = false;
  const highSensitivity = document.getElementById("highSensitivity");
  // The largest hundredth retaining eight of nine positives in the 2019 building panel.
  const buildingThreshold = 0.83;
  let headSession;
  globalThis.classifyFieldScene = async function (image) {
    headSession ??= ort.InferenceSession.create(modelDirectory + "photo/gaze-head.onnx")
      .catch(error => { headSession = undefined; throw error; });
    const [[skyModel, cloudModel], head] = await Promise.all([loadModels("photo"), headSession]);
    const width = image.naturalWidth || image.width;
    const height = image.naturalHeight || image.height;
    async function encode(x, y, w, h) {
      const input = tensor(image, x, y, w, h);
      const {sky: mask} = await skyModel.run({input});
      const {embedding} = await cloudModel.run({pixel_values: input, sky_mask: mask}, ["embedding"]);
      return {mask, embedding};
    }
    const {mask, embedding: full} = await encode(0, 0, width, height);
    const fraction = mask.data.reduce((sum, v) => sum + v, 0) / mask.data.length;
    const box = skyRegion(mask.data);
    let patch = full;
    if (box && (box[2] - box[0] < grid || box[3] - box[1] < grid)) {
      const [x, y, right, bottom] = box.map((v, i) => Math.floor(v * (i % 2 ? height : width) / grid));
      patch = (await encode(x, y, Math.max(1, right - x), Math.max(1, bottom - y))).embedding;
    }
    const sky = new ort.Tensor("float32", Float32Array.of(fraction), [1, 1]);
    const {margin} = await head.run({full, patch, sky});
    return {positive: margin.data[0] >= (highSensitivity.checked ? 0 : buildingThreshold), margin: margin.data[0]};
  };
}
