/**
 * Stratocumulus Cloud Classifier
 * Client-side inference using ONNX Runtime Web
 */

// Global variables
let session = null;
let stream = null;
let animationId = null;
let isInferencing = false;
let frameCount = 0;

// ImageNet normalization values (must match Python training)
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

// Run inference every N frames (30 frames = ~1 second at 30fps)
const FRAME_SKIP = 30;

/**
 * Load the ONNX model when page loads
 */
async function loadModel() {
  try {
    session = await ort.InferenceSession.create(
      "code/scai/stratocumulus_model.onnx",
    );
    console.log("Model loaded successfully");
  } catch (error) {
    console.error("Error loading model:", error);
  }
}

/**
 * Preprocess image to match PyTorch training pipeline
 */
function preprocessImage(img) {
  const canvas = document.createElement("canvas");
  canvas.width = 224;
  canvas.height = 224;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0, 224, 224);
  const imageData = ctx.getImageData(0, 0, 224, 224);
  const data = new Float32Array(1 * 3 * 224 * 224);

  // Convert RGBA to CHW format with normalization
  for (let i = 0; i < 224 * 224; i++) {
    data[i] = (imageData.data[i * 4] / 255.0 - MEAN[0]) / STD[0];
    data[i + 224 * 224] =
      (imageData.data[i * 4 + 1] / 255.0 - MEAN[1]) / STD[1];
    data[i + 2 * 224 * 224] =
      (imageData.data[i * 4 + 2] / 255.0 - MEAN[2]) / STD[2];
  }
  return data;
}

/**
 * Run inference on preprocessed image
 */
async function runInference(imageData) {
  const tensor = new ort.Tensor("float32", imageData, [1, 3, 224, 224]);
  const output = await session.run({ input: tensor });
  const logit = output.output.data[0];
  const probability = 1 / (1 + Math.exp(-logit));
  return probability;
}

/**
 * Handle file upload and display results
 */
async function handleImage(file) {
  document.getElementById("loading").style.display = "block";
  document.getElementById("result").style.display = "none";

  const img = document.getElementById("imagePreview");
  img.src = URL.createObjectURL(file);
  img.style.display = "block";

  await new Promise((resolve) => (img.onload = resolve));

  try {
    const processedImage = preprocessImage(img);
    const probability = await runInference(processedImage);

    const resultDiv = document.getElementById("result");
    const isStratocumulus = probability > 0.5;

    resultDiv.className = isStratocumulus ? "green" : "red";
    resultDiv.textContent = isStratocumulus
      ? `Stratocumulus (${(probability * 100).toFixed(1)}%)`
      : `Not Stratocumulus (${((1 - probability) * 100).toFixed(1)}%)`;

    resultDiv.style.display = "block";
  } catch (error) {
    console.error("Inference error:", error);
    alert("Error analyzing image. Check console for details.");
  } finally {
    document.getElementById("loading").style.display = "none";
  }
}

/**
 * Start webcam and begin live inference
 */
async function startWebcam() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" }, // Use back camera on mobile
    });

    const video = document.getElementById("videoPreview");
    video.srcObject = stream;
    video.style.display = "block";

    // Hide upload elements
    document.getElementById("imagePreview").style.display = "none";
    document.getElementById("result").style.display = "none";
    document.getElementById("loading").style.display = "none";

    // Show stop button, hide start button
    document.getElementById("webcamButton").style.display = "none";
    document.getElementById("stopWebcamButton").style.display = "inline-block";

    // Start the inference loop
    processVideoFrame();
  } catch (error) {
    console.error("Error accessing webcam:", error);
    alert(
      "Error accessing webcam. Please ensure you have granted camera permissions.",
    );
  }
}

/**
 * Stop webcam and cleanup
 */
function stopWebcam() {
  // Stop all video tracks
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }

  // Cancel animation frame
  if (animationId) {
    cancelAnimationFrame(animationId);
    animationId = null;
  }

  // Hide video and result
  const video = document.getElementById("videoPreview");
  video.style.display = "none";
  video.srcObject = null;

  document.getElementById("liveResult").style.display = "none";

  // Show start button, hide stop button
  document.getElementById("webcamButton").style.display = "inline-block";
  document.getElementById("stopWebcamButton").style.display = "none";

  frameCount = 0;
  isInferencing = false;
}

/**
 * Process video frames and run inference every N frames
 */
async function processVideoFrame() {
  const video = document.getElementById("videoPreview");

  // Schedule next frame
  animationId = requestAnimationFrame(processVideoFrame);

  // Skip frames to throttle inference
  frameCount++;
  if (frameCount % FRAME_SKIP !== 0 || isInferencing) {
    return;
  }

  // Run inference
  isInferencing = true;

  try {
    const processedImage = preprocessImage(video);
    const probability = await runInference(processedImage);

    const resultDiv = document.getElementById("liveResult");
    const isStratocumulus = probability > 0.5;

    resultDiv.className = isStratocumulus ? "green" : "red";
    resultDiv.textContent = isStratocumulus
      ? `Stratocumulus (${(probability * 100).toFixed(1)}%)`
      : `Not Stratocumulus (${((1 - probability) * 100).toFixed(1)}%)`;

    resultDiv.style.display = "block";
  } catch (error) {
    console.error("Video inference error:", error);
  } finally {
    isInferencing = false;
  }
}

// Event listeners
document.getElementById("uploadButton").addEventListener("click", () => {
  document.getElementById("fileInput").click();
});

document.getElementById("fileInput").addEventListener("change", (e) => {
  if (e.target.files.length > 0) {
    handleImage(e.target.files[0]);
  }
});

document.getElementById("webcamButton").addEventListener("click", startWebcam);
document
  .getElementById("stopWebcamButton")
  .addEventListener("click", stopWebcam);

// Load model on page load
loadModel();
