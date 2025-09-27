import { InferenceSession, Tensor } from "onnxruntime-web";

const opfsSupported =
  "storage" in navigator && "getDirectory" in navigator.storage;
const modelInfoPath =
  "https://models.hydrui.dev/wd-v1-4-vit-tagger-v2/info.json";
let modelPath = "";
let modelTagPath = "";

const imageInput = getElementByIdOrDie("imageInput", HTMLInputElement);
const uploadArea = getElementByIdOrDie("uploadArea", HTMLDivElement);
const uploadButton = getElementByIdOrDie("uploadButton", HTMLButtonElement);
const processButton = getElementByIdOrDie("processButton", HTMLButtonElement);
const imagePreview = getElementByIdOrDie("imagePreview", HTMLImageElement);
const thresholdInput = getElementByIdOrDie("threshold", HTMLInputElement);
const loadingDiv = getElementByIdOrDie("loading", HTMLDivElement);
const errorDiv = getElementByIdOrDie("error", HTMLDivElement);
const resultsDiv = getElementByIdOrDie("results", HTMLDivElement);
const ratingDiv = getElementByIdOrDie("ratingResult", HTMLDivElement);
const tagsDiv = getElementByIdOrDie("tagsResult", HTMLDivElement);

let modelSession: InferenceSession | null = null;
let tags: Record<string, string>[] = [];
let currentImage: HTMLImageElement | null = null;
let modelInfo: ModelInfo;

function parseCSV(text: string): Record<string, string>[] {
  const lines = text.trim().split("\n");
  const headers = lines[0].split(",");
  const data: Record<string, string>[] = [];
  for (let i = 1; i < lines.length; i++) {
    const values = lines[i].split(",");
    const row: Record<string, string> = {};
    headers.forEach((header, index) => {
      row[header] = values[index];
    });
    data.push(row);
  }
  return data;
}

function relativePath(urlString: URL | string, path: string): URL {
  return new URL(path, new URL(urlString, document.baseURI));
}

async function init() {
  try {
    const infoResponse = await fetch(modelInfoPath);
    modelInfo = await infoResponse.json();
    modelPath = relativePath(
      modelInfoPath,
      modelInfo.modelfile.replace(".onnx", ".ort"),
    ).toString();
    modelTagPath = relativePath(modelInfoPath, modelInfo.tagsfile).toString();
    const tagsText = await loadTagsWithCache();
    tags = parseCSV(tagsText);
    imageInput.addEventListener("change", handleImageSelect);
    uploadButton.addEventListener("click", () => {
      imageInput.click();
    });
    processButton.addEventListener("click", processImage);
    uploadArea.addEventListener("dragover", (e) => {
      e.preventDefault();
      uploadArea.classList.add("dragover");
    });
    uploadArea.addEventListener("dragleave", () => {
      uploadArea.classList.remove("dragover");
    });
    uploadArea.addEventListener("drop", (e) => {
      e.preventDefault();
      uploadArea.classList.remove("dragover");
      const files = e.dataTransfer?.files;
      if (files && files.length > 0) {
        handleImageFile(files[0]);
      }
    });
  } catch (error) {
    showError("Failed to initialize application: " + error.message);
  }
}

function handleImageSelect() {
  const file = imageInput.files?.[0];
  if (file) {
    handleImageFile(file);
  }
}

function handleImageFile(file: File) {
  if (!file.type.startsWith("image/")) {
    showError("Please select a valid image file.");
    return;
  }
  const reader = new FileReader();
  reader.onload = (e) => {
    if (!e.target || typeof e.target.result !== "string") {
      return;
    }
    imagePreview.src = e.target.result;
    imagePreview.style.display = "block";
    currentImage = new Image();
    currentImage.onload = () => {
      processButton.disabled = false;
    };
    currentImage.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

async function loadModel() {
  if (modelSession) {
    return modelSession;
  }
  try {
    console.log("Loading ORT model...");
    modelSession = await InferenceSession.create(await loadModelWithCache());
    console.log("Model loaded successfully");
    return modelSession;
  } catch (error) {
    throw new Error("Failed to load model: " + error.message);
  }
}

function preprocessImage(image: HTMLImageElement, targetSize = 448) {
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    throw new Error("Unable to get 2D Canvas context.");
  }
  canvas.width = image.width;
  canvas.height = image.height;
  ctx.fillStyle = "white";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(image, 0, 0);
  let imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const oldHeight = canvas.height;
  const oldWidth = canvas.width;
  const desiredSize = Math.max(oldWidth, oldHeight, targetSize);
  const deltaW = desiredSize - oldWidth;
  const deltaH = desiredSize - oldHeight;
  const top = Math.floor(deltaH / 2);
  const left = Math.floor(deltaW / 2);
  const squareCanvas = document.createElement("canvas");
  const squareCtx = squareCanvas.getContext("2d");
  if (!squareCtx) {
    throw new Error("Unable to get 2D Canvas context");
  }
  squareCanvas.width = desiredSize;
  squareCanvas.height = desiredSize;
  squareCtx.fillStyle = "white";
  squareCtx.fillRect(0, 0, desiredSize, desiredSize);
  squareCtx.drawImage(canvas, left, top);
  const finalCanvas = document.createElement("canvas");
  const finalCtx = finalCanvas.getContext("2d");
  if (!finalCtx) {
    throw new Error("Unable to get 2D Canvas context");
  }
  finalCanvas.width = targetSize;
  finalCanvas.height = targetSize;
  finalCtx.imageSmoothingEnabled = true;
  finalCtx.imageSmoothingQuality = "high";
  finalCtx.drawImage(squareCanvas, 0, 0, targetSize, targetSize);
  imageData = finalCtx.getImageData(0, 0, targetSize, targetSize);
  const data = imageData.data;
  const tensor = new Float32Array(1 * targetSize * targetSize * 3);
  for (let y = 0; y < targetSize; y++) {
    for (let x = 0; x < targetSize; x++) {
      const pixelIndex = (y * targetSize + x) * 4;
      const tensorBaseIndex = (y * targetSize + x) * 3;
      tensor[tensorBaseIndex] = data[pixelIndex + 2];
      tensor[tensorBaseIndex + 1] = data[pixelIndex + 1];
      tensor[tensorBaseIndex + 2] = data[pixelIndex];
    }
  }
  return new Tensor("float32", tensor, [1, targetSize, targetSize, 3]);
}

async function processImage() {
  if (!currentImage) {
    showError("Please select an image first.");
    return;
  }
  try {
    showLoading(true);
    hideResults();
    hideError();
    console.log("Loading model...");
    const session = await loadModel();
    console.log("Preprocessing image...");
    const inputTensor = preprocessImage(currentImage);
    console.log("Running inference...");
    const feeds = {};
    feeds[session.inputNames[0]] = inputTensor;
    const results = await session.run(feeds);
    const output = results[session.outputNames[0]];
    const confidences = output.data;
    const threshold = parseFloat(thresholdInput.value);
    if (!(confidences instanceof Float32Array)) {
      throw new Error(`Expected Float32Array for tensor output!`);
    }
    const { ratings, tagResults } = processResults(confidences, threshold);
    displayResults(ratings, tagResults);
  } catch (error) {
    console.error("Processing error:", error);
    showError("Failed to process image: " + error.message);
  } finally {
    showLoading(false);
  }
}

function processResults(confidences: Float32Array, threshold: number) {
  const ratings: Record<string, number> = {};
  const tagResults: { name: string; confidence: number }[] = [];
  const numRatings = modelInfo.numberofratings;
  for (let i = 0; i < numRatings && i < tags.length; i++) {
    const tag = tags[i];
    ratings[tag.name] = confidences[i];
  }
  for (let i = numRatings; i < tags.length; i++) {
    const tag = tags[i];
    const confidence = confidences[i];
    if (confidence > threshold) {
      tagResults.push({
        name: tag.name,
        confidence: confidence,
      });
    }
  }
  tagResults.sort((a, b) => b.confidence - a.confidence);
  return { ratings, tagResults };
}

function displayResults(
  ratings: Record<string, number>,
  tagResults: { name: string; confidence: number }[],
) {
  let topRating = "general";
  let topRatingScore = 0;
  for (const [name, score] of Object.entries(ratings)) {
    if (score > topRatingScore) {
      topRating = name;
      topRatingScore = score;
    }
  }
  ratingDiv.innerHTML = `
    <strong>Content Rating:</strong> ${topRating}
    <span style="color: #666;">(confidence: ${(topRatingScore * 100).toFixed(1)}%)</span>
  `;
  tagsDiv.innerHTML = "<h4>Tags:</h4>";
  if (tagResults.length > 0) {
    tagResults.forEach((tag) => {
      const tagElement = document.createElement("span");
      tagElement.className = "tag";
      tagElement.textContent = `${tag.name} (${(tag.confidence * 100).toFixed(1)}%)`;
      tagsDiv.appendChild(tagElement);
    });
  } else {
    tagsDiv.innerHTML += "<p>No tags found above the threshold.</p>";
  }
  resultsDiv.style.display = "block";
}

function showLoading(show: boolean) {
  loadingDiv.style.display = show ? "block" : "none";
  processButton.disabled = show;
}

function hideResults() {
  resultsDiv.style.display = "none";
}

function showError(message: string) {
  errorDiv.textContent = message;
  errorDiv.style.display = "block";
}

function hideError() {
  errorDiv.style.display = "none";
}

function getElementByIdOrDie<T extends typeof HTMLElement>(
  elementId: string,
  cls: T,
): InstanceType<T> {
  const element = document.getElementById(elementId);
  if (!element) {
    throw new Error(`Missing expected element: #${elementId}`);
  }
  if (!(element instanceof cls)) {
    throw new Error(`Element #${elementId} is not of type ${cls.name}`);
  }
  return element as unknown as InstanceType<T>;
}

async function getOPFSRoot() {
  if (!opfsSupported) {
    throw new Error("OPFS not supported");
  }
  return await navigator.storage.getDirectory();
}

async function getCacheKey(url: string): Promise<string> {
  const encoder = new TextEncoder();
  const data = encoder.encode(url);
  const hashBuffer = await crypto.subtle.digest("SHA-256", data);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("")
    .substring(0, 16);
}

async function loadModelWithCache(): Promise<ArrayBuffer> {
  return (
    await loadWithCache(modelPath, "model", "ort", "Model")
  ).arrayBuffer();
}

async function loadTagsWithCache(): Promise<string> {
  return (await loadWithCache(modelTagPath, "tags", "csv", "Tags")).text();
}

async function loadWithCache(
  url: string,
  cachePrefix: string,
  fileExtension: string,
  resourceName: string,
): Promise<Blob> {
  if (!opfsSupported) {
    console.log(`OPFS not supported, fetching ${resourceName} directly`);
    const response = await fetch(url);
    return await response.blob();
  }
  try {
    const root = await getOPFSRoot();
    const cacheKey = await getCacheKey(url);
    const fileName = `${cachePrefix}_${cacheKey}.${fileExtension}`;

    try {
      console.log(`Checking cache for ${resourceName}...`);
      const fileHandle = await root.getFileHandle(fileName);
      const file = await fileHandle.getFile();
      console.log(`${resourceName} loaded from cache`);
      return file;
    } catch {
      console.log(
        `${resourceName} could not be loaded from cache, fetching from network...`,
      );
      const response = await fetch(url);
      const blob = await response.blob();
      try {
        const fileHandle = await root.getFileHandle(fileName, { create: true });
        const writable = await fileHandle.createWritable();
        await writable.write(blob);
        await writable.close();
        console.log(`${resourceName} cached successfully`);
      } catch (writeError) {
        console.warn(`Failed to cache ${resourceName}:`, writeError);
      }
      return blob;
    }
  } catch (error) {
    console.warn(`Error in OPFS cache load:`, error);
    const response = await fetch(url);
    return await response.blob();
  }
}

interface ModelInfo {
  modelname: string;
  source: string;
  modelfile: string;
  tagsfile: string;
  ratingsflag: number;
  numberofratings: number;
}

window.addEventListener("load", init);
