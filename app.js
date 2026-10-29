import { FilesetResolver, HandLandmarker, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22";

const STATIC_LETTERS = [..."ABCDEFGHI" , ..."KLMNOPQRSTUVWXY"];
const STORAGE_KEY = "signspeak.samples.v1";
const K_NEIGHBORS = 5;
const MIN_CONFIDENCE = 0.6;
const video = document.querySelector("#video");
const canvas = document.querySelector("#overlay");
const ctx = canvas.getContext("2d");
const stage = document.querySelector("#videoStage");
const cameraButton = document.querySelector("#cameraButton");
const cameraStatus = document.querySelector("#cameraStatus");
const cameraPlaceholder = document.querySelector("#cameraPlaceholder");
const modelStatus = document.querySelector("#modelStatus");
const stageLabel = document.querySelector("#stageLabel");
const prediction = document.querySelector("#prediction");
const confidenceBar = document.querySelector("#confidenceBar");
const confidenceText = document.querySelector("#confidenceText");
const letterGrid = document.querySelector("#letterGrid");
const selectedLetter = document.querySelector("#selectedLetter");
const captureButton = document.querySelector("#captureButton");
const captureStatus = document.querySelector("#captureStatus");
const sampleTotal = document.querySelector("#sampleTotal");
const trainButton = document.querySelector("#trainButton");
const trainingStatus = document.querySelector("#trainingStatus");
const phrase = document.querySelector("#phrase");
const toast = document.querySelector("#toast");

let landmarker;
let drawing;
let stream;
let samples = loadSamples();
let trained = false;
let capturing = false;
let captureTimer;
let captureCount = 0;
let lastVideoTime = -1;
let lastPrediction = "";
let stableSince = 0;
let lastAddedAt = 0;
let phraseText = "";
let toastTimer;

function loadSamples() {
  try {
    const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
    return Object.fromEntries(STATIC_LETTERS.map((letter) => [letter, Array.isArray(parsed[letter]) ? parsed[letter].filter(validFeatures) : []]));
  } catch {
    return Object.fromEntries(STATIC_LETTERS.map((letter) => [letter, []]));
  }
}

function validFeatures(features) {
  return Array.isArray(features) && features.length === 63 && features.every(Number.isFinite);
}

function saveSamples() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(samples));
  renderSamples();
}

function renderSamples() {
  const total = Object.values(samples).reduce((sum, rows) => sum + rows.length, 0);
  sampleTotal.textContent = `${total} ${total === 1 ? "example" : "examples"}`;
  letterGrid.querySelectorAll(".letter").forEach((button) => {
    button.classList.toggle("has-samples", samples[button.dataset.letter].length > 0);
    button.title = `${button.dataset.letter}: ${samples[button.dataset.letter].length} captured examples`;
  });
  updateTrainingState();
}

function updateTrainingState() {
  const populated = STATIC_LETTERS.filter((letter) => samples[letter].length >= 3).length;
  trainButton.disabled = populated !== STATIC_LETTERS.length;
  if (trained) {
    trainingStatus.textContent = "Recognizer trained · ready to practice";
    trainingStatus.style.color = "var(--lime)";
  } else {
    trainingStatus.textContent = `${populated}/24 letters have at least 3 examples`;
    trainingStatus.style.color = "";
  }
}

function setSelectedLetter(letter) {
  selectedLetter.value = letter;
  letterGrid.querySelectorAll(".letter").forEach((button) => button.classList.toggle("selected", button.dataset.letter === letter));
}

function initializeLetterPicker() {
  for (const letter of [...STATIC_LETTERS.slice(0, 9), "J", ...STATIC_LETTERS.slice(9), "Z"]) {
    const option = document.createElement("option");
    option.value = letter;
    option.textContent = "JZ".includes(letter) ? `${letter} — motion-based, unsupported` : letter;
    option.disabled = "JZ".includes(letter);
    selectedLetter.append(option);
    const button = document.createElement("button");
    button.type = "button";
    button.className = `letter${"JZ".includes(letter) ? " motion" : ""}`;
    button.textContent = letter;
    button.dataset.letter = letter;
    button.disabled = "JZ".includes(letter);
    button.setAttribute("aria-label", "JZ".includes(letter) ? `${letter}, motion-based and unsupported` : `Select letter ${letter}`);
    button.addEventListener("click", () => setSelectedLetter(letter));
    letterGrid.append(button);
  }
  setSelectedLetter(STATIC_LETTERS[0]);
  selectedLetter.addEventListener("change", () => setSelectedLetter(selectedLetter.value));
}

function normalizeLandmarks(points) {
  const wrist = points[0];
  let scale = 0;
  for (const point of points) scale = Math.max(scale, Math.hypot(point.x - wrist.x, point.y - wrist.y));
  if (!Number.isFinite(scale) || scale < 1e-5) return null;
  const features = [];
  for (const point of points) features.push((point.x - wrist.x) / scale, (point.y - wrist.y) / scale, (point.z - wrist.z) / scale);
  return features;
}

function classify(features) {
  if (!trained || !features) return null;
  const nearest = [];
  for (const letter of STATIC_LETTERS) {
    for (const sample of samples[letter]) {
      let distance = 0;
      for (let i = 0; i < 63; i++) {
        const delta = features[i] - sample[i];
        distance += delta * delta;
      }
      nearest.push({ letter, distance });
    }
  }
  nearest.sort((a, b) => a.distance - b.distance);
  const top = nearest.slice(0, K_NEIGHBORS);
  const votes = new Map();
  for (const match of top) votes.set(match.letter, (votes.get(match.letter) || 0) + 1);
  const [letter, count] = [...votes.entries()].sort((a, b) => b[1] - a[1])[0] || [];
  if (!letter) return null;
  return { letter, confidence: count / top.length, distance: Math.sqrt(top[0].distance / 63) };
}

function setPrediction(result) {
  if (!result || result.confidence < MIN_CONFIDENCE || result.distance > 0.72) {
    prediction.textContent = "—";
    confidenceText.textContent = result ? "Uncertain" : "Waiting";
    confidenceBar.style.width = "0%";
    lastPrediction = "";
    return;
  }
  prediction.textContent = result.letter;
  confidenceText.textContent = `${Math.round(result.confidence * 100)}%`;
  confidenceBar.style.width = `${Math.round(result.confidence * 100)}%`;
  const now = performance.now();
  if (lastPrediction !== result.letter) {
    lastPrediction = result.letter;
    stableSince = now;
    return;
  }
  if (now - stableSince >= 800 && now - lastAddedAt >= 1100) {
    appendLetter(result.letter);
    lastAddedAt = now;
    stableSince = now;
  }
}

function renderPhrase() {
  phrase.textContent = "";
  if (!phraseText) {
    const placeholder = document.createElement("span");
    placeholder.className = "phrase-placeholder";
    placeholder.textContent = "Your letters will show up here";
    phrase.append(placeholder);
  } else phrase.textContent = phraseText;
  for (const id of ["spaceButton", "backspaceButton", "clearPhraseButton"]) document.getElementById(id).disabled = !phraseText;
}

function appendLetter(letter) {
  phraseText += letter;
  renderPhrase();
}

function showToast(message) {
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("show"), 2400);
}

async function startCamera() {
  cameraButton.disabled = true;
  cameraButton.querySelector("span:first-child").textContent = "Loading model…";
  modelStatus.textContent = "Loading hand tracking model…";
  try {
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/wasm");
    landmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", delegate: "GPU" },
      runningMode: "VIDEO",
      numHands: 1,
      minHandDetectionConfidence: 0.55,
      minHandPresenceConfidence: 0.55,
      minTrackingConfidence: 0.55,
    });
    drawing = new DrawingUtils(ctx);
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 960 }, height: { ideal: 720 } }, audio: false });
    video.srcObject = stream;
    await video.play();
    stage.classList.add("active");
    cameraPlaceholder.hidden = true;
    cameraStatus.classList.add("ready");
    cameraStatus.innerHTML = "<i></i> Camera on";
    stageLabel.textContent = "LIVE · LOOK AT THE CAMERA";
    modelStatus.textContent = "Camera and hand tracking are ready.";
    cameraButton.querySelector("span:first-child").textContent = "Stop camera";
    cameraButton.disabled = false;
    captureButton.disabled = false;
    requestAnimationFrame(processFrame);
  } catch (error) {
    console.error(error);
    stopCamera();
    cameraStatus.classList.add("error");
    cameraStatus.innerHTML = "<i></i> Camera unavailable";
    modelStatus.textContent = error.name === "NotAllowedError" ? "Allow camera access in your browser, then try again." : "Could not start. Use localhost or HTTPS and check your connection.";
    showToast(error.name === "NotAllowedError" ? "Camera access was blocked." : "Could not load camera or hand model.");
  } finally {
    cameraButton.disabled = false;
  }
}

function stopCamera() {
  if (capturing) stopCapture();
  stream?.getTracks().forEach((track) => track.stop());
  stream = null;
  if (landmarker) landmarker.close();
  landmarker = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  stage.classList.remove("active");
  cameraPlaceholder.hidden = false;
  cameraStatus.classList.remove("ready");
  cameraStatus.innerHTML = "<i></i> Camera off";
  stageLabel.textContent = "WAITING FOR CAMERA";
  modelStatus.textContent = "Hand model loads when you start";
  cameraButton.querySelector("span:first-child").textContent = "Start camera";
  captureButton.disabled = true;
  prediction.textContent = "—";
  confidenceText.textContent = "Waiting";
  confidenceBar.style.width = "0%";
  lastVideoTime = -1;
}

function processFrame() {
  if (!landmarker || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
    if (landmarker) requestAnimationFrame(processFrame);
    return;
  }
  if (video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const result = landmarker.detectForVideo(video, performance.now());
    const points = result.landmarks?.[0];
    if (points) {
      drawing.drawConnectors(points, HandLandmarker.HAND_CONNECTIONS, { color: "#c3f36b", lineWidth: 3 });
      drawing.drawLandmarks(points, { color: "#f2f5ef", lineWidth: 1.5, radius: 3 });
      const features = normalizeLandmarks(points);
      if (capturing && features) captureExample(features);
      setPrediction(classify(features));
      stageLabel.textContent = "HAND DETECTED · HOLD A STATIC SIGN";
    } else {
      setPrediction(null);
      stageLabel.textContent = "LIVE · SHOW ONE HAND TO CAMERA";
    }
  }
  requestAnimationFrame(processFrame);
}

function beginCapture() {
  if (!stream) return;
  if ("JZ".includes(selectedLetter.value)) return;
  capturing = true;
  captureCount = 0;
  captureButton.textContent = "Stop capture";
  captureButton.classList.add("capturing");
  captureStatus.textContent = `Hold ${selectedLetter.value} in view; collecting varied frames…`;
  captureTimer = setTimeout(stopCapture, 7000);
}

function captureExample(features) {
  const rows = samples[selectedLetter.value];
  const last = rows[rows.length - 1];
  if (last) {
    let distance = 0;
    for (let i = 0; i < 63; i++) distance += (features[i] - last[i]) ** 2;
    if (Math.sqrt(distance / 63) < 0.018 && captureCount > 0) return;
  }
  rows.push(features);
  if (rows.length > 120) rows.shift();
  captureCount++;
  sampleTotal.textContent = `${Object.values(samples).reduce((sum, values) => sum + values.length, 0) + 1} examples`;
  captureStatus.textContent = `Captured ${captureCount} this round · ${rows.length} total for ${selectedLetter.value}`;
}

function stopCapture() {
  if (!capturing) return;
  capturing = false;
  clearTimeout(captureTimer);
  captureButton.textContent = "Capture examples";
  captureButton.classList.remove("capturing");
  if (captureCount) {
    saveSamples();
    trained = false;
    captureStatus.textContent = `Saved ${captureCount} new examples for ${selectedLetter.value}.`;
  } else captureStatus.textContent = "No hand detected. Center your hand in the frame and try again.";
  updateTrainingState();
}

function trainRecognizer() {
  if (STATIC_LETTERS.some((letter) => samples[letter].length < 3)) return;
  trained = true;
  trainingStatus.textContent = "Recognizer trained · ready to practice";
  trainingStatus.style.color = "var(--lime)";
  showToast("Recognizer trained with your examples.");
}

function exportSamples() {
  const blob = new Blob([JSON.stringify({ format: "signspeak-samples-v1", letters: samples }, null, 2)], { type: "application/json" });
  const anchor = document.createElement("a");
  anchor.href = URL.createObjectURL(blob);
  anchor.download = "signspeak-calibration.json";
  anchor.click();
  URL.revokeObjectURL(anchor.href);
}

async function importSamples(event) {
  const file = event.target.files?.[0];
  if (!file) return;
  try {
    const parsed = JSON.parse(await file.text());
    const incoming = parsed.letters || parsed;
    const next = Object.fromEntries(STATIC_LETTERS.map((letter) => [letter, Array.isArray(incoming[letter]) ? incoming[letter].filter(validFeatures).slice(-120) : []]));
    samples = next;
    trained = false;
    saveSamples();
    showToast("Calibration examples imported.");
  } catch {
    showToast("That calibration file could not be read.");
  } finally {
    event.target.value = "";
  }
}

function clearSamples() {
  if (!confirm("Delete all saved calibration examples from this browser?")) return;
  samples = Object.fromEntries(STATIC_LETTERS.map((letter) => [letter, []]));
  trained = false;
  saveSamples();
  captureStatus.textContent = "All local calibration examples cleared.";
  showToast("Calibration data cleared.");
}

initializeLetterPicker();
renderSamples();
renderPhrase();
cameraButton.addEventListener("click", () => stream ? stopCamera() : startCamera());
captureButton.addEventListener("click", () => capturing ? stopCapture() : beginCapture());
trainButton.addEventListener("click", trainRecognizer);
document.querySelector("#exportButton").addEventListener("click", exportSamples);
document.querySelector("#importInput").addEventListener("change", importSamples);
document.querySelector("#clearButton").addEventListener("click", clearSamples);
document.querySelector("#spaceButton").addEventListener("click", () => { phraseText += " "; renderPhrase(); });
document.querySelector("#backspaceButton").addEventListener("click", () => { phraseText = phraseText.slice(0, -1); renderPhrase(); });
document.querySelector("#clearPhraseButton").addEventListener("click", () => { phraseText = ""; renderPhrase(); });
