const fileInput = document.getElementById("fileInput");
const cameraBtn = document.getElementById("cameraBtn");
const captureBtn = document.getElementById("captureBtn");
const predictBtn = document.getElementById("predictBtn");
const camera = document.getElementById("camera");
const captureCanvas = document.getElementById("captureCanvas");
const preview = document.getElementById("preview");

const resultBox = document.getElementById("result");
const statusText = document.getElementById("statusText");
const plantText = document.getElementById("plantText");
const diseaseText = document.getElementById("diseaseText");
const confidenceText = document.getElementById("confidenceText");
const messageText = document.getElementById("messageText");
const topPredictions = document.getElementById("topPredictions");

let selectedBlob = null;
let mediaStream = null;
let warmupPromise = null;

async function warmupModel() {
  if (warmupPromise) return warmupPromise;

  warmupPromise = fetch("/info")
    .then(async (response) => {
      const body = await response.text();
      try {
        return JSON.parse(body);
      } catch {
        return null;
      }
    })
    .catch(() => null);

  return warmupPromise;
}

function setPreviewFromBlob(blob) {
  const url = URL.createObjectURL(blob);
  preview.src = url;
  selectedBlob = blob;
  predictBtn.disabled = false;
}

async function startCamera() {
  if (mediaStream) return;

  mediaStream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: "environment",
    },
    audio: false,
  });

  camera.srcObject = mediaStream;
  camera.classList.remove("hidden");
  captureBtn.disabled = false;
  cameraBtn.textContent = "Stop Camera";
}

function stopCamera() {
  if (!mediaStream) return;

  mediaStream.getTracks().forEach((track) => track.stop());
  mediaStream = null;
  camera.srcObject = null;
  camera.classList.add("hidden");
  captureBtn.disabled = true;
  cameraBtn.textContent = "Use Camera";
}

fileInput.addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (!file) return;
  setPreviewFromBlob(file);
  warmupModel();
});

cameraBtn.addEventListener("click", async () => {
  try {
    if (mediaStream) {
      stopCamera();
      return;
    }
    await startCamera();
  } catch (error) {
    alert("Unable to access camera. Please allow permissions and try again.");
  }
});

captureBtn.addEventListener("click", async () => {
  if (!mediaStream) return;

  const width = camera.videoWidth || 640;
  const height = camera.videoHeight || 480;

  captureCanvas.width = width;
  captureCanvas.height = height;

  const ctx = captureCanvas.getContext("2d");
  ctx.drawImage(camera, 0, 0, width, height);

  captureCanvas.toBlob((blob) => {
    if (!blob) {
      alert("Failed to capture image from camera.");
      return;
    }
    setPreviewFromBlob(blob);
    warmupModel();
  }, "image/jpeg", 0.92);
});

predictBtn.addEventListener("click", async () => {
  if (!selectedBlob) return;

  const formData = new FormData();
  formData.append("image", selectedBlob, "leaf.jpg");

  predictBtn.disabled = true;
  predictBtn.textContent = "Predicting...";

  try {
    await warmupModel();

    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    const raw = await response.text();
    let data;
    try {
      data = JSON.parse(raw);
    } catch {
      throw new Error("Server returned an invalid response. Please wait 20-30 seconds and try Predict again.");
    }

    if (!response.ok) {
      throw new Error(data.error || "Prediction failed.");
    }

    resultBox.classList.remove("hidden");
    statusText.textContent = `Status: ${data.status}`;
    statusText.className = data.status.toLowerCase();

    if (data.predicted_class) {
      plantText.textContent = `Plant: ${data.plant || "Unknown"}`;
      diseaseText.textContent = `Disease: ${data.disease || "Unknown"}`;
    } else {
      plantText.textContent = "Plant: Not assigned";
      diseaseText.textContent = "Disease: Not assigned";
    }

    confidenceText.textContent = `Confidence: ${(data.confidence * 100).toFixed(2)}%`;
    messageText.textContent = data.message ? `Message: ${data.message}` : "Message:";

    topPredictions.innerHTML = "";
    data.top_predictions.forEach((item) => {
      const li = document.createElement("li");
      li.textContent = item.display || `${item.class}: ${(item.confidence * 100).toFixed(2)}%`;
      topPredictions.appendChild(li);
    });
  } catch (error) {
    alert(error.message);
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = "Predict";
  }
});

window.addEventListener("beforeunload", () => {
  stopCamera();
});

