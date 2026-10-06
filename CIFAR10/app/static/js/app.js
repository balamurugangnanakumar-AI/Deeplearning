(() => {
  const input = document.querySelector("#image-input");
  const dropzone = document.querySelector("#dropzone");
  const browseButton = document.querySelector("#browse-button");
  const clearButton = document.querySelector("#clear-button");
  const predictButton = document.querySelector("#predict-button");
  const uploadError = document.querySelector("#upload-error");
  const emptyState = document.querySelector("#drop-empty");
  const previewState = document.querySelector("#preview-state");
  const preview = document.querySelector("#image-preview");
  const resultPanel = document.querySelector("#result-panel");
  const resultStatus = document.querySelector("#result-status");
  const resultEmpty = document.querySelector("#result-empty");
  const loadingState = document.querySelector("#loading-state");
  const predictionState = document.querySelector("#prediction-state");
  let selectedFile = null;
  let previewUrl = null;
  let isPredicting = false;

  const allowedTypes = new Set(["image/jpeg", "image/png", "image/webp"]);
  const maxBytes = 5 * 1024 * 1024;

  function showError(message) {
    uploadError.textContent = message;
    uploadError.hidden = false;
  }

  function validateFile(file) {
    const extension = file.name.split(".").pop().toLowerCase();
    if (!allowedTypes.has(file.type) || !["jpg", "jpeg", "png", "webp"].includes(extension)) {
      return "Unsupported file format. Please upload JPG, PNG, or WEBP.";
    }
    if (file.size > maxBytes) return "File size exceeds the 5 MB limit.";
    if (file.size === 0) return "Unable to process this image.";
    return "";
  }

  function formatBytes(bytes) {
    return bytes >= 1024 * 1024 ? `${(bytes / (1024 * 1024)).toFixed(2)} MB` : `${Math.max(1, Math.round(bytes / 1024))} KB`;
  }

  function handleFileSelection(file) {
    if (isPredicting) return;
    uploadError.hidden = true;
    const problem = validateFile(file);
    if (problem) {
      showError(problem);
      return;
    }
    selectedFile = file;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(file);
    preview.src = previewUrl;
    document.querySelector("#file-name").textContent = file.name;
    document.querySelector("#file-size").textContent = formatBytes(file.size);
    preview.onload = () => {
      document.querySelector("#image-dimensions").textContent = `${preview.naturalWidth} × ${preview.naturalHeight} px`;
    };
    emptyState.hidden = true;
    previewState.hidden = false;
    clearButton.disabled = false;
    predictButton.disabled = false;
    resetResult();
  }

  function resetResult() {
    resultEmpty.hidden = false;
    loadingState.hidden = true;
    predictionState.hidden = true;
    resultPanel.setAttribute("aria-busy", "false");
    resultStatus.textContent = "AWAITING IMAGE";
  }

  function clearImage() {
    if (isPredicting) return;
    selectedFile = null;
    input.value = "";
    preview.removeAttribute("src");
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
    previewState.hidden = true;
    emptyState.hidden = false;
    clearButton.disabled = true;
    predictButton.disabled = true;
    uploadError.hidden = true;
    resetResult();
  }

  function showLoading() {
    resultEmpty.hidden = true;
    predictionState.hidden = true;
    loadingState.hidden = false;
    resultPanel.setAttribute("aria-busy", "true");
    resultStatus.textContent = "ANALYZING";
    predictButton.innerHTML = '<span class="button-spark" aria-hidden="true">✳</span> Analyzing image...';
  }

  function hideLoading() {
    loadingState.hidden = true;
    resultPanel.setAttribute("aria-busy", "false");
    predictButton.innerHTML = '<span class="button-spark" aria-hidden="true">✳</span> Predict image <span class="button-arrow" aria-hidden="true">↗</span>';
    predictButton.disabled = !selectedFile;
  }

  function displayTopPredictions(predictions) {
    const list = document.querySelector("#prediction-list");
    list.replaceChildren();
    predictions.forEach((item, index) => {
      const row = document.createElement("li");
      row.className = `prediction-row${index === 0 ? " is-best" : ""}`;
      const label = document.createElement("span");
      label.textContent = item.class;
      const track = document.createElement("span");
      track.className = "bar-track";
      const fill = document.createElement("span");
      fill.className = "bar-fill";
      fill.style.setProperty("--bar-width", `${Math.max(0, Math.min(100, item.confidence * 100))}%`);
      track.append(fill);
      const percentage = document.createElement("span");
      percentage.className = "prediction-percent";
      percentage.textContent = `${(item.confidence * 100).toFixed(2)}%`;
      row.append(label, track, percentage);
      list.append(row);
    });
  }

  function displayPrediction(data) {
    resultEmpty.hidden = true;
    predictionState.hidden = false;
    resultStatus.textContent = "ANALYSIS COMPLETE";
    document.querySelector("#predicted-class").textContent = data.prediction.class;
    const confidence = Math.max(0, Math.min(100, data.prediction.confidence * 100));
    document.querySelector("#confidence-value").textContent = `${confidence.toFixed(2)}%`;
    document.querySelector("#confidence-ring").style.setProperty("--progress", `${confidence}%`);
    document.querySelector("#confidence-ring").setAttribute("aria-label", `${confidence.toFixed(2)} percent confidence`);
    displayTopPredictions(data.top_predictions);
  }

  async function predictImage() {
    if (!selectedFile || isPredicting) return;
    isPredicting = true;
    predictButton.disabled = true;
    clearButton.disabled = true;
    uploadError.hidden = true;
    showLoading();
    const formData = new FormData();
    formData.append("image", selectedFile);
    try {
      const response = await fetch("/api/predict", { method: "POST", body: formData });
      const data = await response.json();
      if (!response.ok || !data.success) throw new Error(data.error || "Unable to process this image.");
      displayPrediction(data);
    } catch (error) {
      resetResult();
      showError(error.message || "Unable to process this image.");
    } finally {
      isPredicting = false;
      hideLoading();
      clearButton.disabled = !selectedFile;
    }
  }

  browseButton.addEventListener("click", (event) => {
    event.stopPropagation();
    input.click();
  });
  dropzone.addEventListener("click", (event) => {
    if (event.target.closest("button")) return;
    input.click();
  });
  dropzone.addEventListener("keydown", (event) => {
    if ((event.key === "Enter" || event.key === " ") && event.target === dropzone) {
      event.preventDefault();
      input.click();
    }
  });
  input.addEventListener("change", () => {
    if (input.files && input.files[0]) handleFileSelection(input.files[0]);
  });
  clearButton.addEventListener("click", clearImage);
  predictButton.addEventListener("click", predictImage);
  ["dragenter", "dragover"].forEach((eventName) => dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("is-dragging");
  }));
  ["dragleave", "drop"].forEach((eventName) => dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("is-dragging");
  }));
  dropzone.addEventListener("drop", (event) => {
    const [file] = event.dataTransfer.files;
    if (file) handleFileSelection(file);
  });
  window.addEventListener("beforeunload", () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
  });
})();