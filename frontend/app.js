const responseBox = document.getElementById("responseBox");
const healthBadge = document.getElementById("healthBadge");
const predictedValue = document.getElementById("predictedValue");
const scoreLabel = document.getElementById("scoreLabel");
const scoreBar = document.getElementById("scoreBar");
const recentPredictions = document.getElementById("recentPredictions");

const maeValue = document.getElementById("maeValue");
const rmseValue = document.getElementById("rmseValue");
const r2Value = document.getElementById("r2Value");
const rowsValue = document.getElementById("rowsValue");

const profiles = {
  steady: {
    study_hours: 4.2,
    attendance_percent: 86,
    previous_grade: 72,
    assignments_completed: 11,
    sleep_hours: 7.1,
  },
  struggling: {
    study_hours: 2.0,
    attendance_percent: 68,
    previous_grade: 49,
    assignments_completed: 5,
    sleep_hours: 5.8,
  },
  topper: {
    study_hours: 6.5,
    attendance_percent: 97,
    previous_grade: 90,
    assignments_completed: 16,
    sleep_hours: 7.8,
  },
};

function getApiBase() {
  const value = document.getElementById("apiBase").value.trim();
  return value.replace(/\/$/, "");
}

function appendConsole(title, payload) {
  const stamp = new Date().toLocaleTimeString();
  const previous = responseBox.textContent.trim();
  const block = `[${stamp}] ${title}\n${JSON.stringify(payload, null, 2)}`;
  responseBox.textContent = previous ? `${block}\n\n${previous}` : block;
}

function setHealthBadge(variant, text) {
  healthBadge.className = `badge ${variant}`;
  healthBadge.textContent = text;
}

function scoreVariant(score) {
  if (score >= 80) return { variant: "good", text: "High Performance" };
  if (score >= 60) return { variant: "warn", text: "Moderate Performance" };
  return { variant: "risk", text: "At Risk" };
}

function renderPrediction(score) {
  predictedValue.textContent = `${score.toFixed(1)} / 100`;
  scoreBar.style.width = `${Math.max(0, Math.min(score, 100))}%`;

  const { variant, text } = scoreVariant(score);
  scoreLabel.className = `badge ${variant}`;
  scoreLabel.textContent = text;

  const item = document.createElement("li");
  item.textContent = `${new Date().toLocaleTimeString()} - Predicted ${score.toFixed(1)}`;
  recentPredictions.prepend(item);
  if (recentPredictions.children.length > 6) {
    recentPredictions.removeChild(recentPredictions.lastElementChild);
  }
}

async function callApi(path, options = {}) {
  const url = `${getApiBase()}${path}`;
  const start = performance.now();

  const response = await fetch(url, options);
  const contentType = response.headers.get("content-type") || "";
  const data = contentType.includes("application/json")
    ? await response.json()
    : await response.text();

  const durationMs = Math.round(performance.now() - start);

  if (!response.ok) {
    const detail = data?.detail || data || `Request failed with ${response.status}`;
    throw new Error(`${detail} (${durationMs} ms)`);
  }

  return { data, durationMs };
}

function setLoading(button, loadingText, isLoading) {
  if (isLoading) {
    button.dataset.originalText = button.textContent;
    button.textContent = loadingText;
    button.disabled = true;
    return;
  }
  button.textContent = button.dataset.originalText || button.textContent;
  button.disabled = false;
}

document.getElementById("healthBtn").addEventListener("click", async (event) => {
  const button = event.currentTarget;
  setLoading(button, "Checking...", true);

  try {
    const { data, durationMs } = await callApi("/health");
    const variant = data.model_ready ? "good" : "warn";
    const label = data.model_ready ? "Healthy + Model Ready" : "Healthy + Not Trained";
    setHealthBadge(variant, label);
    appendConsole(`Health Check (${durationMs} ms)`, data);
  } catch (error) {
    setHealthBadge("risk", "API Unreachable");
    appendConsole("Health Check Failed", { error: error.message });
  } finally {
    setLoading(button, "Checking...", false);
  }
});

document.getElementById("trainSampleBtn").addEventListener("click", async (event) => {
  const button = event.currentTarget;
  setLoading(button, "Training...", true);

  try {
    const { data, durationMs } = await callApi("/train/sample", { method: "POST" });
    appendConsole(`Sample Training Complete (${durationMs} ms)`, data);
    setHealthBadge("good", "Healthy + Model Ready");
  } catch (error) {
    appendConsole("Sample Training Failed", { error: error.message });
  } finally {
    setLoading(button, "Training...", false);
  }
});

document.getElementById("trainUploadBtn").addEventListener("click", async (event) => {
  const button = event.currentTarget;
  const fileInput = document.getElementById("csvFile");
  const file = fileInput.files[0];
  if (!file) {
    appendConsole("Upload Validation", { error: "Please choose a CSV file first." });
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  setLoading(button, "Uploading...", true);

  try {
    const { data, durationMs } = await callApi("/train/upload", {
      method: "POST",
      body: formData,
    });
    appendConsole(`Upload Training Complete (${durationMs} ms)`, data);
    setHealthBadge("good", "Healthy + Model Ready");
  } catch (error) {
    appendConsole("Upload Training Failed", { error: error.message });
  } finally {
    setLoading(button, "Uploading...", false);
  }
});

document.getElementById("predictForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = document.getElementById("predictBtn");
  const form = event.target;
  const payload = {
    study_hours: Number(form.study_hours.value),
    attendance_percent: Number(form.attendance_percent.value),
    previous_grade: Number(form.previous_grade.value),
    assignments_completed: Number(form.assignments_completed.value),
    sleep_hours: Number(form.sleep_hours.value),
  };

  setLoading(button, "Predicting...", true);

  try {
    const { data, durationMs } = await callApi("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    renderPrediction(data.predicted_score);
    appendConsole(`Prediction Complete (${durationMs} ms)`, {
      input: payload,
      output: data,
    });
  } catch (error) {
    appendConsole("Prediction Failed", { error: error.message, input: payload });
  } finally {
    setLoading(button, "Predicting...", false);
  }
});

document.getElementById("metricsBtn").addEventListener("click", async (event) => {
  const button = event.currentTarget;
  setLoading(button, "Refreshing...", true);

  try {
    const { data, durationMs } = await callApi("/metrics");
    maeValue.textContent = Number(data.mae).toFixed(2);
    rmseValue.textContent = Number(data.rmse).toFixed(2);
    r2Value.textContent = Number(data.r2).toFixed(3);
    rowsValue.textContent = String(data.trained_rows);
    appendConsole(`Metrics Loaded (${durationMs} ms)`, data);
  } catch (error) {
    appendConsole("Metrics Fetch Failed", { error: error.message });
  } finally {
    setLoading(button, "Refreshing...", false);
  }
});

document.querySelectorAll(".profileBtn").forEach((button) => {
  button.addEventListener("click", () => {
    const profileKey = button.dataset.profile;
    const profile = profiles[profileKey];
    if (!profile) return;

    const form = document.getElementById("predictForm");
    form.study_hours.value = profile.study_hours;
    form.attendance_percent.value = profile.attendance_percent;
    form.previous_grade.value = profile.previous_grade;
    form.assignments_completed.value = profile.assignments_completed;
    form.sleep_hours.value = profile.sleep_hours;

    appendConsole("Profile Applied", { profile: profileKey, values: profile });
  });
});
