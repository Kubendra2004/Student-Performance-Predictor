const responseBox = document.getElementById("responseBox");
const healthBadge = document.getElementById("healthBadge");
const predictedValue = document.getElementById("predictedValue");
const scoreLabel = document.getElementById("scoreLabel");
const scoreBar = document.getElementById("scoreBar");
const recentPredictions = document.getElementById("recentPredictions");
const csvStatusBadge = document.getElementById("csvStatusBadge");
const csvPreviewTable = document.getElementById("csvPreviewTable");
const activeVersionBadge = document.getElementById("activeVersionBadge");
const versionList = document.getElementById("versionList");

const simFeature = document.getElementById("simFeature");
const simStart = document.getElementById("simStart");
const simEnd = document.getElementById("simEnd");
const simPoints = document.getElementById("simPoints");

const maeValue = document.getElementById("maeValue");
const rmseValue = document.getElementById("rmseValue");
const r2Value = document.getElementById("r2Value");
const rowsValue = document.getElementById("rowsValue");

const featureRanges = {
  study_hours: { min: 0, max: 24, step: 0.1 },
  attendance_percent: { min: 0, max: 100, step: 1 },
  previous_grade: { min: 0, max: 100, step: 1 },
  assignments_completed: { min: 0, max: 100, step: 1 },
  sleep_hours: { min: 0, max: 24, step: 0.1 },
};

let metricsChart = null;
let whatIfChart = null;

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

function toBadgeVariant(isValid) {
  return isValid ? "good" : "risk";
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

function getCurrentPayloadFromForm() {
  const form = document.getElementById("predictForm");
  return {
    study_hours: Number(form.study_hours.value || 0),
    attendance_percent: Number(form.attendance_percent.value || 0),
    previous_grade: Number(form.previous_grade.value || 0),
    assignments_completed: Number(form.assignments_completed.value || 0),
    sleep_hours: Number(form.sleep_hours.value || 0),
  };
}

function renderCsvPreview(text) {
  const rows = text.trim().split(/\r?\n/).slice(0, 8).map((line) => line.split(","));
  if (!rows.length) return;

  const [header, ...dataRows] = rows;
  const thead = csvPreviewTable.querySelector("thead");
  const tbody = csvPreviewTable.querySelector("tbody");
  thead.innerHTML = `<tr>${header.map((cell) => `<th>${cell}</th>`).join("")}</tr>`;
  tbody.innerHTML = dataRows
    .map((row) => `<tr>${row.map((cell) => `<td>${cell}</td>`).join("")}</tr>`)
    .join("");
}

function renderMetricsChart(metrics) {
  const ctx = document.getElementById("metricsChart");
  const chartData = [Number(metrics.mae), Number(metrics.rmse), Number(metrics.r2) * 10];

  if (metricsChart) {
    metricsChart.data.datasets[0].data = chartData;
    metricsChart.update();
    return;
  }

  metricsChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["MAE", "RMSE", "R² x10"],
      datasets: [
        {
          label: "Metric Value",
          data: chartData,
          backgroundColor: ["#1f7a8c", "#2a9d8f", "#ee6c4d"],
          borderRadius: 8,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        y: { beginAtZero: true },
      },
    },
  });
}

function renderVersionList(payload) {
  const activeVersion = payload.active_version_id;
  activeVersionBadge.className = activeVersion ? "badge good" : "badge neutral";
  activeVersionBadge.textContent = activeVersion ? `Active: ${activeVersion}` : "Active: none";

  if (!payload.versions.length) {
    versionList.innerHTML = "<li>No model versions yet.</li>";
    return;
  }

  versionList.innerHTML = payload.versions
    .slice(0, 5)
    .map((item) => {
      const marker = item.version_id === activeVersion ? " (active)" : "";
      return `<li>${item.version_id}${marker} | rows ${item.trained_rows} | r2 ${Number(item.r2).toFixed(3)}</li>`;
    })
    .join("");
}

function updateSimulationInputsByFeature() {
  const chosen = simFeature.value;
  const range = featureRanges[chosen];
  simStart.value = String(range.min);
  simEnd.value = String(range.max);
  simStart.step = String(range.step);
  simEnd.step = String(range.step);
}

async function runWhatIfSimulation() {
  const feature = simFeature.value;
  const start = Number(simStart.value);
  const end = Number(simEnd.value);
  const points = Number(simPoints.value);
  if (Number.isNaN(start) || Number.isNaN(end) || Number.isNaN(points) || points < 2) {
    appendConsole("Simulation Validation", { error: "Invalid simulation range or point count." });
    return;
  }

  const basePayload = getCurrentPayloadFromForm();
  const labels = [];
  const values = [];

  for (let index = 0; index < points; index += 1) {
    const ratio = points === 1 ? 0 : index / (points - 1);
    const current = start + (end - start) * ratio;
    const payload = { ...basePayload, [feature]: Number(current.toFixed(2)) };

    if (feature === "assignments_completed") {
      payload[feature] = Math.round(payload[feature]);
    }

    try {
      const { data } = await callApi("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      labels.push(String(payload[feature]));
      values.push(Number(data.predicted_score));
    } catch (error) {
      appendConsole("Simulation Failed", { error: error.message, payload });
      return;
    }
  }

  const ctx = document.getElementById("whatIfChart");
  if (!whatIfChart) {
    whatIfChart = new Chart(ctx, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: `Predicted Score vs ${feature}`,
            data: values,
            borderColor: "#1f7a8c",
            backgroundColor: "rgba(31, 122, 140, 0.18)",
            tension: 0.3,
            fill: true,
            pointRadius: 2,
          },
        ],
      },
      options: {
        responsive: true,
        plugins: { legend: { display: true } },
        scales: {
          y: { min: 0, max: 100 },
        },
      },
    });
  } else {
    whatIfChart.data.labels = labels;
    whatIfChart.data.datasets[0].label = `Predicted Score vs ${feature}`;
    whatIfChart.data.datasets[0].data = values;
    whatIfChart.update();
  }

  appendConsole("Simulation Complete", {
    feature,
    points,
    min_pred: Math.min(...values),
    max_pred: Math.max(...values),
  });
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
    await refreshVersionMetadata();
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
    await refreshVersionMetadata();
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
    renderMetricsChart(data);
    appendConsole(`Metrics Loaded (${durationMs} ms)`, data);
  } catch (error) {
    appendConsole("Metrics Fetch Failed", { error: error.message });
  } finally {
    setLoading(button, "Refreshing...", false);
  }
});

async function refreshVersionMetadata() {
  try {
    const { data } = await callApi("/models/versions");
    renderVersionList(data);
  } catch (error) {
    appendConsole("Model Versions Fetch Failed", { error: error.message });
  }
}

document.getElementById("csvFile").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) {
    csvStatusBadge.className = "badge neutral";
    csvStatusBadge.textContent = "No file selected";
    csvPreviewTable.querySelector("thead").innerHTML = "";
    csvPreviewTable.querySelector("tbody").innerHTML = "";
    return;
  }

  const text = await file.text();
  renderCsvPreview(text);

  const formData = new FormData();
  formData.append("file", file);

  try {
    const { data } = await callApi("/validate/upload", {
      method: "POST",
      body: formData,
    });
    csvStatusBadge.className = `badge ${toBadgeVariant(data.is_valid_for_training)}`;
    csvStatusBadge.textContent = data.is_valid_for_training ? "Valid for training" : "Validation issues";
    appendConsole("CSV Validation Report", data);
  } catch (error) {
    csvStatusBadge.className = "badge risk";
    csvStatusBadge.textContent = "Validation failed";
    appendConsole("CSV Validation Failed", { error: error.message });
  }
});

document.getElementById("runSimulationBtn").addEventListener("click", async (event) => {
  const button = event.currentTarget;
  setLoading(button, "Simulating...", true);
  await runWhatIfSimulation();
  setLoading(button, "Simulating...", false);
});

simFeature.addEventListener("change", updateSimulationInputsByFeature);

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

updateSimulationInputsByFeature();
refreshVersionMetadata();
