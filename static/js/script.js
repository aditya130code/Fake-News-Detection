const newsInput = document.querySelector("#news-input");
const checkButton = document.querySelector("#check-button");
const clearButton = document.querySelector("#clear-button");
const formError = document.querySelector("#form-error");
const resultPanel = document.querySelector("#result-panel");

function setLoading(isLoading) {
  checkButton.disabled = isLoading;
  checkButton.classList.toggle("is-loading", isLoading);
  checkButton.querySelector(".button-label").textContent = isLoading
    ? "Analyzing..."
    : "Check this story";
}

function clearResult() {
  resultPanel.hidden = true;
  resultPanel.classList.remove("is-fake");
  formError.hidden = true;
  formError.textContent = "";
}

function showError(message) {
  formError.textContent = message;
  formError.hidden = false;
  resultPanel.hidden = true;
}

newsInput.addEventListener("input", () => {
  document.querySelector("#character-count").textContent =
    `${newsInput.value.length.toLocaleString()} / 100,000`;
  if (!formError.hidden) {
    formError.hidden = true;
    formError.textContent = "";
  }
});

clearButton.addEventListener("click", () => {
  newsInput.value = "";
  document.querySelector("#character-count").textContent = "0 / 100,000";
  clearResult();
  newsInput.focus();
});

document.querySelector("#dismiss-result").addEventListener("click", clearResult);

checkButton.addEventListener("click", async () => {
  const text = newsInput.value.trim();
  clearResult();
  if (!text) {
    showError("Please paste a headline or article before checking.");
    newsInput.focus();
    return;
  }

  setLoading(true);
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const payload = await response.json();
    if (!response.ok) {
      showError(payload.error || "The prediction could not be completed.");
      return;
    }

    const isFake = payload.label === 0;
    resultPanel.classList.toggle("is-fake", isFake);
    document.querySelector("#result-title").textContent = payload.prediction;
    document.querySelector("#result-message").textContent = payload.message;
    resultPanel.hidden = false;
  } catch {
    showError("Could not reach the local application. Refresh the page and try again.");
  } finally {
    setLoading(false);
  }
});

function setMetric(id, value) {
  document.querySelector(id).textContent =
    Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "—";
}

async function loadModelInfo() {
  const status = document.querySelector("#model-status");
  try {
    const response = await fetch("/api/info");
    if (!response.ok) throw new Error("Model information is unavailable.");
    const info = await response.json();
    status.classList.toggle("is-ready", info.model_available);
    status.classList.toggle("is-error", !info.model_available);
    status.innerHTML = `<i></i>${info.model_available ? "Ready" : "Train model"}`;

    if (info.model) {
      document.querySelector("#model-algorithm").textContent = info.model.algorithm;
      document.querySelector("#model-features").textContent = info.model.features;
    }
    if (info.metrics) {
      setMetric("#metric-accuracy", info.metrics.accuracy);
      setMetric("#metric-precision", info.metrics.precision);
      setMetric("#metric-recall", info.metrics.recall);
      setMetric("#metric-f1", info.metrics.f1_score);
    }
    if (info.dataset) {
      const total = info.dataset.rows_after_cleaning.toLocaleString();
      const distribution = info.dataset.class_distribution_after_cleaning;
      document.querySelector("#dataset-summary").textContent =
        `${total} cleaned records · ${distribution.Fake.toLocaleString()} fake / ${distribution.Real.toLocaleString()} real`;
    }
  } catch {
    status.classList.add("is-error");
    status.innerHTML = "<i></i>Unavailable";
    document.querySelector("#dataset-summary").textContent =
      "Could not load model details. Check that the local app is running.";
  }
}

loadModelInfo();
