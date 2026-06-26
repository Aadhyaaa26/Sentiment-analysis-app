// script.js
// Handles: typing in the box, calling the backend, and animating the gauge.

const textInput = document.getElementById("textInput");
const charCount = document.getElementById("charCount");
const analyzeBtn = document.getElementById("analyzeBtn");
const errorMsg = document.getElementById("errorMsg");
const needle = document.getElementById("needle");
const readout = document.getElementById("readout");
const readoutDetail = document.getElementById("readoutDetail");
const confidenceNum = document.getElementById("confidenceNum");
const scoreBars = document.getElementById("scoreBars");

const SENTIMENT_COLOR = {
  positive: "var(--positive)",
  neutral: "var(--neutral)",
  negative: "var(--negative)",
};

textInput.addEventListener("input", () => {
  charCount.textContent = textInput.value.length;
});

analyzeBtn.addEventListener("click", analyze);
textInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) analyze();
});

async function analyze() {
  const text = textInput.value.trim();
  hideError();

  if (!text) {
    showError("Type something first — the gauge needs a sentence to read.");
    return;
  }

  setLoading(true);
  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();

    if (!res.ok) {
      showError(data.error || "Something went wrong reading that text.");
      return;
    }

    renderResult(data);
  } catch (err) {
    showError("Couldn't reach the server. Is app.py still running?");
  } finally {
    setLoading(false);
  }
}

function renderResult(data) {
  const { sentiment, confidence, scores } = data;

  // Move the needle: angle from -80 (negative) to +80 (positive)
  const pos = scores.positive || 0;
  const neg = scores.negative || 0;
  const angle = Math.max(-80, Math.min(80, (pos - neg) * 80));
  needle.style.transform = `rotate(${angle}deg)`;

  // Update the word + color
  readout.classList.remove("readout-empty");
  const word = readout.querySelector(".readout-word");
  const help = readout.querySelector(".readout-help");
  word.textContent = sentiment;
  word.style.color = SENTIMENT_COLOR[sentiment] || "var(--text)";
  help.textContent = `This text reads as ${sentiment}.`;

  // Confidence number
  confidenceNum.textContent = confidence;
  readoutDetail.hidden = false;

  // Score bars, ordered negative / neutral / positive
  const order = ["negative", "neutral", "positive"];
  scoreBars.innerHTML = order
    .map((key) => {
      const value = scores[key] ?? 0;
      const pct = Math.round(value * 100);
      return `
        <div class="score-bar-row">
          <span>${key}</span>
          <span class="score-bar-track">
            <span class="score-bar-fill" style="width:${pct}%; background:${SENTIMENT_COLOR[key]}"></span>
          </span>
          <span>${pct}%</span>
        </div>`;
    })
    .join("");
}

function setLoading(isLoading) {
  analyzeBtn.disabled = isLoading;
  analyzeBtn.textContent = isLoading ? "Reading…" : "Analyze";
}

function showError(message) {
  errorMsg.textContent = message;
  errorMsg.hidden = false;
}

function hideError() {
  errorMsg.hidden = true;
}
