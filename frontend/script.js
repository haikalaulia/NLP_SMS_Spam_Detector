const API_BASE = "http://127.0.0.1:8000";
let activeModel = "both";

const input = document.getElementById("sms-input");
const charCount = document.getElementById("char-count");
const btnAnalyze = document.getElementById("btn-analyze");
const btnClear = document.getElementById("btn-clear");
const btnText = document.getElementById("btn-text");
const spinner = document.getElementById("spinner");
const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");
const resultSec = document.getElementById("result-section");
const resultGrid = document.getElementById("result-grid");
const verdictWrap = document.getElementById("verdict-wrap");
const themeToggle = document.getElementById("theme-toggle");
const themeLabel = document.getElementById("theme-label");
const iconSun = document.getElementById("icon-sun");
const iconMoon = document.getElementById("icon-moon");

let isDark = true;
themeToggle.addEventListener("click", () => {
  isDark = !isDark;
  document.documentElement.setAttribute(
    "data-theme",
    isDark ? "dark" : "light",
  );
  themeLabel.textContent = isDark ? "Light" : "Dark";
  iconSun.style.display = isDark ? "block" : "none";
  iconMoon.style.display = isDark ? "none" : "block";
});

const pct = (v) => (v * 100).toFixed(1) + "%";
const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);
const MODEL_NAMES = { indobert: "IndoBERT", xlmroberta: "XLM-RoBERTa" };

async function checkHealth() {
  try {
    const r = await fetch(`${API_BASE}/health`, {
      signal: AbortSignal.timeout(3000),
    });
    if (r.ok) {
      const d = await r.json();
      const loaded = d.loaded_models || [];
      statusDot.className = "status-dot online";
      statusText.textContent =
        loaded.length > 0
          ? `API online · ${loaded.length} model dimuat`
          : "API online · mode mock";
      statusText.style.color = "var(--text-muted)";
    } else throw new Error();
  } catch {
    statusDot.className = "status-dot offline";
    statusText.textContent = "API offline · jalankan uvicorn";
    statusText.style.color = "var(--red)";
  }
}

input.addEventListener("input", () => {
  const len = input.value.length;
  charCount.textContent = `${len} / 500`;
  btnAnalyze.disabled = len === 0;
});

document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document
      .querySelectorAll(".tab-btn")
      .forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    activeModel = btn.dataset.model;
  });
});

document.querySelectorAll(".ex-pill").forEach((pill) => {
  pill.addEventListener("click", () => {
    input.value = pill.dataset.text;
    input.dispatchEvent(new Event("input"));
    input.focus();
  });
});

btnClear.addEventListener("click", () => {
  input.value = "";
  input.dispatchEvent(new Event("input"));
  resultSec.classList.remove("visible");
  setTimeout(() => {
    resultGrid.innerHTML = "";
    verdictWrap.innerHTML = "";
  }, 400);
});

function buildResultCard(modelKey, res) {
  const { label, confidence, scores, mode, latency_ms } = res;
  const delay = modelKey === "xlmroberta" ? "0.1s" : "0s";

  const scoreRows = ["normal", "promo", "penipuan"]
    .map(
      (l) => `
    <div class="score-row">
      <span class="score-name">${l}</span>
      <div class="score-bar-bg">
        <div class="score-bar-fill ${l}" style="width:0%" data-target="${(scores[l] * 100).toFixed(1)}%"></div>
      </div>
      <span class="score-pct">${pct(scores[l])}</span>
    </div>`,
    )
    .join("");

  return `
    <div class="model-result-card result-${label}" style="animation-delay:${delay}">
      <div class="result-model-name">${MODEL_NAMES[modelKey] || modelKey}</div>
      <div class="result-label-wrap">
        <div class="result-indicator ${label}"></div>
        <span class="result-label ${label}">${cap(label)}</span>
      </div>
      <div class="confidence-block">
        <div class="conf-header">
          <span>Confidence</span>
          <span class="conf-val">${pct(confidence)}</span>
        </div>
        <div class="conf-bar-bg">
          <div class="conf-bar-fill ${label}" style="width:0%" data-target="${pct(confidence)}"></div>
        </div>
      </div>
      <div class="scores-wrap">${scoreRows}</div>
      <div class="result-meta">
        <span class="meta-mode">${mode}</span>
        <span>${latency_ms} ms</span>
      </div>
    </div>`;
}

function animateBars() {
  requestAnimationFrame(() => {
    document.querySelectorAll("[data-target]").forEach((el) => {
      setTimeout(() => {
        el.style.width = el.dataset.target;
      }, 50);
    });
  });
}

const SVG_AGREE = `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg>`;
const SVG_DIFF = `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="2"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/></svg>`;

function buildVerdict(results) {
  const keys = Object.keys(results);
  if (keys.length < 2) return "";
  const [a, b] = keys;
  if (results[a].label === results[b].label) {
    return `
      <div class="verdict-card">
        <div class="verdict-icon-wrap">${SVG_AGREE}</div>
        <div class="verdict-text">
          <strong>Kedua model sepakat:</strong> Pesan ini diklasifikasikan sebagai
          <strong>${cap(results[a].label)}</strong>.
          IndoBERT (${pct(results[a].confidence)}) &amp; XLM-RoBERTa (${pct(results[b].confidence)}).
        </div>
      </div>`;
  } else {
    const winner = results[a].confidence >= results[b].confidence ? a : b;
    return `
      <div class="verdict-card">
        <div class="verdict-icon-wrap">${SVG_DIFF}</div>
        <div class="verdict-text">
          <strong>Kedua model berbeda pendapat.</strong>
          ${MODEL_NAMES[winner]} lebih percaya diri dengan label
          <strong>${cap(results[winner].label)}</strong> (${pct(results[winner].confidence)}).
          Ini bisa menunjukkan pesan yang ambigu atau mengandung code-switching.
        </div>
      </div>`;
  }
}

async function analyze() {
  const text = input.value.trim();
  if (!text) return;
  btnAnalyze.disabled = true;
  spinner.style.display = "block";
  btnText.textContent = "Menganalisis...";
  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, model: activeModel }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Terjadi kesalahan pada server.");
    }
    const data = await res.json();
    resultGrid.innerHTML = Object.entries(data.results)
      .map(([key, result]) => buildResultCard(key, result))
      .join("");
    verdictWrap.innerHTML = buildVerdict(data.results);
    resultSec.classList.add("visible");
    animateBars();
    setTimeout(
      () => resultSec.scrollIntoView({ behavior: "smooth", block: "nearest" }),
      100,
    );
  } catch (e) {
    resultGrid.innerHTML = `
      <div style="grid-column:1/-1;text-align:center;padding:32px;color:var(--red);">
        <svg xmlns="http://www.w3.org/2000/svg" width="36" height="36" fill="none" viewBox="0 0 24 24" stroke="currentColor" stroke-width="1.5" style="margin-bottom:12px;display:block;margin-left:auto;margin-right:auto;">
          <circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>
        </svg>
        <div style="font-family:var(--font-mono);font-size:13px;">${e.message}</div>
        <div style="font-size:12px;color:var(--text-dim);margin-top:8px;">
          Pastikan backend berjalan: <code>uvicorn main:app --reload</code>
        </div>
      </div>`;
    verdictWrap.innerHTML = "";
    resultSec.classList.add("visible");
  } finally {
    spinner.style.display = "none";
    btnText.textContent = "Analisis";
    btnAnalyze.disabled = input.value.length === 0;
  }
}

btnAnalyze.addEventListener("click", analyze);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) analyze();
});

checkHealth();
setInterval(checkHealth, 15000);
