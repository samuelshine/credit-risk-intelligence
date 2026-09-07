// Credit Risk Intelligence Platform — UI controller.
// No build step, no framework: the app is five sections of one page, each
// fetching from the FastAPI backend at the same origin. State that crosses
// sections (the last-scored applicant) lives in `state` below; everything
// else is fetched fresh when a section is opened.

const state = {
  lastScoreRequest: null,   // { sk_id_curr } or { raw_fields }
  lastScoreResult: null,
  rulesBaseRate: null,
};

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

async function api(path, options) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const err = new Error(body.detail || `Request failed (${res.status})`);
    err.status = res.status;
    err.body = body;
    throw err;
  }
  return body;
}

function pct(x, digits = 1) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return (x * 100).toFixed(digits) + "%";
}
function num(x) {
  if (x === null || x === undefined) return "—";
  return Number(x).toLocaleString(undefined, { maximumFractionDigits: 0 });
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

// ============================================================= Navigation
function initNav() {
  $$(".rail-item[data-section]").forEach((btn) => {
    btn.addEventListener("click", () => showSection(btn.dataset.section));
  });
  $$(".link-btn[data-section]").forEach((btn) => {
    btn.addEventListener("click", () => showSection(btn.dataset.section));
  });
}

function showSection(name) {
  $$(".rail-item[data-section]").forEach((b) => b.classList.toggle("is-active", b.dataset.section === name));
  $$(".panel").forEach((p) => p.classList.toggle("is-active", p.id === `section-${name}`));
  if (name === "portfolio") loadPortfolio();
  if (name === "score") loadApplicantSample();
  if (name === "why") renderWhySection();
  if (name === "rules") loadRules();
}

// ================================================================= Health
async function loadHealth() {
  const el = $("#rail-status");
  try {
    const h = await api("/health");
    const problems = [];
    if (!h.database_ready) problems.push("no database");
    if (!h.model_ready) problems.push("no trained model");
    if (!h.llm_ready) problems.push("no LLM key");
    if (problems.length) {
      el.textContent = "Not fully set up: " + problems.join(", ") + ". See README.";
      el.classList.add("warn");
    } else {
      el.textContent = `Ready, running on ${h.data_mode} data`;
      el.classList.remove("warn");
    }
    el.hidden = false;
  } catch {
    el.textContent = "Could not reach the API.";
    el.classList.add("warn");
    el.hidden = false;
  }
}

// ============================================================== Portfolio
let portfolioLoaded = false;
async function loadPortfolio() {
  if (portfolioLoaded) return;
  const loading = $("#portfolio-loading");
  const body = $("#portfolio-body");
  try {
    const [summary, quality, insights] = await Promise.all([
      api("/api/eda/summary"), api("/api/eda/quality"), api("/api/eda/insights"),
    ]);
    renderTableSummary(summary.table_summary);
    renderQuality(quality.data_quality_findings);
    renderInsights(insights.insights);
    loading.hidden = true;
    body.hidden = false;
    portfolioLoaded = true;
  } catch (e) {
    loading.textContent = `Could not load the portfolio: ${e.message}`;
  }
}

function renderTableSummary(tables) {
  const t = $("#table-summary");
  t.innerHTML = `<thead><tr><th>Table</th><th>Description</th><th class="num">Rows</th><th class="num">Columns</th></tr></thead>
    <tbody>${tables.map((r) => `
      <tr><td>${escapeHtml(r.name)}</td><td>${escapeHtml(r.description || "")}</td>
      <td class="num">${num(r.row_count)}</td><td class="num">${r.column_count}</td></tr>
    `).join("")}</tbody>`;
}

function formatQualityValue(value) {
  // Each finding's `value` is a small dict of measured numbers - rendered
  // as individual "label: value" facts (real DOM elements, spaced by CSS),
  // not a raw JSON dump or a middot-joined string, so the portfolio page
  // reads as a document rather than a debug console.
  return Object.entries(value).map(([key, v]) => {
    const label = key.replace(/_/g, " ");
    let shown;
    if (v && typeof v === "object") {
      shown = Object.keys(v).length ? JSON.stringify(v) : "none";
    } else if (typeof v === "number") {
      const isFraction = /pct|rate|ratio/.test(key) && Math.abs(v) <= 1;
      shown = isFraction ? pct(v, 2) : v.toLocaleString();
    } else {
      shown = String(v);
    }
    return `<span class="fact">${escapeHtml(label)}: ${escapeHtml(shown)}</span>`;
  }).join("");
}

function renderQuality(findings) {
  const list = $("#quality-list");
  list.innerHTML = findings.map((f) => `
    <li class="quality-item">
      <span class="quality-sev sev-${f.severity}">${f.severity.replace("_", " ")}</span>
      <div>
        <div>${escapeHtml(f.description)}</div>
        <div class="quality-detail">${formatQualityValue(f.value)}</div>
      </div>
    </li>`).join("");
}

function renderInsights(insights) {
  $("#insight-list").innerHTML = insights.map((i) => `
    <div class="insight-card">
      <h3>${escapeHtml(i.title)}</h3>
      <div class="insight-headline">${escapeHtml(i.headline)}</div>
      <div class="insight-sowhat">${escapeHtml(i.so_what)}</div>
      <img src="${i.chart_url}" alt="${escapeHtml(i.title)}" loading="lazy">
    </div>`).join("");
}

// ================================================================= Score
function initScoreTabs() {
  $$(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      $$(".tab-btn").forEach((b) => b.classList.toggle("is-active", b === btn));
      $$(".tab-panel").forEach((p) => p.classList.toggle("is-active", p.dataset.tabPanel === btn.dataset.tab));
    });
  });
}

let sampleLoaded = false;
async function loadApplicantSample(force = false) {
  if (sampleLoaded && !force) return;
  const tbody = $("#applicant-picker tbody");
  tbody.innerHTML = `<tr><td colspan="7">Loading applicants…</td></tr>`;
  try {
    const rows = await api("/api/applicants/sample?n=20");
    tbody.innerHTML = rows.map((r) => `
      <tr data-sk-id="${r.sk_id_curr}">
        <td>${r.sk_id_curr}</td>
        <td class="num">${num(r.amt_income_total)}</td>
        <td class="num">${num(r.amt_credit)}</td>
        <td>${escapeHtml(r.code_gender || "—")}</td>
        <td>${escapeHtml(r.name_education_type || "—")}</td>
        <td class="${r.target === 1 ? "outcome-defaulted" : "outcome-repaid"}">
          ${r.target === 1 ? "Defaulted" : "Repaid"}
        </td>
        <td><button class="link-btn score-pick">Score →</button></td>
      </tr>`).join("");
    $$(".score-pick", tbody).forEach((btn) => {
      btn.addEventListener("click", (e) => {
        const skId = Number(e.target.closest("tr").dataset.skId);
        scoreApplicant({ sk_id_curr: skId });
      });
    });
    sampleLoaded = true;
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="7">Could not load applicants: ${escapeHtml(e.message)}</td></tr>`;
  }
}

function initManualForm() {
  $("#manual-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const data = new FormData(e.target);
    const raw = {};
    for (const [key, value] of data.entries()) {
      if (value === "") continue;
      if (key === "AGE_YEARS") { raw.DAYS_BIRTH = -Math.round(Number(value) * 365.25); continue; }
      if (key === "YEARS_EMPLOYED") { raw.DAYS_EMPLOYED = -Math.round(Number(value) * 365.25); continue; }
      raw[key] = isNaN(value) || value === "" ? value : Number(value);
    }
    scoreApplicant({ raw_fields: raw });
  });
}

async function scoreApplicant(request) {
  const resultEl = $("#score-result");
  try {
    const result = await api("/api/score", { method: "POST", body: JSON.stringify(request) });
    state.lastScoreRequest = request;
    state.lastScoreResult = result;
    renderScoreResult(result);
    resultEl.hidden = false;
    resultEl.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (e) {
    alert(`Could not score that applicant: ${e.message}`);
  }
}

function renderScoreResult(r) {
  $("#result-prob").textContent = pct(r.probability);
  const band = $("#result-band");
  band.textContent = r.risk_band + " risk";
  band.className = "band-pill band-" + r.risk_band;
  const outcome = r.actual_target !== null && r.actual_target !== undefined
    ? ` This applicant actually ${r.actual_target === 1 ? "defaulted" : "repaid"}.`
    : "";
  $("#result-sub").textContent =
    `${r.lift_over_base_rate}x the portfolio average of ${pct(r.base_rate)}.${outcome}`;
  positionRuler($("#ruler-mark"), r.probability, r.risk_band);
}

function positionRuler(markEl, probability, band) {
  // The ruler's three zones are sized 50/40/10 (see .ruler-zone flex
  // values in styles.css) to roughly match the real band population
  // shares from models/bands.json - not the probability axis itself,
  // since risk bands are cut by population percentile, not by evenly
  // spaced probability. The mark's position within its zone is
  // approximate for that reason; the band label is the exact fact.
  let leftPct;
  if (band === "Low") leftPct = Math.min(48, (probability / 0.045) * 48);
  else if (band === "Medium") leftPct = 50 + Math.min(38, ((probability - 0.045) / (0.19 - 0.045)) * 38);
  else leftPct = 90 + Math.min(9, ((probability - 0.19) / 0.4) * 9);
  markEl.style.left = `${Math.max(1, Math.min(99, leftPct))}%`;
  markEl.setAttribute("data-label", pct(probability));
}

// =================================================================== Why
$("#explain-this")?.addEventListener("click", () => showSection("why"));

async function renderWhySection() {
  const empty = $("#why-empty");
  const body = $("#why-body");
  if (!state.lastScoreRequest) {
    empty.hidden = false;
    body.hidden = true;
    return;
  }
  empty.hidden = true;
  body.hidden = false;
  body.innerHTML = `<p class="loading">Computing the explanation…</p>`;

  try {
    const r = await api("/api/explain", {
      method: "POST", body: JSON.stringify(state.lastScoreRequest),
    });
    renderWhy(r);
  } catch (e) {
    body.innerHTML = `<p class="loading">Could not explain this score: ${escapeHtml(e.message)}</p>`;
  }
}

function renderWhy(r) {
  const body = $("#why-body");
  const rulerHtml = `
    <div class="result-headline">
      <div class="result-prob">${pct(r.probability)}</div>
      <div class="result-meta">
        <div class="band-pill band-${r.risk_band}">${r.risk_band} risk</div>
      </div>
    </div>
    <div class="ruler-track">
      <div class="ruler-zone low"></div><div class="ruler-zone medium"></div><div class="ruler-zone high"></div>
      <div class="ruler-mark" id="why-ruler-mark"></div>
    </div>`;

  const narrativeHtml = r.narrative
    ? `<div class="narrative">${escapeHtml(r.narrative)}</div>`
    : `<div class="narrative is-unavailable">Plain-language explanation unavailable: ${escapeHtml(r.narrative_error || "no LLM key configured")}. The factors below are exact regardless.</div>`;

  const maxAbs = Math.max(...r.top_factors.map((f) => Math.abs(f.shap_value)), 0.01);
  const factorsHtml = r.top_factors.map((f) => {
    const widthPct = (Math.abs(f.shap_value) / maxAbs) * 48;
    const cls = f.direction === "increases_risk" ? "up" : "down";
    return `
      <div class="factor-row">
        <div class="factor-label">${escapeHtml(f.label)}<div class="factor-value">value: ${escapeHtml(String(f.value))}</div></div>
        <div class="factor-bar-track"><div class="mid"></div><div class="factor-bar ${cls}" style="width:${widthPct}%"></div></div>
        <div class="factor-shap">${f.shap_value > 0 ? "+" : ""}${f.shap_value.toFixed(3)}</div>
      </div>`;
  }).join("");

  body.innerHTML = `
    <div class="ruler-wrap">${rulerHtml}</div>
    ${narrativeHtml}
    <h2>What moved the score</h2>
    <div class="factor-chart">${factorsHtml}</div>`;
  positionRuler($("#why-ruler-mark"), r.probability, r.risk_band);
}

// ================================================================= Rules
let rulesLoaded = false;
async function loadRules() {
  if (rulesLoaded) return;
  const loading = $("#rules-loading");
  const body = $("#rules-body");
  try {
    const r = await api("/api/rules");
    state.rulesBaseRate = r.base_rate;
    $("#fidelity-note").textContent =
      `This simplified tree explains ${(r.fidelity.r2_vs_calibrated_score * 100).toFixed(0)}% ` +
      `of the full model's score variation (R²), and ranks applicants with ${(r.fidelity.roc_auc_vs_actual_target * 100).toFixed(0)}% ` +
      `ROC-AUC against the full model's ${(r.fidelity.full_model_roc_auc_vs_actual_target * 100).toFixed(0)}% — ` +
      `a useful, readable summary, not a replacement for the model.`;
    $("#rule-list").innerHTML = r.rules.map((rule, i) => `
      <li class="rule-item">
        <span class="rule-rank">${i + 1}</span>
        <div class="rule-text">
          <div class="rule-conditions">${escapeHtml(rule.conditions.join(" and ") || "always")}</div>
        </div>
        <div class="rule-figures">
          <div class="rule-rate">${pct(rule.observed_default_rate)}</div>
          <div class="rule-lift">${rule.lift_over_base_rate}x the base rate, ${pct(rule.population_share)} of applicants</div>
        </div>
      </li>`).join("");
    loading.hidden = true;
    body.hidden = false;
    rulesLoaded = true;
  } catch (e) {
    loading.textContent = `Could not load rules: ${e.message}`;
  }
}

// =================================================================== Ask
function initAsk() {
  $("#ask-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const input = $("#ask-input");
    const q = input.value.trim();
    if (!q) return;
    askQuestion(q);
    input.value = "";
  });
  $$(".example-chip").forEach((chip) => {
    chip.addEventListener("click", () => askQuestion(chip.textContent));
  });
}

async function askQuestion(question) {
  const thread = $("#ask-thread");
  const turn = document.createElement("div");
  turn.className = "ask-turn";
  turn.innerHTML = `
    <div class="ask-question">${escapeHtml(question)}</div>
    <div class="ask-loading">Thinking…</div>`;
  thread.prepend(turn);

  try {
    const r = await api("/api/ask", { method: "POST", body: JSON.stringify({ question }) });
    renderAskTurn(turn, question, r);
  } catch (e) {
    turn.innerHTML = `
      <div class="ask-question">${escapeHtml(question)}</div>
      <div class="ask-answer is-error">${escapeHtml(e.message)}</div>`;
  }
}

function renderAskTurn(turn, question, r) {
  const answerClass = r.error ? "is-error" : "";
  const answerText = r.error || r.answer;
  const sqlHtml = r.sql ? `<div class="ask-sql">${escapeHtml(r.sql)}</div>` : "";

  let rowsHtml = "";
  if (r.columns && r.columns.length && r.rows && r.rows.length) {
    const shown = r.rows.slice(0, 20);
    rowsHtml = `<div class="ask-rows-wrap"><table>
      <thead><tr>${r.columns.map((c) => `<th>${escapeHtml(c)}</th>`).join("")}</tr></thead>
      <tbody>${shown.map((row) => `<tr>${row.map((v) => `<td>${escapeHtml(v === null ? "—" : v)}</td>`).join("")}</tr>`).join("")}</tbody>
    </table></div>`;
  }

  const meta = `<div class="ask-meta">
    <span>${r.row_count} row(s)${r.truncated ? " (truncated)" : ""}</span>
    ${r.repaired ? "<span>query needed one correction</span>" : ""}
    <span>${r.total_tokens} tokens</span>
    <span>${r.duration_ms}ms</span>
  </div>`;

  turn.innerHTML = `
    <div class="ask-question">${escapeHtml(question)}</div>
    <div class="ask-answer ${answerClass}">${escapeHtml(answerText)}</div>
    ${sqlHtml}
    ${rowsHtml}
    ${meta}`;
}

// =================================================================== Init
document.addEventListener("DOMContentLoaded", () => {
  initNav();
  initScoreTabs();
  initManualForm();
  initAsk();
  loadHealth();
  loadPortfolio(); // portfolio is the default active section
});
