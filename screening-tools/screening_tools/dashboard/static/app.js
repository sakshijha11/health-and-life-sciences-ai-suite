// Copyright © 2026 Intel Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Dashboard client: polls /api/snapshot and draws with the canvas API.
// All data is written with textContent; nothing from the server is parsed as HTML.
"use strict";

const STAGES = [
  ["features", "Ingest + features"],
  ["auto_qc", "Auto-QC + drift"],
  ["hit_scoring", "Hit scoring"],
  ["fusion", "Decision fusion"],
];
const FLAG_NOTE = "high hit score but QC failure, possible false positive";
const COLORS = { blue: "#0068b5", green: "#2e8540", amber: "#c77700", red: "#c0392b", grey: "#b8c2cc", ink: "#1d2733", muted: "#5d6b7a", line: "#dde3ea" };

const $ = (id) => document.getElementById(id);
let refreshMs = 1000;
let lastSeenWall = null;
let finished = false;

function el(tag, text, cls) {
  const e = document.createElement(tag);
  if (text !== undefined && text !== null) e.textContent = String(text);
  if (cls) e.className = cls;
  return e;
}
const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
const ms = (s) => (s && s.p50_ms !== null ? `P50 ${fmt(s.p50_ms, 2)} ms · P95 ${fmt(s.p95_ms, 2)} ms` : "warming up");

function kpis(s) {
  const box = $("kpis");
  box.replaceChildren();
  const items = [
    ["Plates", `${s.plates_done} / ${s.plates_planned}`],
    ["Samples processed", s.samples_out.toLocaleString()],
    ["Throughput (samples/s)", fmt(s.steady_throughput_sps ?? s.throughput_sps, 0)],
    ["End-to-end latency P95", s.latency.end_to_end.p95_ms === null ? "–" : `${fmt(s.latency.end_to_end.p95_ms, 1)} ms`],
    ["Candidates", s.dispositions.candidate],
    ["QC-flagged candidates", s.dispositions.candidate_qc_flagged],
    ["QC events", Object.values(s.event_counts).reduce((a, b) => a + b, 0)],
    ["Elapsed", `${fmt(s.elapsed_s, 0)} s`],
  ];
  for (const [label, value] of items) {
    const k = el("div", null, "kpi");
    k.append(el("div", value, "v"), el("div", label, "l"));
    box.append(k);
  }
}

function pipeline(s) {
  const box = $("pipeline");
  box.replaceChildren();
  const q = (s.series.length ? s.series[s.series.length - 1].queues : {}) || {};
  const queueFor = { features: "buffer", auto_qc: "qc_queue", hit_scoring: "hit_queue", fusion: "fusion_queue" };
  STAGES.forEach(([key, name], i) => {
    if (i) box.append(el("div", "→", "arrow"));
    const st = el("div", null, "stage");
    st.append(el("div", name, "n"), el("div", s.devices[key] || "CPU", "d"), el("div", ms(s.latency[key]), "m"));
    st.append(el("div", `input queue: ${q[queueFor[key]] ?? "–"}`, "m"));
    box.append(st);
  });
  $("backpressure").textContent = `Backpressure events (source waited on a full buffer): ${s.backpressure_events}`;
}

function hits(s) {
  const body = $("hits").tBodies[0];
  body.replaceChildren();
  s.run_hit_list.forEach((r, i) => {
    const flagged = r.disposition === "candidate_qc_flagged";
    const tr = el("tr", null, flagged ? "t2" : "t1");
    tr.append(
      el("td", r.rank_in_run ?? i + 1),
      el("td", r.sample_id),
      el("td", fmt(r.hit_score, 3)),
      el("td", fmt(r.novelty_score, 3)),
      el("td", r.qc_status, r.qc_status === "fail" ? "sev-fail" : r.qc_status === "warn" ? "sev-warn" : ""),
      el("td", flagged ? FLAG_NOTE : r.reason_codes.join(", "), "msg"),
    );
    body.append(tr);
  });
}

function events(s) {
  const body = $("events").tBodies[0];
  body.replaceChildren();
  for (const e of s.recent_events) {
    const tr = el("tr");
    tr.append(el("td", e.reason_code, `sev-${e.severity}`), el("td", e.plate_id), el("td", e.sample_id || "plate"), el("td", e.message, "msg"));
    body.append(tr);
  }
}

function frame(canvas) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.font = "11px Segoe UI, sans-serif";
  return ctx;
}

function lineChart(canvas, series, opts) {
  const ctx = frame(canvas);
  const W = canvas.width, H = canvas.height, L = 40, R = 10, T = 22, B = 22;
  const all = series.flatMap((s) => s.values.filter((v) => v !== null && v !== undefined));
  if (!all.length) { ctx.fillStyle = COLORS.muted; ctx.fillText(opts.empty || "waiting for data", L, H / 2); return; }
  let lo = opts.min ?? Math.min(...all), hi = opts.max ?? Math.max(...all);
  if (hi - lo < 1e-9) { hi += 1; lo -= 1; }
  const n = Math.max(...series.map((s) => s.values.length));
  const x = (i) => L + (n <= 1 ? 0 : (i / (n - 1)) * (W - L - R));
  const y = (v) => T + (1 - (v - lo) / (hi - lo)) * (H - T - B);
  ctx.strokeStyle = COLORS.line; ctx.fillStyle = COLORS.muted;
  for (let k = 0; k <= 4; k++) {
    const v = lo + ((hi - lo) * k) / 4;
    ctx.beginPath(); ctx.moveTo(L, y(v)); ctx.lineTo(W - R, y(v)); ctx.stroke();
    ctx.fillText(fmt(v, opts.digits ?? 0), 2, y(v) + 4);
  }
  if (opts.xlabel) ctx.fillText(opts.xlabel, L, H - 6);
  for (const s of series) {
    ctx.strokeStyle = s.color; ctx.lineWidth = s.width || 1.5; ctx.setLineDash(s.dash || []);
    ctx.beginPath();
    let pen = false;
    s.values.forEach((v, i) => {
      if (v === null || v === undefined) { pen = false; return; }
      if (pen) ctx.lineTo(x(i), y(v)); else ctx.moveTo(x(i), y(v));
      pen = true;
    });
    ctx.stroke();
  }
  ctx.setLineDash([]); ctx.lineWidth = 1;
  let lx = W - R;
  for (const s of [...series].reverse()) {
    if (!s.label) continue;
    const w = ctx.measureText(s.label).width;
    lx -= w + 22;
    ctx.fillStyle = s.color; ctx.fillRect(lx, 6, 10, 3);
    ctx.fillStyle = COLORS.ink; ctx.fillText(s.label, lx + 14, 11);
  }
  if (opts.markers) {
    ctx.fillStyle = COLORS.red;
    for (const [i, label] of opts.markers) { ctx.fillRect(x(i) - 1, T, 2, H - T - B); ctx.fillText(label, x(i) + 3, H - B - 4); }
  }
}

function trend(s) {
  const d = s.drift;
  const g = (f) => d.map(f);
  const markers = [];
  d.forEach((p, i) => { if (i && p.reagent_lot !== d[i - 1].reagent_lot) markers.push([i, p.reagent_lot]); });
  lineChart($("trend"), [
    { label: "negative", color: COLORS.blue, values: g((p) => p.negative_ratio) },
    { label: "positive", color: COLORS.green, values: g((p) => p.positive_ratio) },
    { color: COLORS.blue, dash: [4, 3], width: 1, values: g((p) => p.negative.low) },
    { color: COLORS.blue, dash: [4, 3], width: 1, values: g((p) => p.negative.high) },
    { color: COLORS.green, dash: [4, 3], width: 1, values: g((p) => p.positive.low) },
    { color: COLORS.green, dash: [4, 3], width: 1, values: g((p) => p.positive.high) },
  ], { digits: 2, xlabel: "plate →", markers });
}

function heatmap(s) {
  const canvas = $("heatmap");
  const ctx = frame(canvas);
  const p = s.last_plate;
  if (!p) { ctx.fillStyle = COLORS.muted; ctx.fillText("waiting for the first plate", 40, 150); return; }
  $("plate-title").textContent = `${p.plate_id} · ${p.status} · Z' ${fmt(p.zprime, 2)} · lot ${p.reagent_lot}`;
  const { rows, cols, wells } = p.heatmap;
  const L = 24, T = 16, cw = (canvas.width - L - 4) / cols, ch = (canvas.height - T - 4) / rows;
  ctx.fillStyle = COLORS.muted;
  for (let c = 0; c < cols; c++) ctx.fillText(String(c + 1), L + c * cw + cw / 2 - 4, 11);
  for (let r = 0; r < rows; r++) ctx.fillText(String.fromCharCode(65 + r), 6, T + r * ch + ch / 2 + 4);
  const byWell = new Map(wells.map((w) => [w.well, w]));
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const name = String.fromCharCode(65 + r) + String(c + 1).padStart(2, "0");
      const w = byWell.get(name);
      let color = "#eef1f4";
      if (w) {
        const a = Math.max(0.08, Math.min(1, w.hit_score));
        color = `rgba(0,104,181,${a.toFixed(2)})`;
        if (w.qc_status === "fail" && w.disposition === "non_hit") color = COLORS.red;
        if (w.disposition === "candidate") color = COLORS.green;
        if (w.disposition === "candidate_qc_flagged") color = COLORS.amber;
      }
      ctx.fillStyle = color;
      ctx.fillRect(L + c * cw + 1, T + r * ch + 1, cw - 2, ch - 2);
    }
  }
}

function throughput(s) {
  lineChart($("throughput"), [{ label: "samples/s", color: COLORS.blue, values: s.series.map((p) => p.throughput_sps) }], { min: 0, xlabel: "time →" });
}

function util(s) {
  const lines = [
    { label: "CPU", color: COLORS.blue, values: s.series.map((p) => p.cpu_percent) },
    { label: "iGPU", color: COLORS.green, values: s.series.map((p) => p.gpu_percent) },
  ];
  lineChart($("util"), lines, { min: 0, max: 100, xlabel: "time →" });
  $("gpu-note").textContent = s.gpu_note || "";
}

function plates(s) {
  const canvas = $("plates");
  const ctx = frame(canvas);
  const list = s.plates;
  if (!list.length) return;
  const W = canvas.width, H = canvas.height, L = 40, B = 22, T = 10;
  const bw = (W - L - 10) / Math.max(list.length, 1);
  const maxC = Math.max(1, ...list.map((p) => p.candidates + p.qc_flagged));
  ctx.fillStyle = COLORS.muted; ctx.fillText(`max ${maxC}`, 2, T + 8); ctx.fillText("plate →", L, H - 6);
  list.forEach((p, i) => {
    const x = L + i * bw;
    ctx.fillStyle = p.status === "fail" ? COLORS.red : p.status === "warn" ? COLORS.amber : COLORS.green;
    ctx.fillRect(x, H - B - 6, Math.max(1, bw - 1), 6);
    const h1 = ((H - B - T - 12) * p.candidates) / maxC, h2 = ((H - B - T - 12) * p.qc_flagged) / maxC;
    ctx.fillStyle = COLORS.blue; ctx.fillRect(x, H - B - 8 - h1, Math.max(1, bw - 1), h1);
    ctx.fillStyle = COLORS.amber; ctx.fillRect(x, H - B - 8 - h1 - h2, Math.max(1, bw - 1), h2);
  });
}

function evaluation(s) {
  const e = s.evaluation;
  $("eval-card").hidden = !e;
  if (!e || !e.plates_evaluated) return;
  const box = $("eval");
  box.replaceChildren();
  const grid = el("div", null, "eval-grid");
  const cell = (title, lines) => { const d = el("div"); d.append(el("strong", title)); for (const l of lines) d.append(el("div", l, "note")); grid.append(d); };
  cell("Tier 1 hit list", [`precision ${fmt(e.tier1.precision, 3)}`, `recall ${fmt(e.tier1.recall, 3)}`, `${e.tier1.true_hits} of ${e.tier1.listed} listed are true actives`]);
  cell("Tier 1 + QC-flagged", [`precision ${fmt(e.tier1_and_2.precision, 3)}`, `recall ${fmt(e.tier1_and_2.recall, 3)}`]);
  for (const [name, f] of Object.entries(e.faults)) {
    const lines = Object.entries(f).map(([k, v]) => `${k.replaceAll("_", " ")}: ${v === null ? "–" : v}`);
    cell(name.replaceAll("_", " "), lines);
  }
  const fa = Object.entries(e.events_on_plates_without_fault);
  cell(`Events on ${e.plates_without_fault} fault-free plates`, fa.length ? fa.map(([k, v]) => `${k}: ${v}`) : ["none"]);
  box.append(grid);
}

// UI refresh lag = how old the newest result was when the server sent it
// (server clock only) + how long the browser took to fetch and draw it
// (browser clock only). No clock comparison between machines.
async function reportLag(s, t0) {
  if (s.last_result_wall === null || s.last_result_wall === lastSeenWall) return;
  lastSeenWall = s.last_result_wall;
  const lag = (s.server_time - s.last_result_wall) * 1000 + (performance.now() - t0);
  try {
    await fetch("/api/ui-lag", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ lag_ms: Math.max(0, lag) }) });
  } catch (_) { /* the run may have ended */ }
}

function render(s) {
  $("run-id").textContent = s.run_id;
  const status = $("status");
  status.textContent = s.status;
  status.className = `pill ${s.status}`;
  const done = s.status === "completed" || s.status === "failed";
  $("download").classList.toggle("disabled", !done);
  $("stop").disabled = done;
  kpis(s); pipeline(s); hits(s); events(s); trend(s); heatmap(s); throughput(s); util(s); plates(s); evaluation(s);
  const err = $("error");
  err.hidden = !s.errors.length;
  err.textContent = s.errors.join("; ");
  return done;
}

async function tick() {
  try {
    const t0 = performance.now();
    const r = await fetch("/api/snapshot", { cache: "no-store" });
    const s = await r.json();
    refreshMs = s.refresh_ms || refreshMs;
    const wasFinished = finished;
    finished = render(s);
    if (!wasFinished) reportLag(s, t0);
  } catch (e) {
    $("error").hidden = false;
    $("error").textContent = "The app is not running (dashboard server unreachable).";
  }
  setTimeout(tick, refreshMs);
}

$("stop").addEventListener("click", () => {
  fetch("/api/stop", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
});
tick();
