// Operator console. All server data is inserted with textContent / DOM APIs (never innerHTML),
// so names, notes and explanation text cannot inject markup (CSP also forbids inline script).
"use strict";
const HZ = {fall: "Fall", fire_smoke: "Smoke / fire", crowd: "Crowd", intrusion: "Restricted area", camera_fault: "Camera fault"};
const $ = s => document.querySelector(s);
const el = (tag, props = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") e.className = v; else if (k === "style") e.style.cssText = v;
    else if (k.startsWith("data-") || k.startsWith("aria-")) e.setAttribute(k, v); else e[k] = v;
  }
  for (const k of kids) if (k != null) e.append(k);
  return e;
};
const api = async (path, opts = {}) => {
  const r = await fetch(path, {credentials: "same-origin", headers: {"Content-Type": "application/json"}, ...opts});
  if (r.status === 401) { showLogin(); throw new Error("login required"); }
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
};
let me = null, ws = null, currentCam = null, role = "viewer";
const meters = {}, alerts = new Map(), cams = new Map();

function showLogin() {
  $("#loginview").hidden = false; $("#mainview").hidden = true; $("#logout").hidden = true;
  if (ws) { ws.onclose = null; ws.close(); ws = null; }
}
function showMain() { $("#loginview").hidden = true; $("#mainview").hidden = false; $("#logout").hidden = !me || !me.auth; }

$("#loginform").addEventListener("submit", async ev => {
  ev.preventDefault();
  $("#loginerr").textContent = "";
  try {
    await api("/api/login", {method: "POST", body: JSON.stringify({username: $("#lu").value, password: $("#lp").value})});
    $("#lp").value = "";
    boot();
  } catch (e) { $("#loginerr").textContent = "Sign-in failed: " + e.message; }
});
$("#logout").addEventListener("click", async () => { try { await api("/api/logout", {method: "POST"}); } catch (e) {} showLogin(); });
$("#filter").addEventListener("change", () => renderQueue());
$("#camsel").addEventListener("change", () => { currentCam = $("#camsel").value; $("#noframe").style.display = ""; });

function meter(hz) {
  if (meters[hz]) return meters[hz];
  const d = el("div", {class: "meter"},
    el("span", {class: "name", textContent: HZ[hz] || hz}),
    el("div", {class: "bar"}, el("div", {class: "fill", style: `background:var(--${hz in HZ ? hz : "for"})`}), el("div", {class: "tick"})),
    el("span", {class: "val", textContent: "0%"}), el("span", {class: "why"}));
  $("#meters").append(d); meters[hz] = d; return d;
}

function updateState(cam, s) {
  if (cam !== currentCam) return;
  $("#fps").textContent = s.fps ?? "–"; $("#ppl").textContent = s.people ?? "–";
  for (const [hz, g] of Object.entries(s.gauges || {})) {
    const m = meter(hz); m.classList.toggle("active", !!g.active);
    m.querySelector(".fill").style.width = (g.prob * 100).toFixed(1) + "%";
    m.querySelector(".tick").style.left = (g.threshold * 100) + "%";
    m.querySelector(".val").textContent = Math.round(g.prob * 100) + "%";
    const pos = (g.top || []).filter(c => c.c > 0 && !c.name.endsWith("_gate")).map(c => c.name.replace(/_/g, " "));
    let why = pos.length ? "pushed up by " + pos.join(", ") + (g.subject ? ` (${g.subject})` : "") : "nothing unusual";
    if (g.gated && g.gate_phrase) why += " - held below the line: " + g.gate_phrase;
    m.querySelector(".why").textContent = why;
  }
  const t = $("#timings"); t.replaceChildren();
  for (const [k, v] of Object.entries(s.timings || {})) t.append(el("span", {}, k + " ", el("b", {textContent: v})));
  const sens = $("#sensors"), items = [];
  for (const [z, f] of Object.entries(s.sensors || {})) for (const [k, v] of Object.entries(f))
    items.push(el("div", {class: "sensor" + (v.online ? "" : " stale")},
      el("div", {class: "k", textContent: `${z} · ${k.replace(/_/g, " ")}${v.online ? "" : " (offline)"}`}),
      el("div", {class: "v", textContent: (+v.value).toFixed(k === "temp_c" ? 1 : 0)})));
  if (items.length) sens.replaceChildren(...items);
}

function renderCams(list) {
  const sel = $("#camsel"), keep = currentCam;
  for (const c of list) cams.set(c.id, c);
  sel.replaceChildren(...[...cams.values()].map(c => el("option", {value: c.id, textContent: `${c.id} · ${c.zone}`})));
  currentCam = keep && cams.has(keep) ? keep : (list[0] && list[0].id);
  if (currentCam) sel.value = currentCam;
  const h = $("#camhealth"); h.replaceChildren();
  for (const c of cams.values()) {
    const st = c.health || {}, faults = Object.entries(st.faults || {}).filter(([, v]) => v).map(([k]) => k);
    h.append(el("span", {class: c.paused ? "paused" : (st.state === "live" && !faults.length ? "" : "bad"),
      textContent: `${c.id}: ${c.paused ? "AI paused by " + c.paused_by : (st.state || "?")}${faults.length ? " · " + faults.join(", ") : ""}`}));
  }
  $("#camctl").hidden = !(role === "operator" || role === "admin") || !currentCam;
}

function ledger(a) {
  const cues = a.cues || [], wrap = el("div");
  const thrP = Math.min(0.999, Math.max(0.001, parseInt(((a.explanation || {}).confidence_text || "").match(/threshold (\d+)%/)?.[1] || "60") / 100));
  const thr = Math.log(thrP / (1 - thrP)), p = Math.min(0.999999, Math.max(1e-6, a.prob));
  const bias = Math.log(p / (1 - p)) - cues.reduce((s, c) => s + c.contribution, 0);
  let x = bias, lo = Math.min(bias, thr), hi = Math.max(bias, thr); const steps = [];
  for (const c of cues) { const nx = x + c.contribution; steps.push([c, x, nx]); x = nx; lo = Math.min(lo, x); hi = Math.max(hi, x); }
  lo -= .5; hi += .5;
  const P = v => ((v - lo) / (hi - lo) * 100).toFixed(2) + "%", W = (a0, b0) => (Math.abs(b0 - a0) / (hi - lo) * 100).toFixed(2) + "%";
  const row = (label, cls, a0, b0) => el("div", {class: "row"}, el("span", {class: "lbl", textContent: label}),
    el("div", {class: "track"}, el("div", {class: "seg " + cls, style: `left:${P(Math.min(a0, b0))};width:${W(a0, b0)}`}), el("div", {class: "thr", style: `left:${P(thr)}`})));
  wrap.append(row("Starting point (normal for this zone)", "base", lo + .5, bias));
  for (const [c, a0, b0] of steps) {
    if (Math.abs(c.contribution) < 0.05) continue;
    const label = `${c.phrase}  ${c.contribution > 0 ? "+" : ""}${c.contribution.toFixed(2)}${c.reliability < 1 ? ` · reliability ${Math.round(c.reliability * 100)}%` : ""}`;
    wrap.append(row(label, c.contribution > 0 ? "for" : "against", a0, b0));
  }
  wrap.append(el("div", {class: "total", textContent: `Final score ${Math.round(a.prob * 100)}% against an alert line of ${Math.round(thrP * 100)}% (dashed). Bars to the right raised the risk, bars to the left lowered it.`}));
  return wrap;
}

// "What if?": the server re-scores the alert exactly (additive ledger) with the slider values.
function setupWhatIf(det, a) {
  if (!det || a.hazard === "camera_fault") { if (det) det.hidden = true; return; }
  const body = det.querySelector(".wi-body");
  let loaded = false, timer = null;
  const values = {};
  const out = el("div", {class: "wi-out", "aria-live": "polite"});
  const fmt = v => Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2);
  const render = r => {
    const pct = x => Math.round(x * 100) + "%";
    out.textContent = `Score ${pct(r.prob)} (was ${pct(r.prob_before)}) against an alert line of ${pct(r.threshold)}: ` +
      (r.would_alert ? "this would still raise an alert." : "this would NOT raise an alert.");
    out.className = "wi-out " + (r.would_alert ? "yes" : "no");
  };
  const query = async () => {
    try { render(await api(`/api/alerts/${encodeURIComponent(a.id)}/whatif`, {method: "POST", body: JSON.stringify({values})})); }
    catch (e) { out.textContent = e.message; }
  };
  det.addEventListener("toggle", async () => {
    if (!det.open || loaded) return;
    loaded = true;
    try {
      const r = await api(`/api/alerts/${encodeURIComponent(a.id)}/whatif`, {method: "POST", body: JSON.stringify({values: {}})});
      for (const c of r.cues) {
        const lab = el("span", {class: "wi-val", textContent: `${fmt(c.value)} ${c.unit || ""}`});
        const step = (c.hi - c.lo) / 200;
        const sl = el("input", {type: "range", min: c.lo, max: c.hi, step, value: c.value, "aria-label": c.name.replace(/_/g, " ")});
        sl.addEventListener("input", () => {
          values[c.name] = parseFloat(sl.value);
          lab.textContent = `${fmt(values[c.name])} ${c.unit || ""}`;
          clearTimeout(timer); timer = setTimeout(query, 120);
        });
        body.append(el("label", {class: "wi-row"}, el("span", {class: "wi-name", textContent: c.name.replace(/_/g, " ")}), sl, lab));
      }
      const reset = el("button", {class: "btn small", textContent: "Reset", onclick: () => {
        for (const k of Object.keys(values)) delete values[k];
        body.querySelectorAll("input[type=range]").forEach((s, i) => { s.value = r.cues[i].value; s.dispatchEvent(new Event("input")); });
      }});
      body.append(out, reset);
      render(r);
    } catch (e) { body.append(el("div", {textContent: e.message})); }
  });
}

function card(a) {
  const node = $("#alerttpl").content.firstElementChild.cloneNode(true);
  node.id = "a-" + a.id;
  node.style.setProperty("--c", `var(--${a.hazard in HZ ? a.hazard : "for"})`);
  const ex = a.explanation || {}, v = a.verification || {};
  node.querySelector(".title").textContent = ex.headline || a.hazard;
  node.querySelector(".sev").textContent = a.severity || "";
  node.querySelector(".meta").textContent = `${new Date((a.wall_time || 0) * 1000).toLocaleTimeString()} · ${a.camera_id || ""} · ${a.status}${a.operator ? " by " + a.operator : ""}`;
  node.querySelector(".because").textContent = ex.because || "";
  const tri = typeof a.triage === "string" ? a.triage : (a.triage || {}).mode;
  const tr = node.querySelector(".triage"); tr.textContent = tri === "VERIFY" ? "Please verify before escalating" : (tri === "ALARM" ? "Raise the alarm" : "");
  tr.classList.add(tri === "VERIFY" ? "verify" : "alarm");
  node.querySelector(".ledger").append(ledger(a));
  node.querySelector(".cf").textContent = ex.counterfactual || "";
  setupWhatIf(node.querySelector(".whatif"), a);
  const acts = node.querySelector(".acts");
  for (const r of a.recommendations || []) acts.append(el("li", {}, el("label", {}, el("input", {type: "checkbox"}), el("span", {class: r.priority === 1 ? "p1" : "", textContent: r.text}))));
  const ver = node.querySelector(".verified");
  ver.textContent = `Explanation checked against the evidence: ${Math.round((v.groundedness ?? 1) * 100)}% of ${v.n_claims ?? 0} claims supported`;
  ver.classList.toggle("bad", (v.groundedness ?? 1) < 1);
  const head = node.querySelector(".head");
  head.addEventListener("click", () => { node.classList.toggle("closed"); head.setAttribute("aria-expanded", !node.classList.contains("closed")); });
  node.querySelector(".report").href = `/api/alerts/${encodeURIComponent(a.id)}/report.html`;
  const canAct = role === "operator" || role === "admin";
  node.querySelectorAll("[data-c]").forEach(b => {
    b.disabled = !canAct || a.status !== "open";
    b.addEventListener("click", async () => {
      try {
        const r = await api(`/api/alerts/${encodeURIComponent(a.id)}/feedback`, {method: "POST", body: JSON.stringify({confirmed: b.dataset.c === "1"})});
        node.querySelectorAll("[data-c]").forEach(x => x.disabled = true); node.classList.add("done");
        node.querySelector(".answer").textContent = r.status === "confirmed" ? "Confirmed. Logged in the audit trail." :
          `Dismissed. ${HZ[a.hazard] || a.hazard} sensitivity for this zone adjusted${r.bias_offset != null ? ` (offset ${r.bias_offset.toFixed(2)} log-odds, capped)` : ""}.`;
      } catch (e) { node.querySelector(".answer").textContent = "Could not record: " + e.message; }
    });
  });
  if (!canAct) node.querySelector(".report").hidden = true;
  const ask = async q => {
    if (!q) return;
    try { node.querySelector(".answer").textContent = (await api(`/api/alerts/${encodeURIComponent(a.id)}/ask`, {method: "POST", body: JSON.stringify({question: q})})).answer; }
    catch (e) { node.querySelector(".answer").textContent = e.message; }
  };
  const inp = node.querySelector(".ask input");
  node.querySelector(".askbtn").addEventListener("click", () => ask(inp.value));
  inp.addEventListener("keydown", e => { if (e.key === "Enter") ask(inp.value); });
  for (const q of ["Why?", "What would change it?", "How sure are you?", "What should I do?"])
    node.querySelector(".chips").append(el("button", {class: "chip", textContent: q, onclick: () => ask(q)}));
  api(`/api/alerts/${encodeURIComponent(a.id)}`).then(full => {
    if (full.snapshot_jpg_b64) node.querySelector(".snap").src = "data:image/jpeg;base64," + full.snapshot_jpg_b64;
    if (full.evidence_jpg_b64) node.querySelector(".evid").src = "data:image/jpeg;base64," + full.evidence_jpg_b64;
    if (full.has_clip && canAct) { const c = node.querySelector(".clip"); c.hidden = false; c.href = `/api/alerts/${encodeURIComponent(a.id)}/clip.mp4`; }
  }).catch(() => {});
  if (a.status !== "open") node.classList.add("done", "closed");
  return node;
}

function renderQueue() {
  const q = $("#queue"), showAll = $("#filter").value === "all";
  const list = [...alerts.values()].filter(a => showAll || a.status === "open").sort((x, y) => (y.wall_time || 0) - (x.wall_time || 0));
  if (!list.length) { q.replaceChildren(el("div", {class: "qempty", textContent: showAll ? "No alerts yet." : "No open alerts."})); return; }
  q.replaceChildren(...list.slice(0, 100).map(a => a._node || (a._node = card(a))));
}

function upsertAlert(a, fresh) {
  const old = alerts.get(a.id);
  if (old && old._node) old._node.remove();
  alerts.set(a.id, a);
  renderQueue();
  if (fresh && a.hazard === "camera_fault") api("/api/cameras").then(renderCams).catch(() => {});
  if (fresh && a._node) { document.querySelectorAll(".alert:not(.closed)").forEach(x => { if (x !== a._node) x.classList.add("closed"); }); }
}

// ---- audible alarm (IEC 62676 / FM 3232: alarm and trouble must be distinguishable and audible)
// Web Audio only (no files). Browsers allow sound only after a user gesture, hence the button.
let audio = null, soundOn = false, ringTimer = null;
function tone(freq, start, dur, type = "square", gain = 0.08) {
  const o = audio.createOscillator(), g = audio.createGain();
  o.type = type; o.frequency.value = freq; g.gain.value = gain;
  o.connect(g); g.connect(audio.destination);
  o.start(audio.currentTime + start); o.stop(audio.currentTime + start + dur);
}
function pattern(kind) {
  if (!soundOn || !audio) return;
  if (kind === "alarm") { for (let i = 0; i < 4; i++) { tone(960, i * 0.5, 0.25); tone(720, i * 0.5 + 0.25, 0.25); } }
  else if (kind === "fault") { for (let i = 0; i < 3; i++) tone(330, i * 0.35, 0.2, "sawtooth", 0.06); }
  else tone(880, 0, 0.35, "sine", 0.1);  // VERIFY: single chime
}
function ringing() {
  // keep ringing while any ALARM-mode hazard alert is open and unacknowledged
  return [...alerts.values()].some(a => a.status === "open" && a.hazard !== "camera_fault" && (a.triage || {}).mode === "ALARM");
}
function updateRing() {
  if (soundOn && ringing()) { if (!ringTimer) { pattern("alarm"); ringTimer = setInterval(() => ringing() ? pattern("alarm") : updateRing(), 4000); } }
  else if (ringTimer) { clearInterval(ringTimer); ringTimer = null; }
}
function announce(a) {
  if (a.hazard === "camera_fault") pattern("fault");
  else if ((a.triage || {}).mode !== "ALARM") pattern("verify");
  updateRing();
}
$("#sound").addEventListener("click", () => {
  soundOn = !soundOn;
  if (soundOn && !audio) { try { audio = new (window.AudioContext || window.webkitAudioContext)(); } catch (e) { soundOn = false; } }
  $("#sound").textContent = soundOn ? "Sound on" : "Sound off";
  $("#sound").setAttribute("aria-pressed", String(soundOn));
  $("#sound").classList.toggle("on", soundOn);
  api("/api/console/sound", {method: "POST", body: JSON.stringify({on: soundOn})}).catch(() => {});
  if (soundOn) pattern("verify");
  updateRing();
});

function connect() {
  ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws");
  ws.onopen = () => { $("#conn").textContent = "Live"; $("#conn").className = "pill live"; };
  ws.onclose = ev => {
    $("#conn").textContent = "Offline, retrying"; $("#conn").className = "pill off";
    if (ev.code === 4401) { showLogin(); return; }
    setTimeout(connect, 2000);
  };
  ws.onmessage = e => {
    const m = JSON.parse(e.data);
    if (m.type === "hello") { renderCams(m.cameras || []); (m.alerts || []).reverse().forEach(a => upsertAlert(a, false)); }
    else if (m.type === "frame") {
      if (m.camera === currentCam) { $("#frame").src = "data:image/jpeg;base64," + m.jpg; $("#noframe").style.display = "none"; }
      updateState(m.camera, m.state);
    }
    else if (m.type === "alert") { upsertAlert(m.alert, true); announce(m.alert); }
    else if (m.type === "alert_update") { const a = alerts.get(m.id); if (a) { a.status = m.status; a.operator = m.operator; a._node = null; renderQueue(); } updateRing(); }
    else if (["camera_fault", "camera_paused", "camera_resumed"].includes(m.type)) api("/api/cameras").then(renderCams).catch(() => {});
  };
}

$("#pausebtn").addEventListener("click", async () => {
  try {
    await api(`/api/cameras/${encodeURIComponent(currentCam)}/pause`, {method: "POST",
      body: JSON.stringify({minutes: +$("#pausemin").value, reason: $("#pausereason").value})});
    renderCams(await api("/api/cameras"));
  } catch (e) { alert("Could not pause: " + e.message); }
});
$("#resumebtn").addEventListener("click", async () => {
  try { await api(`/api/cameras/${encodeURIComponent(currentCam)}/resume`, {method: "POST"}); renderCams(await api("/api/cameras")); }
  catch (e) { alert("Could not resume: " + e.message); }
});

async function refreshSelfTest() {
  try {
    const r = await api("/api/selftest");
    const p = $("#selftest");
    p.textContent = r.ok === true ? "Self-test OK" : (r.ok === false ? "Self-test FAILED" : "Self-test –");
    p.className = "pill " + (r.ok === true ? "live" : (r.ok === false ? "off" : ""));
    if (r.time_utc) p.title = "Last self-test " + r.time_utc;
  } catch (e) { /* not logged in yet */ }
}

async function boot() {
  try { me = await api("/api/me"); } catch (e) { return; }
  role = me.role;
  $("#who").textContent = me.auth ? `${me.username} · ${me.role}` : "demo mode";
  showMain();
  api("/api/status").then(s => { $("#priv").textContent = s.privacy; }).catch(() => {});
  connect();
  setInterval(() => api("/api/cameras").then(renderCams).catch(() => {}), 10000);
  $("#handover").hidden = !(role === "operator" || role === "admin");
  refreshSelfTest(); setInterval(refreshSelfTest, 60000);
}
boot();
