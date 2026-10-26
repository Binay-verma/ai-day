// Operator console. All server data is inserted with textContent / DOM APIs (never innerHTML),
// so names, notes and explanation text cannot inject markup (CSP also forbids inline script).
"use strict";
const HZ = {fall: "Fall", fire_smoke: "Smoke / fire", crowd: "Crowd", intrusion: "Restricted area", camera_fault: "Camera fault",
            exit_blocked: "Exit unsafe", crowd_toward_hazard: "Crowd heading into danger"};
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
setupSearch();
$("#camsel").addEventListener("change", () => { currentCam = $("#camsel").value; $("#noframe").style.display = ""; renderCams([...cams.values()]); });

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
  // a camera only shows the hazards it analyses (a browser camera has no crowd branch)
  for (const [hz, m] of Object.entries(meters)) m.hidden = !(s.gauges && hz in s.gauges);
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
  cams.clear();
  for (const c of list) if (!c.browser) cams.set(c.id, c);
  const opts = [...cams.values()].map(c => el("option", {value: c.id, textContent: `${c.id} · ${c.zone}`}));
  if (myCam.id) opts.push(el("option", {value: myCam.id, textContent: `My camera · ${myCam.zone}`}));
  sel.replaceChildren(...opts);
  currentCam = keep && (cams.has(keep) || keep === myCam.id) ? keep : (list[0] && list[0].id);
  if (currentCam) sel.value = currentCam;
  const h = $("#camhealth"); h.replaceChildren();
  for (const c of cams.values()) {
    const st = c.health || {}, faults = Object.entries(st.faults || {}).filter(([, v]) => v).map(([k]) => k);
    h.append(el("span", {class: c.paused ? "paused" : (st.state === "live" && !faults.length ? "" : "bad"),
      textContent: `${c.id}: ${c.paused ? "AI paused by " + c.paused_by : (st.state || "?")}${faults.length ? " · " + faults.join(", ") : ""}` +
                   (c.throttled_fps ? ` · slowed to ${c.throttled_fps} FPS while a live camera is analysed` : "")}));
  }
  $("#camctl").hidden = !(role === "operator" || role === "admin") || !currentCam || currentCam === myCam.id;
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
  if (["fall", "fire_smoke", "crowd", "intrusion"].includes(a.hazard)) setupWhatIf(node.querySelector(".whatif"), a);
  else node.querySelector(".whatif").hidden = true;  // site advisories and trouble signals have no score model to re-run
  const acts = node.querySelector(".acts");
  for (const r of a.recommendations || []) acts.append(el("li", {}, el("label", {}, el("input", {type: "checkbox"}), el("span", {class: r.priority === 1 ? "p1" : "", textContent: r.text}))));
  const ver = node.querySelector(".verified");
  const advisory = ["exit_blocked", "crowd_toward_hazard", "camera_fault", "self_test"].includes(a.hazard);
  if (advisory) {
    // site advisories and trouble signals are built from live camera measurements, not from a score model
    node.querySelector(".ledger").hidden = true;
    ver.textContent = "Built from live camera measurements: every number above comes from the camera named.";
  } else {
    ver.textContent = `Explanation checked against the evidence: ${Math.round((v.groundedness ?? 1) * 100)}% of ${v.n_claims ?? 0} claims supported`;
    ver.classList.toggle("bad", (v.groundedness ?? 1) < 1);
  }
  const head = node.querySelector(".head");
  head.addEventListener("click", () => { node.classList.toggle("closed"); head.setAttribute("aria-expanded", !node.classList.contains("closed")); });
  node.querySelector(".report").href = `/api/alerts/${encodeURIComponent(a.id)}/report.html`;
  node.querySelector(".evidence").href = `/api/alerts/${encodeURIComponent(a.id)}/evidence.zip`;
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
  if (!canAct) { node.querySelector(".report").hidden = true; node.querySelector(".evidence").hidden = true; }
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
    if (!full.snapshot_jpg_b64 && !full.evidence_jpg_b64) node.querySelector(".ev").hidden = true;
    if (full.track_record) node.querySelector(".trackrec").textContent = full.track_record.text;
    if (full.has_clip && canAct) { const c = node.querySelector(".clip"); c.hidden = false; c.href = `/api/alerts/${encodeURIComponent(a.id)}/clip.mp4`; }
  }).catch(() => {});
  if (a.status !== "open") node.classList.add("done", "closed");
  return node;
}

// Exit safety (site.exits): clear / congested / unsafe, with the camera evidence
function renderExits(ex) {
  const box = $("#exits");
  if (!box) return;
  const names = Object.keys(ex);
  if (!names.length) return;
  box.replaceChildren(...names.map(n => {
    const e = ex[n];
    return el("div", {class: "exit " + e.state.replace(" ", "-"), title: e.reason || (e.cameras || []).join(", ")},
      el("span", {class: "xname", textContent: n}), el("b", {textContent: e.state}),
      e.reason ? el("span", {class: "small xwhy", textContent: e.reason}) : el("span"));
  }));
}

// Natural-language search of the incident store (deterministic on the server; cites alert ids)
function setupSearch() {
  const form = $("#searchform"), out = $("#searchres");
  if (!form) return;
  form.addEventListener("submit", async ev => {
    ev.preventDefault();
    const q = $("#searchq").value.trim();
    if (!q) { out.hidden = true; return; }
    out.hidden = false; out.replaceChildren(el("div", {class: "small", textContent: "Searching..."}));
    try {
      const r = await api("/api/search", {method: "POST", body: JSON.stringify({q})});
      const chips = el("div", {class: "chips"}, ...r.understood.map(c => el("span", {class: "chip static", textContent: c})));
      const list = el("ul", {class: "sresults"}, ...r.alerts.map(a => el("li", {},
        el("button", {class: "linkish", textContent: `${new Date((a.wall_time || 0) * 1000).toLocaleString()} · ${a.headline || a.hazard} · ${a.status}`,
                      onclick: () => { $("#filter").value = "all"; renderQueue(); const n = document.getElementById("a-" + a.id); if (n) { n.classList.remove("closed"); n.scrollIntoView({behavior: "smooth"}); } }}))));
      const close = el("button", {class: "btn small", textContent: "Close", onclick: () => { out.hidden = true; }});
      out.replaceChildren(el("div", {class: "ssum", textContent: r.summary}), chips, list, close);
    } catch (e) { out.replaceChildren(el("div", {textContent: e.message})); }
  });
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
    else if (m.type === "site") renderExits(m.exits || {});
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

// ---- "Use my camera": this device's camera, analysed on the server exactly like a site camera.
// One frame in flight: the next frame is captured only after the server has answered the previous one,
// so the frame rate adapts to the server's real speed. The annotated, privacy-filtered frame comes back
// on this socket only (nobody else sees the live view); alerts arrive on the main feed like any camera's.
const myCam = {ws: null, stream: null, video: null, canvas: null, id: null, zone: "", prevCam: null,
               inflight: false, lastSend: 0, timer: null, watchdog: null, lastError: "", cfg: {}};
function myCamBar(text, kind = "", showStop = false) {
  const bar = $("#mycambar");
  bar.hidden = !text; bar.className = "mycambar" + (kind ? " " + kind : "");
  $("#mycamtext").textContent = text || ""; $("#mycamstop").hidden = !showStop;
  if (!showStop) $("#mycamtime").textContent = "";
}
const mmss = s => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
function setupMyCam() {
  const b = $("#mycam"), bc = (me && me.browser_camera) || {};
  const can = bc.enabled && window.isSecureContext && navigator.mediaDevices && navigator.mediaDevices.getUserMedia;
  b.hidden = !can;
  if (!can || b.dataset.wired) return;
  b.dataset.wired = "1";
  b.addEventListener("click", () => myCam.ws || myCam.stream ? stopMyCam("Your camera is off.") : startMyCam());
  $("#mycamstop").addEventListener("click", () => stopMyCam("Your camera is off."));
  document.addEventListener("visibilitychange", () => { if (!document.hidden) scheduleFrame(); });
}
async function startMyCam() {
  const bc = me.browser_camera || {};
  myCam.cfg = bc; myCam.lastError = "";
  $("#mycam").setAttribute("aria-pressed", "true"); $("#mycam").textContent = "Stop my camera";
  myCamBar("Allow the camera when your browser asks. Frames are analysed on this server and faces are blurred; only you see your live view.");
  try {
    myCam.stream = await navigator.mediaDevices.getUserMedia({audio: false,
      video: {width: {ideal: 1280}, height: {ideal: 720}, frameRate: {ideal: 15, max: 30}, facingMode: "user"}});
  } catch (e) {
    const why = e.name === "NotAllowedError" ? "Camera permission was refused. Allow the camera for this site (icon in the address bar) and try again."
      : e.name === "NotFoundError" ? "No camera was found on this device."
      : e.name === "NotReadableError" ? "The camera is busy in another app (close Zoom/Teams/Camera) and try again."
      : "Could not open the camera: " + e.message;
    stopMyCam(why, true); return;
  }
  const v = myCam.video = el("video", {muted: true, playsInline: true, autoplay: true});
  v.srcObject = myCam.stream;
  try { await v.play(); } catch (e) { /* autoplay of a muted stream is allowed; frames still arrive */ }
  myCam.canvas = el("canvas");
  myCamBar("Connecting to the analysis server...");
  const ws = myCam.ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws/camera");
  ws.binaryType = "arraybuffer";
  ws.onmessage = e => {
    if (myCam.ws !== ws) return;
    let m; try { m = JSON.parse(e.data); } catch (err) { return; }
    if (m.type === "ready") {
      myCam.id = m.camera; myCam.zone = m.zone || ""; myCam.prevCam = currentCam; currentCam = m.camera;
      myCam.cfg = {...bc, width: m.width || bc.width || 640, max_fps: m.max_fps || bc.max_fps || 10};
      renderCams([...cams.values()]);
      $("#noframe").style.display = ""; $("#noframe").textContent = "Starting your camera...";
      myCamBar(`Your camera is live as a camera in ${myCam.zone}. Faces are ${m.privacy === "raw" ? "NOT blurred (privacy off)" : "blurred"}; ` +
               "only you see this live view. Alerts it raises appear in the alert list for everyone signed in.", "live", true);
      tickRemaining(m.remaining_s); scheduleFrame(0);
    } else if (m.type === "frame") {
      if (currentCam === myCam.id && m.jpg) { $("#frame").src = "data:image/jpeg;base64," + m.jpg; $("#noframe").style.display = "none"; }
      updateState(m.camera, m.state || {});
      tickRemaining(m.remaining_s, (m.state || {}).fps); frameDone();
    } else if (m.type === "skip") frameDone();
    else if (m.type === "error") { myCam.lastError = m.detail || m.code; if (m.code === "bad_frame") frameDone(); }
  };
  ws.onclose = ev => {
    if (myCam.ws !== ws) return;
    const why = myCam.lastError ? `Your camera stopped: ${myCam.lastError}.`
      : ev.code === 1000 ? "Your camera is off." : "The connection to the server was lost; your camera is off.";
    stopMyCam(why, ev.code !== 1000);
  };
}
function tickRemaining(rem, fps) {
  // the fall benchmarks run at 15 FPS; on GMDCSA24 recall is 86 % at 15 FPS, 81 % at ~11 FPS, 76 % at 7.5 FPS
  const t = $("#mycamtime"), slow = fps != null && fps > 0 && fps < 13;
  t.textContent = [fps ? `${fps} FPS${slow ? " (below 15: fast falls can be missed)" : ""}` : "", rem == null ? "" : "ends in " + mmss(rem)]
    .filter(Boolean).join(" · ");
  t.classList.toggle("slow", slow);
  t.title = slow ? "Fall cues were validated at 15 FPS. A fast fall can happen between two frames when the server " +
                   "is busy (fall recall on GMDCSA24: 86 % at 15 FPS, 81 % at ~11 FPS)." : "";
}
function frameDone() { myCam.inflight = false; clearTimeout(myCam.watchdog); scheduleFrame(); }
function scheduleFrame(delay) {
  clearTimeout(myCam.timer);
  if (!myCam.ws || myCam.ws.readyState !== WebSocket.OPEN || myCam.inflight) return;
  const period = 1000 / (myCam.cfg.max_fps || 10);
  const wait = delay != null ? delay : Math.max(0, period - (performance.now() - myCam.lastSend));
  myCam.timer = setTimeout(sendFrame, wait);
}
function sendFrame() {
  const v = myCam.video, ws = myCam.ws;
  if (!v || !ws || ws.readyState !== WebSocket.OPEN || myCam.inflight) return;
  if (!v.videoWidth) { myCam.timer = setTimeout(sendFrame, 100); return; }
  const w = Math.min(myCam.cfg.width || 640, v.videoWidth), h = Math.round(v.videoHeight * w / v.videoWidth);
  const c = myCam.canvas; c.width = w; c.height = h;
  c.getContext("2d").drawImage(v, 0, 0, w, h);
  const t = performance.now() / 1000;   // capture time: keeps speed/descent cues right despite network jitter
  myCam.inflight = true; myCam.lastSend = performance.now();
  myCam.watchdog = setTimeout(() => { myCam.inflight = false; scheduleFrame(); }, 8000);  // a lost reply never stalls
  c.toBlob(async blob => {
    if (!blob || myCam.ws !== ws || ws.readyState !== WebSocket.OPEN) { myCam.inflight = false; return; }
    const jpg = new Uint8Array(await blob.arrayBuffer()), out = new Uint8Array(8 + jpg.length);
    new DataView(out.buffer).setFloat64(0, t, true); out.set(jpg, 8);
    ws.send(out);
  }, "image/jpeg", 0.75);
}
function stopMyCam(message, isError = false) {
  const ws = myCam.ws;
  myCam.ws = null; clearTimeout(myCam.timer); clearTimeout(myCam.watchdog); myCam.inflight = false;
  if (ws && ws.readyState === WebSocket.OPEN) { try { ws.send(JSON.stringify({type: "stop"})); ws.close(1000); } catch (e) {} }
  if (myCam.stream) myCam.stream.getTracks().forEach(tr => tr.stop());
  myCam.stream = null; if (myCam.video) myCam.video.srcObject = null; myCam.video = null;
  const wasMine = currentCam === myCam.id;
  myCam.id = null;
  if (wasMine) { currentCam = myCam.prevCam; $("#noframe").textContent = "Waiting for the camera stream"; $("#noframe").style.display = ""; }
  renderCams([...cams.values()]);
  $("#mycam").setAttribute("aria-pressed", "false"); $("#mycam").textContent = "Use my camera";
  myCamBar(message, isError ? "err" : "");
}
window.addEventListener("pagehide", () => { if (myCam.ws) stopMyCam(""); });

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
  $("#who").textContent = me.auth ? `${me.username} · ${me.role}${me.public_demo ? " · public demo (recorded footage)" : ""}` : "demo mode";
  showMain();
  api("/api/status").then(s => { $("#priv").textContent = s.privacy; }).catch(() => {});
  connect();
  setInterval(() => api("/api/cameras").then(renderCams).catch(() => {}), 10000);
  $("#handover").hidden = !(role === "operator" || role === "admin");
  refreshSelfTest(); setInterval(refreshSelfTest, 60000);
  const site = () => api("/api/site").then(r => renderExits(r.exits || {})).catch(() => {});
  const sd = $("#sitedemo");
  if (sd && (!me.auth || (me.public_demo && role !== "viewer"))) {
    sd.hidden = false;
    sd.addEventListener("click", async () => {
      try { const r = await api("/api/demo/site_scenario", {method: "POST"}); renderExits(r.exits || {}); }
      catch (e) { alert("Scenario failed: " + e.message); }
    });
  }
  site(); setInterval(site, 10000);
  setupMyCam();
}
boot();
