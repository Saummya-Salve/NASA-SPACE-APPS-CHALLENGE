// SPHEREx Sky Time-Lapse Explorer - front end
const $ = (id) => document.getElementById(id);

const state = { target: "barnard", ra: null, dec: null, band: "D1", size: 0.05, mode: "movie", speed: 600, offline: false };
let targets = [];
let manifest = null, idx = 0, timer = null, playing = true;
let aladin = null, specChart = null, lcChart = null;
let zoom = 1, panX = 0, panY = 0, dragging = false, dragStart = null;
let scanSources = [];
const SPEED_MIN_MS = 150, SPEED_MAX_MS = 1500;

// ---------------------------------------------------------------- helpers
async function api(path) {
  const r = await fetch(path);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || `${r.status} ${r.statusText}`);
  return body;
}
function status(msg, isError = false) {
  const s = $("status");
  if (!msg) { s.classList.add("hidden"); return; }
  s.textContent = msg;
  s.classList.toggle("error", isError);
  s.classList.remove("hidden");
  s.classList.toggle("loading", !isError);
}
function targetQuery() {
  const p = new URLSearchParams();
  if (state.target) p.set("target", state.target);
  else { p.set("ra", state.ra); p.set("dec", state.dec); }
  return p;
}

// ---------------------------------------------------------------- URL hash (shareable links)
function readHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  if (p.get("ra") && p.get("dec") && !p.get("target")) {
    state.target = null; state.ra = parseFloat(p.get("ra")); state.dec = parseFloat(p.get("dec"));
  }
  if (p.get("target")) state.target = p.get("target");
  for (const k of ["band", "mode"]) if (p.get(k)) state[k] = p.get(k);
  for (const k of ["size", "speed"]) if (p.get(k)) state[k] = parseFloat(p.get(k));
  state.offline = p.get("offline") === "1";
}
function writeHash() {
  const p = targetQuery();
  p.set("band", state.band); p.set("size", state.size); p.set("mode", state.mode); p.set("speed", state.speed);
  if (state.offline) p.set("offline", "1"); else p.delete("offline");
  history.replaceState(null, "", "#" + p.toString());
}

// ---------------------------------------------------------------- presets
async function loadPresets() {
  targets = await api("/api/targets");
  const box = $("presets");
  box.innerHTML = "";
  for (const t of targets) {
    const b = document.createElement("button");
    b.textContent = t.name;
    b.dataset.key = t.key;
    b.onclick = () => { state.target = t.key; state.band = t.best_band; state.size = 0.05; syncControls(); load(); };
    box.appendChild(b);
  }
}
function labelDetectors() {
  const t = targets.find(x => x.key === state.target);
  document.querySelectorAll("#band option").forEach(o => {
    if (!o.dataset.base) o.dataset.base = o.textContent;
    let avail = true;
    if (t && Array.isArray(t.data_bands)) {
      avail = o.value === "RGB" ? ["D1", "D3", "D6"].every(b => t.data_bands.includes(b))
                                : t.data_bands.includes(o.value);
    }
    o.textContent = o.dataset.base + (avail ? "" : " · needs live download");
  });
  // Field of view: built-in targets are pre-downloaded at the standard size only
  document.querySelectorAll("#size option").forEach(o => {
    if (!o.dataset.base) o.dataset.base = o.textContent;
    const size = parseFloat(o.value);
    let avail = true;
    if (t && Math.abs(size - 0.05) > 1e-9) {
      avail = Array.isArray(t.cached_views) && t.cached_views.some(v => Math.abs(v.size_deg - size) < 1e-9);
    }
    o.textContent = o.dataset.base + (avail ? "" : " · needs live download");
  });
}
function syncControls() {
  labelDetectors();
  $("band").value = state.band;
  $("size").value = String(state.size);
  // The control reads left-to-right as slower-to-faster; state.speed remains the frame delay in ms.
  $("speed").value = SPEED_MIN_MS + SPEED_MAX_MS - state.speed;
  $("offline").checked = state.offline;
  document.querySelectorAll("#presets button").forEach(b => b.classList.toggle("active", b.dataset.key === state.target));
  document.querySelectorAll(".mode").forEach(b => b.classList.toggle("active", b.dataset.mode === state.mode));
}

// ---------------------------------------------------------------- main load
async function load() {
  stop();
  writeHash();
  $("frame").style.visibility = "hidden";
  $("empty-state").classList.remove("hidden");
  $("empty-title").textContent = state.offline ? "Looking in saved data" : "Loading your sky view";
  $("empty-message").textContent = state.offline ? "Searching this server’s cache for a matching observation…" : "Fetching and aligning SPHEREx observations…";
  resetZoom();
  $("frameinfo").textContent = "Loading…";
  const slow = state.size > 0.05 ? " Larger fields of view need new, bigger downloads." : "";
  status(state.offline ? "Loading saved SPHEREx observations…" : "Fetching SPHEREx images live from NASA IRSA and aligning them… a new position or field of view can take 1–3 minutes (longer if NASA's archive is busy)." + slow);
  const q = targetQuery(); q.set("band", state.band); q.set("size", state.size); q.set("fetch", String(!state.offline));
  try {
    manifest = await api("/api/frames?" + q.toString());
  } catch (e) {
    // clear panels from the previous view so nothing stale is shown
    ["motioncard", "moverscard", "lccard"].forEach(id => $(id).classList.add("hidden"));
    $("markers").innerHTML = ""; $("gaiaoverlay").innerHTML = ""; $("gaiainfo").textContent = "";
    if (specChart) { specChart.destroy(); specChart = null; }
    $("speclabel").textContent = "";
    $("empty-title").textContent = state.offline ? "No saved observation found" : "Could not load this sky view";
    $("empty-message").textContent = state.offline ? "Turn off ‘Use saved data only’ and press Go to fetch this position." : e.message;
    status("Could not build frames: " + e.message, true);
    return;
  }
  status("");
  $("frame").style.visibility = "visible";
  $("empty-state").classList.add("hidden");
  const t = manifest.target;
  $("ra").value = t.ra.toFixed(5);
  $("dec").value = t.dec.toFixed(5);
  $("tname").textContent = t.name || `RA ${t.ra.toFixed(4)}, Dec ${t.dec.toFixed(4)}`;
  $("tdesc").textContent = t.description || "Custom position. Look for anything that moves or changes brightness between frames.";
  const frameOrigin = manifest.cache_fallback ? "saved preview · IRSA unavailable" :
    (state.offline || manifest.cached_frame ? "saved on this server" : "NASA/IPAC IRSA");
  $("freshness").textContent = `${manifest.n_images_total} SPHEREx images in ${manifest.n_epochs} epochs · latest image ${manifest.latest_image_date} · ${frameOrigin}`;
  $("scrub").max = manifest.frames.length - 1;
  $("wiseview").href = `http://byw.tools/wiseview#ra=${t.ra}&dec=${t.dec}&size=176&band=3&speed=500&minbright=-50&maxbright=500&window=0.5&diff_window=1&linear=1&zoom=9&gaia=1`;
  idx = 0;
  renderGallery();
  $("rgblegend").classList.toggle("hidden", state.band !== "RGB");
  setMode(state.mode);
  updateMap(t.ra, t.dec);

  loadGaiaOverlay();

  // science panels
  $("markers").innerHTML = "";
  scanSources = [];
  $("moverscard").classList.add("hidden");
  $("motioncard").classList.add("hidden");
  $("lccard").classList.add("hidden");
  $("speclabel").textContent = "";
  if (specChart) { specChart.destroy(); specChart = null; }
  $("specbtn").classList.toggle("hidden", !state.target);
  if (state.target && t.kind === "motion") loadMotion();
  if (state.target) loadCachedSpectrum();
  if (state.target && t.kind === "variable") loadLightcurve();
}

// ---------------------------------------------------------------- Gaia overlay (no CDS needed)
async function loadGaiaOverlay() {
  const box = $("gaiaoverlay");
  box.innerHTML = "";
  $("gaiainfo").textContent = "";
  if (!$("showgaia").checked || !manifest) return;
  const q = targetQuery(); q.set("band", state.band); q.set("size", state.size); q.set("fetch", "false");
  const forLabel = manifest.target.label;
  $("gaiainfo").textContent = "Loading Gaia DR3 stars…";
  try {
    const g = await api("/api/gaia?" + q.toString());
    if (!manifest || manifest.target.label !== forLabel) return;  // user moved on meanwhile
    const [pw, ph] = g.png_size;
    box.innerHTML = g.stars.map(st =>
      `<span class="gx${st.gmag !== null && st.gmag > 18 ? " faint" : ""}" style="left:${st.x_png / pw * 100}%;top:${st.y_png / ph * 100}%"
             title="Gaia DR3 ${st.source_id}${st.gmag !== null ? " · G=" + st.gmag : ""}${st.pm_total !== null ? " · " + st.pm_total + "″/yr" : ""}"></span>`).join("");
    $("gaiainfo").textContent = `✚ ${g.n_stars} Gaia DR3 stars in this field (moved to the SPHEREx date; via ${g.service}). ` +
      "A SPHEREx source with no cross is not in Gaia: often a cold, red object such as a brown dwarf.";
  } catch (e) {
    $("gaiainfo").textContent = "Gaia overlay unavailable right now: " + e.message;
  }
}

// ---------------------------------------------------------------- player
function setMode(mode) {
  state.mode = mode;
  syncControls();
  writeHash();
  stop();
  if (!manifest) return;
  const img = $("frame");
  if (mode === "movie") { playing = true; showFrame(idx); start(); }
  else if (mode === "blink") { playing = true; idx = 0; showFrame(0); start(); }
  else if (mode === "diff") { img.src = manifest.diff_url; $("frameinfo").textContent = `Difference: ${last().date} minus ${manifest.frames[0].date} — movers appear as a red/blue pair (red = arrived, blue = left)`; }
  else if (mode === "deep") { img.src = manifest.deep_url; $("frameinfo").textContent = `Deep: all ${manifest.n_images_total} images added together — best for faint objects`; }
  $("play").textContent = playing && (mode === "movie" || mode === "blink") ? "⏸" : "▶";
}
function last() { return manifest.frames[manifest.frames.length - 1]; }
function showFrame(i) {
  const f = manifest.frames[i];
  $("frame").src = f.url;
  $("scrub").value = i;
  const sat = f.missing_fraction > 0 ? ` · saturated/missing pixels ${(f.missing_fraction * 100).toFixed(2)}%` : "";
  $("frameinfo").textContent = `Epoch ${f.epoch}/${manifest.frames.length} · ${f.date} · ${f.band} · ${f.n_images} images stacked${sat}`;
  document.querySelectorAll(".epoch-card").forEach((el, n) => el.classList.toggle("selected", n === i));
}
function renderGallery() {
  const box = $("frame-gallery");
  if (!manifest?.frames?.length) { $("gallery-count").textContent = "No frames loaded"; box.innerHTML = '<p class="gallery-empty">Load a target to browse its individual observations.</p>'; return; }
  $("gallery-count").textContent = `${manifest.frames.length} ${manifest.frames.length === 1 ? "epoch" : "epochs"}`;
  box.innerHTML = manifest.frames.map((f, i) => `<button class="epoch-card${i === idx ? " selected" : ""}" type="button" data-epoch="${i}" aria-label="Show epoch ${i + 1}, ${f.date}"><img src="${f.url}" alt="" loading="lazy"><span class="epoch-date">${f.date}</span><span class="epoch-meta">Epoch ${f.epoch} · ${f.n_images} images</span></button>`).join("");
  box.querySelectorAll(".epoch-card").forEach(b => b.onclick = () => { idx = Number(b.dataset.epoch); state.mode = "movie"; syncControls(); writeHash(); stop(); playing = false; $("play").textContent = "▶"; showFrame(idx); });
}
function tick() {
  if (state.mode === "movie") idx = (idx + 1) % manifest.frames.length;
  else if (state.mode === "blink") idx = idx === 0 ? manifest.frames.length - 1 : 0;
  showFrame(idx);
}
function start() { stop(); if (playing) timer = setInterval(tick, state.speed); }
function stop() { if (timer) clearInterval(timer); timer = null; }

// ---------------------------------------------------------------- click a star -> spectrum
function pixelToSky(ev) {
  const img = $("frame"), g = manifest.grid;
  const rect = img.getBoundingClientRect();
  // the image is square and fills the box ("contain"), so map click -> natural pixel
  const px = (ev.clientX - rect.left) / rect.width * img.naturalWidth;
  const py = (ev.clientY - rect.top) / rect.height * img.naturalHeight;
  const dx = px / g.upscale - (g.nx - 1) / 2;
  const dy = (g.ny - 1) / 2 - py / g.upscale;
  const ra = g.ra0 - dx * g.scale_arcsec / 3600 / Math.cos(g.dec0 * Math.PI / 180);
  const dec = g.dec0 + dy * g.scale_arcsec / 3600;
  return { ra, dec, offset: Math.hypot(dx, dy) * g.scale_arcsec };
}
function applyZoom() {
  const stage = $("frame-stage");
  const maxPanX = stage.clientWidth * (zoom - 1) / 2;
  const maxPanY = stage.clientHeight * (zoom - 1) / 2;
  panX = Math.max(-maxPanX, Math.min(maxPanX, panX));
  panY = Math.max(-maxPanY, Math.min(maxPanY, panY));
  const img = $("frame");
  const transform = `translate(${panX}px, ${panY}px) scale(${zoom})`;
  img.style.transform = transform;
  $("markers").style.transform = transform;
  $("gaiaoverlay").style.transform = transform;
  $("zoom-level").value = `${Math.round(zoom * 100)}%`;
  $("zoom-level").textContent = `${Math.round(zoom * 100)}%`;
  $("zoom-out").disabled = zoom <= 1;
  $("zoom-in").disabled = zoom >= 5;
  $("zoom-reset").disabled = zoom === 1 && panX === 0 && panY === 0;
  $("frame-stage").classList.toggle("zoomed", zoom > 1);
  $("frame").style.cursor = zoom > 1 ? (dragging ? "grabbing" : "grab") : "crosshair";
}
function setZoom(next, x = null, y = null) {
  const old = zoom; zoom = Math.max(1, Math.min(5, next));
  if (x !== null && y !== null && old !== zoom) {
    const rect = $("frame-stage").getBoundingClientRect();
    const cx = x - (rect.left + rect.width / 2), cy = y - (rect.top + rect.height / 2);
    // Keep the sky point under the pointer fixed, including the first zoom step from 100%.
    const ratio = zoom / old;
    panX = cx - ratio * (cx - panX); panY = cy - ratio * (cy - panY);
  }
  if (zoom === 1) panX = panY = 0;
  applyZoom();
}
function resetZoom() { zoom = 1; panX = panY = 0; if ($("frame-stage")) applyZoom(); }
async function loadSpectrum(ra = null, dec = null) {
  const q = targetQuery();
  if (ra !== null) { q.set("ra", ra.toFixed(5)); q.set("dec", dec.toFixed(5)); }
  $("speclabel").textContent = ra === null ? "(target) — loading… may download all 6 detectors" :
    `(RA ${ra.toFixed(4)}, Dec ${dec.toFixed(4)}) — loading…`;
  try {
    q.set("fetch", String(!state.offline));
    const s = await api("/api/spectrum?" + q.toString());
    drawSpectrum(s);
    $("speclabel").textContent = ra === null ? `(target · ${s.n_points} image points)` :
      `(RA ${ra.toFixed(4)}, Dec ${dec.toFixed(4)} · ${s.n_points} points)`;
  } catch (e) {
    $("speclabel").textContent = "— " + e.message;
  }
}
async function loadCachedSpectrum() {
  try {
    const s = await api(`/api/spectrum?target=${state.target}&cached_only=true`);
    if (!s.n_points || s.n_points < 20) return;  // too few points to be a meaningful spectrum
    drawSpectrum(s);
    $("speclabel").textContent = `(target · ${s.n_points} image points)`;
  } catch (e) { /* no saved spectrum yet: the button stays available */ }
}
const featurePlugin = {
  id: "features",
  afterDraw(chart, args, opts) {
    const feats = opts.features || [];
    const { ctx, chartArea, scales } = chart;
    ctx.save();
    ctx.font = "10px system-ui";
    ctx.fillStyle = "#8d97b0";
    ctx.strokeStyle = "rgba(141,151,176,.35)";
    ctx.setLineDash([3, 3]);
    for (const f of feats) {
      const x = scales.x.getPixelForValue(f.wavelength_um);
      if (x < chartArea.left || x > chartArea.right) continue;
      ctx.beginPath(); ctx.moveTo(x, chartArea.top); ctx.lineTo(x, chartArea.bottom); ctx.stroke();
      ctx.save(); ctx.translate(x + 3, chartArea.top + 4); ctx.rotate(Math.PI / 2); ctx.fillText(f.molecule, 0, 0); ctx.restore();
    }
    ctx.restore();
  },
};
function drawSpectrum(s) {
  $("specbtn").classList.add("hidden");
  if (specChart) specChart.destroy();
  const pts = s.points.map(p => ({ x: p.wavelength_um, y: p.flux_mJy }));
  const line = s.wavelength_um.map((w, i) => ({ x: w, y: s.flux[i] }));
  const top = Math.max(...s.flux.filter(Number.isFinite)) * 1.3;
  specChart = new Chart($("specchart"), {
    type: "scatter",
    data: { datasets: [
      { label: "single images", data: pts, pointRadius: 2, backgroundColor: "rgba(90,169,255,.35)" },
      { label: "spectrum", data: line, showLine: true, pointRadius: 0, borderColor: "#e6e9f2", borderWidth: 2 },
    ] },
    options: {
      animation: false, responsive: true, maintainAspectRatio: false,
      scales: {
        x: { min: 0.7, max: 5.1, title: { display: true, text: "wavelength (µm)", color: "#8d97b0" }, ticks: { color: "#8d97b0" }, grid: { color: "#263049" } },
        y: { suggestedMin: 0, max: top, title: { display: true, text: "flux (mJy)", color: "#8d97b0" }, ticks: { color: "#8d97b0" }, grid: { color: "#263049" } },
      },
      plugins: { legend: { labels: { color: "#e6e9f2" } }, features: { features: s.features } },
    },
    plugins: [featurePlugin],
  });
}

// ---------------------------------------------------------------- motion + light curve
function scienceBand() {
  // Built-in targets are always measured in their validated detector (motion, light curve, scan),
  // whatever detector is being viewed. Custom positions use the viewed detector.
  const t = targets.find(x => x.key === state.target);
  if (t) return t.best_band;
  return state.band === "RGB" ? "D1" : state.band;
}
function dataFolder() {
  if (manifest.data_folder) return manifest.data_folder;
  const sz = Number(manifest.size_deg);
  return manifest.target.label + (Math.abs(sz - 0.05) < 1e-9 ? "" : `_size${sz}`);
}
async function loadMotion(ra = null, dec = null) {
  const card = $("motioncard"), body = $("motionbody");
  card.classList.remove("hidden");
  const clicked = ra !== null;
  $("motiontitle").textContent = clicked ? `Motion of clicked star (RA ${ra.toFixed(4)}, Dec ${dec.toFixed(4)})`
                                         : "Motion measured from SPHEREx";
  const saved = !clicked ? savedMotionResult(state.target) : null;
  body.innerHTML = saved ? savedMotionHtml(saved, scienceBand(), true) :
    `<p class="muted"><span class="mini-spinner"></span>Measuring motion${clicked ? " and looking it up in Gaia DR3" : ""}…</p>`;
  const q = clicked
    ? `field=${encodeURIComponent(dataFolder())}&ra=${ra.toFixed(6)}&dec=${dec.toFixed(6)}&band=${scienceBand()}`
    : `target=${state.target}&band=${scienceBand()}`;
  try {
    const m = await api("/api/motion?" + q + `&fetch=${!state.offline}`);
    if (clicked && m.star) {
      $("motiontitle").textContent = `Motion of star at RA ${m.star.ra.toFixed(4)}, Dec ${m.star.dec.toFixed(4)}`;
    }
    const me = m.measured, r = m.reference;
    const nAll = m.n_used + m.n_rejected_outliers;
    const sat = m.n_core_missing > 0.2 * nAll
      ? `<p><span class="badge warn">saturated</span> The star is so bright that its centre is saturated in ${m.n_core_missing} of ${nAll} images; positions come from a fit to the unsaturated pixels around it.</p>`
      : "";
    let refHtml, badge;
    if (r) {
      const ok = m.verdict.startsWith("consistent");
      badge = `<span class="badge ${ok ? "good" : "bad"}">${m.verdict}</span>`;
      refHtml = `<p>Reference: <b>${r.total.toFixed(2)}″/yr</b> <span class="muted small">(${r.source})</span> &nbsp;${badge}</p>
        <table class="mtable">
          <tr><th></th><th>SPHEREx</th><th>Reference</th></tr>
          <tr><td>East (pmRA)</td><td>${me.pmra.toFixed(2)} ± ${me.pmra_err.toFixed(2)}</td><td>${r.pmra.toFixed(2)}</td></tr>
          <tr><td>North (pmDec)</td><td>${me.pmdec.toFixed(2)} ± ${me.pmdec_err.toFixed(2)}</td><td>${r.pmdec.toFixed(2)}</td></tr>
        </table>`;
    } else {
      const cand = m.verdict.startsWith("DISCOVERY");
      const offline = m.verdict.includes("offline");
      badge = `<span class="badge ${cand || offline ? "warn" : ""}">${m.verdict}</span>`;
      refHtml = `<p>${badge}</p><p class="muted small">${m.reference_note}</p>
        <table class="mtable">
          <tr><th></th><th>SPHEREx</th></tr>
          <tr><td>East (pmRA)</td><td>${me.pmra.toFixed(2)} ± ${me.pmra_err.toFixed(2)}</td></tr>
          <tr><td>North (pmDec)</td><td>${me.pmdec.toFixed(2)} ± ${me.pmdec_err.toFixed(2)}</td></tr>
        </table>
        ${cand ? '<p class="small">This object seems to move but Gaia has no motion for it. Check it in the Blink and Difference views and in WiseView before getting excited: artefacts and blended stars can mimic motion.</p>' : ""}`;
    }
    body.innerHTML = `
      <div class="big">${me.total.toFixed(2)} ± ${me.total_err.toFixed(2)}″/yr</div>
      ${refHtml}
      <p class="muted small">${m.n_used} images in ${m.n_epochs} epochs over ${m.baseline_years} years · detector ${m.band} · method ${m.method}</p>
      ${sat}
      ${clicked && state.target ? '<p><button id="backtotarget">Back to the target</button></p>' : ""}`;
    const back = document.getElementById("backtotarget");
    if (back) back.onclick = () => { loadMotion(); loadSpectrum(); };
    return m.star || null;
  } catch (e) {
    if (saved) {
      body.innerHTML = savedMotionHtml(saved, scienceBand(), false);
    } else {
      const why = /proxy|IRSA|saved data|not responding/i.test(e.message)
        ? (clicked ? "This selected source needs the original FITS data and a working IRSA connection; cached PNG frames cannot measure its motion." : "No saved motion result exists for this target, and IRSA could not provide FITS data.")
        : e.message;
      body.innerHTML = `<p class="muted">Motion not available: ${why}</p>`;
    }
    return null;
  }
}
function savedMotionResult(target) {
  const snapshots = {
    barnard: { D3: { value: "10.17", error: "0.44", reference: "10.39", source: "Gaia DR3", band: "D3" } },
    epsindib: {
      D3: { value: "4.53", error: "0.28", reference: "4.71", source: "published value", band: "D3" },
      D6: { value: "4.51", error: "0.28", reference: "4.71", source: "published value", band: "D6" },
    },
    wise0855: { D6: { value: "8.96", error: "1.21", reference: "8.13", source: "published value", band: "D6" } },
  };
  return snapshots[target]?.D3 || snapshots[target]?.D6 || null;
}
function savedMotionHtml(saved, viewingBand, checkingLive) {
  return `<div class="big">${saved.value} ± ${saved.error}″/yr</div>
    <p><span class="badge warn">Saved project result · not recalculated</span></p>
    <p>Reference: <b>${saved.reference}″/yr</b> (${saved.source}) · saved detector ${saved.band} · viewing ${viewingBand}</p>
    <p class="muted small">This is the team's previously measured SPHEREx result. ${checkingLive
      ? '<span class="mini-spinner"></span>Checking for a live update…'
      : 'Live motion needs the original FITS data and a working IRSA connection.'}</p>`;
}
// ---------------------------------------------------------------- mover scan
function moverClass(st) {
  if (st.startsWith("DISCOVERY")) return "cand";
  if (st.startsWith("known mover") || st.startsWith("known target")) return "known";
  return st === "static" ? "source-static" : "other";
}
function inspectSource(s, index = -1) {
  document.querySelectorAll("#markers .mk").forEach((el, i) => el.classList.toggle("selected", i === index));
  document.querySelectorAll("#moversbody tr.clickable").forEach((el, i) => el.classList.toggle("selected", i === index));
  loadMotion(s.ra, s.dec).then(star => loadSpectrum(star ? star.ra : s.ra, star ? star.dec : s.dec));
}
async function scanMovers() {
  const card = $("moverscard"), body = $("moversbody");
  card.classList.remove("hidden");
  body.innerHTML = '<p class="muted"><span class="mini-spinner"></span>Finding every source, measuring its motion in each epoch, and checking movers against Gaia DR3… (10–60 s)</p>';
  const q = targetQuery(); q.set("band", scienceBand()); q.set("size", state.size);
  try {
    q.set("fetch", String(!state.offline));
    const r = await api("/api/movers?" + q.toString());
    scanSources = r.sources || [];
    const [pw, ph] = r.png_size;
    $("markers").innerHTML = scanSources.map((s, i) =>
      `<button type="button" class="mk ${moverClass(s.status)}" data-source="${i}" style="left:${s.x_png / pw * 100}%;top:${s.y_png / ph * 100}%"
            title="${s.moving ? `${s.total.toFixed(1)}″/yr · ` : "No significant motion · "}${s.status}" aria-label="Inspect source ${i + 1}, RA ${s.ra.toFixed(5)}, Dec ${s.dec.toFixed(5)}"></button>`).join("");
    $("markers").classList.toggle("hidden", !$("showmarkers").checked);
    const rows = scanSources.map((s, i) => `
      <tr class="clickable" data-i="${i}">
        <td>${s.total.toFixed(2)} ± ${s.total_err.toFixed(2)}″/yr</td>
        <td><span class="badge ${moverClass(s.status) === "cand" ? "warn" : moverClass(s.status) === "known" ? "good" : ""}">${s.moving ? s.status : "No significant motion detected"}</span></td>
        <td>${s.snr.toFixed(1)}</td>
        <td class="muted small">${s.gaia || "—"}</td>
      </tr>`).join("");
    body.innerHTML = `
      <p><b>${r.n_sources}</b> detected sources · <b>${r.n_movers}</b> moving · <b style="color:var(--warn)">${r.n_candidates}</b> discovery candidates
         ${r.gaia_checked ? "" : '<span class="badge warn">Gaia offline: movers not verified</span>'}</p>
      ${scanSources.length ? `<div class="source-results"><table class="mtable"><tr><th>Motion</th><th>Detection</th><th>S/N</th><th>Gaia DR3</th></tr>${rows}</table></div>
        <p class="muted small">Select a source row or circular image marker to inspect it. Grey markers are detected sources without significant measured motion; colored markers are movers. ${r.note}</p>`
        : '<p class="muted">No sources passed the current image detection threshold in this field.</p>'}`;
    body.querySelectorAll("tr.clickable").forEach(tr => tr.onclick = () => {
      inspectSource(scanSources[parseInt(tr.dataset.i, 10)], parseInt(tr.dataset.i, 10));
    });
    $("markers").querySelectorAll(".mk").forEach(marker => marker.onclick = ev => {
      ev.stopPropagation(); const i = Number(marker.dataset.source); inspectSource(scanSources[i], i);
    });
  } catch (e) {
    body.innerHTML = `<p class="muted">Scan failed: ${e.message}</p>`;
  }
}

async function loadLightcurve() {
  $("lccard").classList.remove("hidden");
  try {
    const lc = await api(`/api/lightcurve?target=${state.target}&band=${scienceBand()}&fetch=${!state.offline}`);
    const toYear = (d) => { const t = new Date(d + "T00:00:00Z"); const y = t.getUTCFullYear(); return y + (t - Date.UTC(y, 0, 1)) / 31557600000; };
    if (lcChart) lcChart.destroy();
    lcChart = new Chart($("lcchart"), {
      type: "scatter",
      data: { datasets: [{ label: `${lc.band} brightness`, data: lc.points.map(p => ({ x: toYear(p.date), y: p.flux })), backgroundColor: "#5aa9ff" }] },
      options: { animation: false, responsive: true, maintainAspectRatio: false, scales: {
        x: { title: { display: true, text: "year", color: "#8d97b0" }, ticks: { color: "#8d97b0" }, grid: { color: "#263049" } },
        y: { title: { display: true, text: "flux (relative)", color: "#8d97b0" }, ticks: { color: "#8d97b0" }, grid: { color: "#263049" } } },
        plugins: { legend: { labels: { color: "#e6e9f2" } } } },
    });
  } catch (e) {
    $("lccard").innerHTML = `<h2>Brightness over time</h2><p class="muted">Not available: ${e.message}</p>`;
  }
}

// ---------------------------------------------------------------- sky map (Aladin Lite + Gaia / WISE)
// Load Aladin Lite from the first address that works (main CDS server, older CDS domain, npm CDN)
const ALADIN_SOURCES = [
  "https://cdn.jsdelivr.net/npm/aladin-lite@3/dist/aladin.umd.cjs",
  "https://aladin.u-strasbg.fr/AladinLite/api/v3/latest/aladin.js",
  "https://aladin.cds.unistra.fr/AladinLite/api/v3/latest/aladin.js",
];
let aladinLoading = null;
let lastMapPosition = null;
const HIPS_MIRRORS = {
  twomass: [
    "https://casda.csiro.au/hips/2MASS/Color",
    "https://hips.china-vo.org/m/CDS_P_2MASS_color",
    "https://skies.esac.esa.int/2MASSColor",
  ],
  wise: [
    "https://hips.china-vo.org/m/CDS_P_allWISE_color",
    "https://skies.esac.esa.int/AllWISEColor",
  ],
};
async function availableHips(mirrors) {
  for (const base of mirrors) {
    try {
      const response = await fetch(`${base}/properties`, { method: "HEAD", mode: "cors", cache: "no-store", signal: AbortSignal.timeout(5000) });
      if (response.ok) return base;
    } catch (e) { console.warn("HiPS mirror unavailable:", base); }
  }
  return null;
}
async function loadAladin() {
  if (typeof A !== "undefined") return Promise.resolve(true);
  if (aladinLoading) return aladinLoading;
  const loading = (async () => {
    for (const src of ALADIN_SOURCES) {
      const ok = await new Promise((resolve) => {
        const el = document.createElement("script");
        el.src = src; el.charset = "utf-8";
        const timer = setTimeout(() => { el.remove(); resolve(false); }, 7000);
        el.onload = () => { clearTimeout(timer); resolve(typeof A !== "undefined"); };
        el.onerror = () => { clearTimeout(timer); resolve(false); };
        document.head.appendChild(el);
      });
      if (ok) { console.log("Aladin loaded from", src); return true; }
      console.warn("Aladin not available from", src);
    }
    return false;
  })();
  aladinLoading = loading;
  const ok = await loading;
  if (!ok && aladinLoading === loading) aladinLoading = null;
  return ok;
}
function mapMessage(text) {
  $("aladin").closest(".map-card").classList.add("map-error");
  $("aladin").innerHTML = `<div class="map-error-message"><p class="muted">${text}</p>
    <button id="map-retry" type="button">Retry</button></div>`;
  $("map-retry").onclick = () => {
    if (!lastMapPosition) return;
    aladinLoading = null; aladin = null;
    updateMap(...lastMapPosition);
  };
}
async function updateMap(ra, dec) {
  lastMapPosition = [ra, dec];
  $("aladin").closest(".map-card").classList.remove("map-error");
  if (typeof A === "undefined") mapMessage("Loading sky map…");
  await loadAladin();
  if (typeof A === "undefined") {
    mapMessage("Interactive sky map (Aladin, CDS Strasbourg) is currently unavailable. Gaia stars are shown directly on the SPHEREx image above; for WISE, use ‘Compare in WiseView’.");
    return;
  }
  try {
    await A.init;
    if (!aladin) {
      $("aladin").innerHTML = "";
      aladin = A.aladin("#aladin", { fov: 0.08, target: `${ra} ${dec}`, showReticle: true });
    } else {
      aladin.gotoRaDec(ra, dec);
    }
    try { aladin.removeLayers(); } catch (e) { /* older API */ }
    const surveyUrl = await availableHips($("wise").checked ? HIPS_MIRRORS.wise : HIPS_MIRRORS.twomass);
    if (!surveyUrl) throw new Error("No 2MASS/WISE mirror is reachable from this browser.");
    aladin.setBaseImageLayer(aladin.newImageSurvey(surveyUrl));
    if ($("gaia").checked) {
      try {
        const cat = A.catalogFromVizieR("I/355/gaiadr3", `${ra} ${dec}`, 0.04, { onClick: "showTable", color: "#ffcc4d", sourceSize: 10 });
        if (cat && typeof cat.then === "function") cat.then(c => aladin.addCatalog(c)).catch(e => console.warn("Gaia overlay unavailable:", e));
        else if (cat) aladin.addCatalog(cat);
      } catch (e) { console.warn("Gaia overlay unavailable:", e); }
    }
    $("aladin").closest(".map-card").classList.remove("map-error");
  } catch (e) {
    console.error("Aladin failed:", e);
    mapMessage("Interactive sky map (Aladin, CDS Strasbourg) is currently unavailable. Gaia stars are shown directly on the SPHEREx image above; for WISE, use ‘Compare in WiseView’.");
  }
}

// ---------------------------------------------------------------- wire up
function applyFilters() {
  const b = $("bright").value, c = $("contrast").value, inv = $("invert").checked ? 1 : 0;
  $("frame").style.filter = `brightness(${b}%) contrast(${c}%) invert(${inv})`;
}
window.addEventListener("DOMContentLoaded", async () => {
  readHash();
  try { await loadPresets(); } catch (e) { status("Server not reachable: " + e.message, true); return; }
  syncControls();

  $("go").onclick = () => {
    const ra = parseFloat($("ra").value), dec = parseFloat($("dec").value);
    if (Number.isFinite(ra) && Number.isFinite(dec)) {
      const t = manifest && manifest.target;
      const moved = !t || Math.abs(ra - t.ra) > 1e-4 || Math.abs(dec - t.dec) > 1e-4;
      if (moved) { state.target = null; state.ra = ra; state.dec = dec; }
    }
    state.band = $("band").value; state.size = parseFloat($("size").value);
    syncControls(); load();
  };
  $("band").onchange = () => { state.band = $("band").value; load(); };
  $("size").onchange = () => { state.size = parseFloat($("size").value); load(); };
  document.querySelectorAll(".mode").forEach(b => b.onclick = () => setMode(b.dataset.mode));
  $("play").onclick = () => { playing = !playing; $("play").textContent = playing ? "⏸" : "▶"; playing ? start() : stop(); };
  $("scrub").oninput = () => { idx = parseInt($("scrub").value, 10); if (state.mode !== "movie") setMode("movie"); stop(); playing = false; $("play").textContent = "▶"; showFrame(idx); };
  $("speed").oninput = () => {
    state.speed = SPEED_MIN_MS + SPEED_MAX_MS - parseInt($("speed").value, 10);
    writeHash(); if (timer) start();
  };
  ["bright", "contrast", "invert"].forEach(id => $(id).oninput = applyFilters);
  $("share").onclick = async () => { writeHash(); await navigator.clipboard.writeText(location.href); $("share").textContent = "Copied!"; setTimeout(() => $("share").textContent = "Copy link", 1500); };
  $("specbtn").onclick = () => loadSpectrum();
  $("scanbtn").onclick = () => scanMovers();
  $("showmarkers").onchange = () => $("markers").classList.toggle("hidden", !$("showmarkers").checked);
  $("showgaia").onchange = () => loadGaiaOverlay();
  $("frame").onclick = (ev) => {
    if (!manifest) return;
    if (zoom > 1 || dragging) { status("Reset zoom to click a star and measure it."); return; }
    const s = pixelToSky(ev);
    // clicks on the target itself use the tracked (moving) position
    if (state.target && s.offset < 9) {
      loadSpectrum();
      if (manifest.target.kind === "motion") loadMotion();
    } else {
      $("speclabel").textContent = "— finding the star you clicked…";
      loadMotion(s.ra, s.dec).then(star => star ? loadSpectrum(star.ra, star.dec) : loadSpectrum(s.ra, s.dec));
    }
  };
  $("zoom-in").onclick = () => setZoom(zoom + .5);
  $("zoom-out").onclick = () => setZoom(zoom - .5);
  $("zoom-reset").onclick = resetZoom;
  $("frame-stage").addEventListener("wheel", ev => { ev.preventDefault(); setZoom(zoom + (ev.deltaY < 0 ? .25 : -.25), ev.clientX, ev.clientY); }, { passive:false });
  $("frame-stage").addEventListener("pointerdown", ev => {
    if (zoom <= 1 || ev.target.closest("button")) return;
    dragging = true; dragStart = { x:ev.clientX, y:ev.clientY, panX, panY }; $("frame-stage").setPointerCapture(ev.pointerId); applyZoom();
  });
  $("frame-stage").addEventListener("pointermove", ev => {
    if (!dragging || !dragStart) return;
    if (Math.hypot(ev.clientX-dragStart.x, ev.clientY-dragStart.y) > 3) ev.preventDefault();
    panX = dragStart.panX + ev.clientX-dragStart.x; panY = dragStart.panY + ev.clientY-dragStart.y; applyZoom();
  });
  const endDrag = () => { if (dragging) setTimeout(() => { dragging = false; applyZoom(); }, 0); dragStart = null; };
  $("frame-stage").addEventListener("pointerup", endDrag);
  $("frame-stage").addEventListener("pointercancel", endDrag);
  $("offline").onchange = () => { state.offline = $("offline").checked; writeHash(); load(); };
  $("close-guide").onclick = () => {
    const collapsed = $("how-to").classList.toggle("collapsed");
    $("close-guide").textContent = collapsed ? "+" : "−";
    $("close-guide").setAttribute("aria-expanded", String(!collapsed));
    $("close-guide").setAttribute("aria-label", collapsed ? "Expand quick start guide" : "Collapse quick start guide");
    $("close-guide").title = collapsed ? "Expand guide" : "Collapse guide";
  };
  $("gaia").onchange = () => manifest && updateMap(manifest.target.ra, manifest.target.dec);
  $("wise").onchange = () => manifest && updateMap(manifest.target.ra, manifest.target.dec);

  load();
});
