import { parseProgram, nParams, haloFor, GPUEngine, runCPU } from './engine.js';
import { Zarr, Slab, clamp, S3 } from './zarr.js';
import { banner } from './banner.js';
import { ngState } from './ng-state.js';
import { VolumeView } from './volume.js';
import { FAMILIES, familyForZoom, scaledModel, scaleLabel } from './route.js';
import { MULTI, MULTI_COLORS, recommendExposure, segVolume, labelVolume, paletteFor, segRegion, renderSegCanvas, regionStats, toCanvas, COLORS, IDENT, isIdent, adjustU, adjustLogVol, percentileWindow } from './seg.js';

let TS = 128;                   // segmentation tile size (in-plane voxels); per-program via models/index.json
let MAX_TILES = 48;           // tiles segmented per refresh (nearest the view center first)
const $ = (id) => document.getElementById(id);

// ------------------------------------------------------------------------------------------ state
const assets = await (await fetch('assets.json')).json();
const modelIndex = await (await fetch('models/index.json')).json();
const MODEL_LABEL = { fable_blobs_v2: 'Hand-written cells v2: nucleus / soma (levels 0-1, cheap)', fable_vessels_v2: 'Hand-written vessels v2: capillary / vessel (levels 0-1, LEC only)', fable_blobs: 'EXPERIMENTAL hand-written cells: nucleus / soma / cluster (heavy)', fable_vessels: 'EXPERIMENTAL hand-written vessels: capillary / vessel / large (heavy, weak on LEC)', tissue: 'Tissue (bg / gray / white)', layers: 'Cortical layers I-VI (experimental, unreliable on real slabs)', cells: 'Cells (nuclei / somata)', vessels_13dir: 'Vessels, 13 directions', vessels_26dir: 'Vessels, 26 directions' };
const programs = {};
async function getProgram(name) { return programs[name] ||= parseProgram(await (await fetch(`models/${modelIndex[name].file}`)).text()); }

let engine = null, backend = 'none';
try { engine = await GPUEngine.create(); backend = 'WebGPU'; } catch (e) { backend = 'CPU reference (no WebGPU: slow, small tiles only)'; }
banner.log('compute backend: ' + backend);

const view = { cx: 0, cy: 0, p: 100, z: 1024 };       // center in um (x,y); um per screen pixel; level-0 z index
let zarr = null, stain = '', winByLevel = {}, gen = 0;
banner.init(); banner.setStream(() => (zarr ? zarr.stat : null));
const rawTiles = new Map(), segTiles = new Map(), segStats = new Map();
const cv = $('cv'), ctx = cv.getContext('2d');

function fitCanvas() { const r = cv.getBoundingClientRect(), d = Math.min(devicePixelRatio || 1, 2); cv.width = Math.round(r.width * d); cv.height = Math.round(r.height * d); view.dpr = d; }
addEventListener('resize', () => { fitCanvas(); schedule(); });

// ------------------------------------------------------------------------------------------ UI wiring
for (const s of [...new Set(assets.map((a) => a.stain))].sort((a, b) => (['YO', 'NN', 'LEC'].indexOf(b) + 1) - (['YO', 'NN', 'LEC'].indexOf(a) + 1) || a.localeCompare(b)))
  $('stain').add(new Option(s, s));
$('model').add(new Option('Auto (follows zoom)', 'auto'));
for (const k of Object.keys(modelIndex)) if (modelIndex[k].scale === 1 && modelIndex[k].kind !== 'exposure') $('model').add(new Option(`${MODEL_LABEL[k] || k} - ${modelIndex[k].nonzero} params`, k));
$('model').add(new Option('None (raw only)', 'none'));
const sel = () => assets.filter((a) => a.stain === $('stain').value && a.sample === $('sample').value && (!$('session').value || a.session === $('session').value));
function fillSamples() { const ss = [...new Set(assets.filter((a) => a.stain === $('stain').value).map((a) => a.sample))].sort((a, b) => +a - +b);
  $('sample').innerHTML = ''; ss.forEach((v) => $('sample').add(new Option('sample ' + v, v))); fillSessions(); }
function fillSessions() { const ss = [...new Set(assets.filter((a) => a.stain === $('stain').value && a.sample === $('sample').value).map((a) => a.session))].sort();
  $('session').innerHTML = ''; ss.forEach((v) => $('session').add(new Option(`${v.slice(0, 4)}-${v.slice(4, 6)}-${v.slice(6, 8)} (${assets.filter((a) => a.stain === $('stain').value && a.sample === $('sample').value && a.session === v).length} strips)`, v))); fillChunks(); }
function fillChunks() { const cs = sel().sort((a, b) => a.chunk - b.chunk);
  $('chunk').innerHTML = ''; $('chunk').add(new Option(`whole slab, all ${cs.length} strips stitched`, 'slab')); cs.forEach((a) => $('chunk').add(new Option('strip ' + a.chunk + ' only', a.zarr))); }
$('stain').onchange = () => { fillSamples(); openAsset(); };
// Paste a dashboard / Neuroglancer link: every zarr:// source in its layers becomes the stitched slab to open.
$('nglink').onchange = () => {
  let txt = $('nglink').value.trim(); try { txt = decodeURIComponent(txt); } catch (e) { /* keep raw */ }
  const ids = [...new Set([...txt.matchAll(/dandiarchive\.s3\.amazonaws\.com\/zarr\/([0-9a-f-]{36})/g)].map((m) => m[1]))], hit = ids.map((id) => assets.find((a) => a.zarr === id)).filter(Boolean);
  if (!hit.length) { banner.log('no known DANDI 000108 zarr sources found in that link', 'error'); setStatus('No known zarr sources found in that link.', true); return; }
  banner.log(`link: ${ids.length} zarr sources, ${hit.length} known; opening ${hit[0].stain} sample ${hit[0].sample} stitched`);
  selectAsset(hit[0], true); openAsset();
};
$('sample').onchange = () => { fillSessions(); openAsset(); };
$('session').onchange = () => { fillChunks(); openAsset(); };
$('chunk').onchange = () => openAsset();
$('model').onchange = $('fscale').onchange = () => { invalidateSeg(); schedule(); };
$('debounce').oninput = () => { dbn.delay = +$('debounce').value; $('debval').textContent = dbn.delay; };
$('segauto').onchange = () => { if ($('segauto').checked) schedule(0); else setStatus('manual mode: press "Run segmentation now" to segment this view'); };
$('segrun').onclick = () => { clearTimeout(dbn.timer); gen++; dbn.chain = dbn.chain.then(() => refresh({ force: true })).catch((e) => console.error(e)); };
$('opacity').oninput = () => draw();
$('z').oninput = () => { view.z = +$('z').value; $('zval').textContent = view.z; draw(); schedule(); };
$('reset').onclick = () => jumpToTissue();
$('fit').onclick = () => zarr && fitView();
$('seams').onchange = async () => {   // opt-in local seam correction for stitched slabs (see README for the measured effect)
  if (!zarr || !zarr.setSeams) { banner.log('seam correction applies to stitched slabs only'); $('seams').checked = false; return; }
  const bt = banner.start('other', 'Estimating seam difference fields (coarsest level)');
  try { await zarr.setSeams($('seams').checked); banner.finish(bt, true, `${zarr.seam ? zarr.seam.size : 0} seams`); } catch (e) { banner.finish(bt, false, e.message); }
  rawTiles.clear(); segTiles.clear(); segStats.clear(); fovSeg = null; draw(); schedule(0);
};
$('ngopen').onclick = () => {   // the same slab, position, zoom and model-recommended exposure in our hosted Neuroglancer
  if (!zarr) return; const L0 = zarr.levels[0], ids = zarr.ids || [new URL(zarr.base).pathname.split('/').filter(Boolean).pop()];
  const st = ngState(ids, { sp: L0.sp, off: zarr.ids ? L0.origin : L0.tr }, view, rec);
  window.open('ng/index.html#!' + encodeURIComponent(JSON.stringify(st)), '_blank');
};
function readAdj() { adj.b = +$('bri').value; adj.c = +$('con').value; adj.g = +$('gam').value; $('brival').textContent = adj.b.toFixed(2); $('conval').textContent = adj.c.toFixed(2); $('gamval').textContent = adj.g.toFixed(2); }
for (const id of ['bri', 'con', 'gam']) $(id).oninput = () => { readAdj(); repaintRaw(); invalidateSeg(); schedule(dbn.delay + 250); };   // the model sees the display image, so it re-runs, after a longer settle
$('autoexp').onchange = () => { repaintRaw(); invalidateSeg(); schedule(); };
$('adjreset').onclick = () => { $('bri').value = 0; $('con').value = 1; $('gam').value = 1; readAdj(); repaintRaw(); invalidateSeg(); schedule(); };

let dragging = null;
cv.addEventListener('pointerdown', (e) => { cv.setPointerCapture(e.pointerId); dragging = { x: e.clientX, y: e.clientY, cx: view.cx, cy: view.cy }; dbn.interacting++; });
cv.addEventListener('pointermove', (e) => { if (!dragging) return; const k = view.p * (cv.width / cv.getBoundingClientRect().width);
  view.cx = dragging.cx - (e.clientX - dragging.x) * k; view.cy = dragging.cy - (e.clientY - dragging.y) * k; draw(); schedule(); });
const endDrag = () => { if (!dragging) return; dragging = null; dbn.interacting = Math.max(0, dbn.interacting - 1); schedule(); };
cv.addEventListener('pointerup', endDrag); cv.addEventListener('pointercancel', endDrag);
function stepZ(dir, big) {
  const l = rawLevel(), nz = zarr.levels[0].shape[0], st = 2 ** l * (big ? 10 : 1);
  view.z = clamp(Math.round(view.z + dir * st), 0, nz - 1); $('z').value = view.z; $('zval').textContent = view.z; draw(); schedule();
}
addEventListener('keydown', (e) => {
  if (!zarr || /^(INPUT|SELECT|TEXTAREA)$/.test(document.activeElement.tagName) && document.activeElement.type !== 'range') return;
  const d = { PageUp: 1, ArrowUp: 1, ']': 1, PageDown: -1, ArrowDown: -1, '[': -1 }[e.key];
  if (d) { e.preventDefault(); stepZ(d, e.shiftKey); }
});
$('depth').onchange = $('dmode').onchange = () => draw();
cv.addEventListener('wheel', (e) => { e.preventDefault(); if (e.shiftKey) { stepZ(e.deltaY > 0 ? -1 : 1, false); return; } const r = cv.getBoundingClientRect(), k = cv.width / r.width;
  const mx = (e.clientX - r.left) * k, my = (e.clientY - r.top) * k, f = Math.exp(e.deltaY * 0.0015);
  const np = clamp(view.p * f, 0.4 * zarr.levels[0].sp[2] / view.dpr, 400);
  view.cx += (mx - cv.width / 2) * (view.p - np); view.cy += (my - cv.height / 2) * (view.p - np); view.p = np; draw(); schedule(); }, { passive: false });

// ------------------------------------------------------------------------------------------ routing by zoom
// p = um per screen pixel. Fine programs run at level 0 (their sigmas are physical and were checked there);
// tissue was checked at level 4 and is run at level 4/5.
function route() {
  const sel = $('model').value;
  let family = sel, note = '';
  if (sel === 'auto') family = familyForZoom(view.p, stain);
  if (family === 'none') return { model: 'none', level: 0 };
  const m = scaledModel(family, view.p, zarr, family === 'layers' ? 'auto' : $('fscale').value), kind = FAMILIES[family].kind;
  TS = modelIndex[m.model].tile || 128;
  if (['cells', 'vessels', 'blobs', 'vessels_ms'].includes(kind) && view.p > 16 && !FAMILIES[family].invariant) note = 'zoom in for ' + kind;
  return { ...m, kind, note, label: scaleLabel(kind, m.scale, m.sigmaUm) };
}

// ------------------------------------------------------------------------------------------ rendering
function rawLevel() { let l = 0; for (let i = 0; i < zarr.levels.length; i++) if (zarr.levels[i].sp[2] <= view.p * 1.0) l = i; return l; }

const adj = { ...IDENT };
let rec = null;   // model-recommended exposure for the current view: {lo, hi, gamma, mean, level}
const auto = () => $('autoexp').checked && rec;
const effAdj = () => ({ b: adj.b, c: adj.c, g: adj.g * (auto() ? rec.gamma : 1) });          // sliders act on top of the recommended exposure
const currentWin = (l) => auto() ? [rec.lo, rec.hi] : winByLevel[l];
const viewWin = (v) => auto() ? [rec.lo, rec.hi] : percentileWindow(v);
function paintRaw(lg, w, h, win) {
  return toCanvas(null, w, h, (px) => { const r = win[1] - win[0];
    for (let i = 0; i < w * h; i++) { const v = adjustU(clamp((lg[i] - win[0]) / r, 0, 1), effAdj()) * 255; px[4 * i] = px[4 * i + 1] = px[4 * i + 2] = v; px[4 * i + 3] = 255; } });
}
const RT = 256;   // raw display tile size (voxels)
function rawTile(l, zl, ty, tx) {
  const n = +$('depth').value, mode = $('dmode').value;
  const key = `${zarr.base}|${l}|${zl}|${ty}|${tx}|${n}|${mode}`, hit = rawTiles.get(key);
  if (hit) return hit.c;
  if (hit === null) return null;
  rawTiles.set(key, null);
  const L = zarr.levels[l], half = (n - 1) >> 1, z0 = Math.max(0, zl - half), z1 = Math.min(L.shape[0], zl + half + 1);
  const y0 = ty * RT, y1 = Math.min(L.shape[1], y0 + RT), x0 = tx * RT, x1 = Math.min(L.shape[2], x0 + RT), w = x1 - x0, h = y1 - y0;
  if (w <= 0 || h <= 0 || z1 <= z0) return null;
  const base = zarr.base;
  zarr.crop(l, z0, z1, y0, y1, x0, x1).then((v) => {
    if (zarr.base !== base) return;
    const lg = new Float32Array(w * h), pl = w * h, np = z1 - z0;
    for (let i = 0; i < pl; i++) { let acc = mode === 'max' ? -Infinity : 0; for (let k = 0; k < np; k++) { const x = v[k * pl + i]; acc = mode === 'max' ? (x > acc ? x : acc) : acc + x; } lg[i] = mode === 'max' ? acc : acc / np; }
    if (!winByLevel[l]) winByLevel[l] = percentileWindow(lg);
    rawTiles.set(key, { c: paintRaw(lg, w, h, currentWin(l)), lg, w, h, l });
    if (rawTiles.size > 900) rawTiles.delete(rawTiles.keys().next().value); draw();
  });
  return null;
}
function repaintRaw() { for (const t of rawTiles.values()) if (t) t.c = paintRaw(t.lg, t.w, t.h, currentWin(t.l)); draw(); }

function drawGrid(l, getTile, alpha, gcy, gcx) {
  // gcy/gcx: grid cell size in voxels (zarr chunk size for raw tiles, TS for segmentation tiles)
  const L = zarr.levels[l], sp = L.sp, W = cv.width, H = cv.height, p = view.p;
  const x0 = view.cx - W / 2 * p, y0 = view.cy - H / 2 * p, x1 = view.cx + W / 2 * p, y1 = view.cy + H / 2 * p;
  const nx = Math.ceil(L.shape[2] / gcx), ny = Math.ceil(L.shape[1] / gcy);
  const gx0 = clamp(Math.floor(x0 / sp[2] / gcx), 0, nx - 1), gx1 = clamp(Math.floor(x1 / sp[2] / gcx), 0, nx - 1);
  const gy0 = clamp(Math.floor(y0 / sp[1] / gcy), 0, ny - 1), gy1 = clamp(Math.floor(y1 / sp[1] / gcy), 0, ny - 1);
  ctx.globalAlpha = alpha;
  for (let gy = gy0; gy <= gy1; gy++) for (let gx = gx0; gx <= gx1; gx++) {
    const t = getTile(gy, gx); if (!t) continue;
    ctx.drawImage(t, (gx * gcx * sp[2] - x0) / p, (gy * gcy * sp[1] - y0) / p, t.width * sp[2] / p + 0.6, t.height * sp[1] / p + 0.6);
  }
  ctx.globalAlpha = 1;
}

let curRoute = null, fovSeg = null, segWinUsed = null;
function draw() {
  if (!zarr) return;
  const W = cv.width, H = cv.height;
  ctx.fillStyle = '#000'; ctx.fillRect(0, 0, W, H); ctx.imageSmoothingEnabled = false;
  const l = rawLevel(), zl = clamp(Math.floor(view.z / 2 ** l), 0, zarr.levels[l].shape[0] - 1);
  drawGrid(l, (gy, gx) => rawTile(l, zl, gy, gx), 1, RT, RT);
  const r = curRoute;
  if (r && fovSeg && fovSeg.model === r.model) {   // one forward pass over the visible region
    { const f = fovSeg, sp = zarr.levels[f.level].sp, x0 = view.cx - W / 2 * view.p, y0 = view.cy - H / 2 * view.p;
      ctx.globalAlpha = +$('opacity').value / 100; ctx.drawImage(f.c, (f.x0 * sp[2] - x0) / view.p, (f.y0 * sp[1] - y0) / view.p, f.w * sp[2] / view.p + 0.6, f.h * sp[1] / view.p + 0.6); ctx.globalAlpha = 1; }
  } else if (r && r.model !== 'none' && !['tissue', 'layers'].includes(r.kind)) {
    const zs = clamp(Math.floor(view.z / 2 ** r.level), 0, zarr.levels[r.level].shape[0] - 1);
    drawGrid(r.level, (gy, gx) => segTiles.get(segKey(r.model, r.level, zs, gy, gx)) || null, +$('opacity').value / 100, TS, TS);
  }
  $('hud').textContent = `${view.p.toFixed(2)} um/px · level ${l} (raw) · z ${view.z} (${(view.z * zarr.levels[0].sp[0] / 1000).toFixed(2)} mm)` + (+$('depth').value > 1 ? ` · ${$('depth').value}-plane ${$('dmode').value}` : '');
  const barPx = 100 * view.dpr, um = barPx * view.p, nice = [1, 2, 5].map((m) => m * 10 ** Math.floor(Math.log10(um))).reduce((a, b) => Math.abs(b - um) < Math.abs(a - um) ? b : a);
  const sb = $('scale'); sb.firstElementChild.style.width = (nice / view.p / view.dpr) + 'px'; sb.lastElementChild.textContent = nice >= 1000 ? (nice / 1000) + ' mm' : nice + ' um';
}

// ------------------------------------------------------------------------------------------ segmentation tiles
const segKey = (m, l, z, gy, gx) => `${zarr.base}|${m}|${l}|${z}|${gy}|${gx}`;
// ---- debounced, interaction-aware, serialized segmentation scheduling
const dbn = { delay: 350, interacting: 0, timer: null, chain: Promise.resolve() };
const counts = { scheduled: 0, started: 0, tiles: 0, fov: 0 };
let segStale = false;                                    // cached segmentation invalid (exposure / model changed): cleared when the next run starts
const invalidateSeg = () => { segStale = true; };
function schedule(delay) {
  counts.scheduled++;
  gen++;                                                 // cancels any in-flight run at its next checkpoint; stale results are never stored
  clearTimeout(dbn.timer);
  const d = delay ?? dbn.delay;
  if ($('segauto').checked && curRoute && curRoute.model !== 'none') setStatus(`view changed: segmentation runs ${d} ms after you stop`);
  banner.setPending($('segauto').checked && curRoute && curRoute.model !== 'none' ? d : 0);
  dbn.timer = setTimeout(fire, d);
}
function fire() {
  if (dbn.interacting > 0) { dbn.timer = setTimeout(fire, 120); return; }   // never while dragging
  banner.setPending(0);
  dbn.chain = dbn.chain.then(() => refresh()).catch((e) => console.error(e)); // serialized: never two GPU jobs at once
}

async function segTile(prog, name, l, zl, gy, gx, fovWin) {
  const L = zarr.levels[l], y0 = gy * TS, x0 = gx * TS, y1 = Math.min(y0 + TS, L.shape[1]), x1 = Math.min(x0 + TS, L.shape[2]);
  counts.tiles++;
  const ea = effAdj();
  const r = await segRegion(zarr, engine, prog, l, zl, y0, y1, x0, x1, (v) => adjustLogVol(v, fovWin[0], fovWin[1], ea));
  segStats.set(segKey(name, l, zl, gy, gx), regionStats(modelIndex[name].kind, r, L.sp));
  return renderSegCanvas(modelIndex[name].kind, r);
}

async function refreshFov(prog, r, myGen, local = false) {
  const W = cv.width, H = cv.height, top = zarr.top, budget = engine ? Math.floor(engine.maxTokens(prog.d) * 0.6) : 150000;
  const ux0 = view.cx - W / 2 * view.p, ux1 = view.cx + W / 2 * view.p, uy0 = view.cy - H / 2 * view.p, uy1 = view.cy + H / 2 * view.p;
  let level = r.level, box, note = r.note || '';
  for (;;) {   // coarsest-needed level so that the whole FOV fits one forward pass
    const L = zarr.levels[level], sp = L.sp, [hz] = haloFor(prog, sp), zslab = Math.min(L.shape[0], 2 * hz + 1);
    box = [clamp(Math.floor(uy0 / sp[1]), 0, L.shape[1]), clamp(Math.ceil(uy1 / sp[1]), 0, L.shape[1]), clamp(Math.floor(ux0 / sp[2]), 0, L.shape[2]), clamp(Math.ceil(ux1 / sp[2]), 0, L.shape[2])];
    if (zslab * (box[1] - box[0]) * (box[3] - box[2]) <= budget || level >= top) break;
    if (local) return false;                                   // local programs keep their own scale: fall back to tiles instead of coarsening
    level++;
  }
  if (local && (box[1] <= box[0] || box[3] <= box[2])) return false;
  if (box[1] <= box[0] || box[3] <= box[2]) { fovSeg = null; draw(); setStatus('FOV outside the volume', true); return true; }
  r.level = level; updateInfo(r);
  const L = zarr.levels[level], zl = clamp(Math.floor(view.z / 2 ** level), 0, L.shape[0] - 1), t0 = performance.now();
  counts.fov++;
  const ea = effAdj();
  const bt = banner.start('infer', `${r.model} · FOV ${box[3] - box[2]}x${box[1] - box[0]} voxels · level ${level}`);
  let seg;
  try {
    seg = await segRegion(zarr, engine, prog, level, zl, box[0], box[1], box[2], box[3], (v) => { const w = viewWin(v); return adjustLogVol(v, w[0], w[1], ea); }, { halo: false, onStage: (st) => banner.update(bt, { detail: st === 'fetch' ? 'fetching the volume' : 'running the model' }) });
  } catch (e) { banner.finish(bt, false, e.message); throw e; }
  if (myGen !== gen) { banner.finish(bt, true, 'superseded by a newer view'); return true; }
  banner.finish(bt, true);
  const stats = regionStats(r.kind, seg, L.sp);
  fovSeg = { model: r.model, level, c: renderSegCanvas(r.kind, seg), y0: box[0], x0: box[2], w: box[3] - box[2], h: box[1] - box[0], stats };
  segStats.clear(); segStats.set(`${zarr.base}|${r.model}|${level}|${zl}|fov`, stats);
  draw(); setStatus(`${r.model}: FOV forward pass at level ${level} (${fovSeg.w}x${fovSeg.h} voxels) in ${(performance.now() - t0).toFixed(0)} ms` + (note ? ' · ' + note : ''), !!note);
  return true;
}

function updateViewWindow() {   // display window = percentiles over the raw tiles currently loaded for this view/level
  const l = rawLevel(), vals = [];
  for (const t of rawTiles.values()) if (t && t.l === l) for (let i = 0; i < t.lg.length; i += 7) vals.push(t.lg[i]);
  if (vals.length > 200) { winByLevel[l] = percentileWindow(vals); if (!auto()) repaintRaw(); }
}

async function updateExposure() {
  if (!zarr) return;
  const l = rawLevel(), L = zarr.levels[l], W = cv.width, H = cv.height, sp = L.sp, zl = clamp(Math.floor(view.z / 2 ** l), 0, L.shape[0] - 1);
  const y0 = clamp(Math.floor((view.cy - H / 2 * view.p) / sp[1]), 0, L.shape[1] - 1), y1 = clamp(Math.ceil((view.cy + H / 2 * view.p) / sp[1]), y0 + 1, L.shape[1]);
  const x0 = clamp(Math.floor((view.cx - W / 2 * view.p) / sp[2]), 0, L.shape[2] - 1), x1 = clamp(Math.ceil((view.cx + W / 2 * view.p) / sp[2]), x0 + 1, L.shape[2]);
  const bt = banner.start('exposure', `Exposure program on the FOV (level ${l})`);
  try { const r = await recommendExposure(zarr, engine, await getProgram('exposure'), l, zl, y0, y1, x0, x1); rec = { ...r, level: l }; if ($('autoexp').checked) repaintRaw(); banner.finish(bt, true, `window ${r.lo.toFixed(2)}..${r.hi.toFixed(2)}, gamma ${r.gamma.toFixed(2)}`); } catch (e) { rec = null; banner.finish(bt, false, e.message); }
}

async function refresh(opts = {}) {
  updateViewWindow(); await updateExposure();
  if (!zarr) return;
  if (!$('segauto').checked && !opts.force) { curRoute = route(); draw(); setStatus('manual mode: press "Run segmentation now" to segment this view'); return; }
  if (segStale) { segTiles.clear(); segStats.clear(); fovSeg = null; segStale = false; }
  counts.started++;
  const myGen = ++gen, r = route(); curRoute = r;
  const L = zarr.levels[r.level], sp = L.sp, W = cv.width, H = cv.height;
  updateInfo(r);
  if (r.model === 'none' || (r.kind && !['tissue', 'layers'].includes(r.kind) && view.p > 14)) { draw(); setStatus(r.model === 'none' ? 'Raw only.' : r.note, true); return; }
  const prog = await getProgram(r.model); updateInfo(r);
  if (['tissue', 'layers'].includes(r.kind) || (FAMILIES[r.family].invariant && !FAMILIES[r.family].fineOnly)) { await refreshFov(prog, r, myGen); return; }   // scale-invariant programs may coarsen the level until the FOV fits
  fovSeg = null;
  if (await refreshFov(prog, r, myGen, true)) return;          // the model sees exactly the FOV (statistics, window) when it fits in device memory
  r.note = (r.note ? r.note + ' · ' : '') + 'FOV too large for device memory: tiles with per-tile statistics';
  const x0 = view.cx - W / 2 * view.p, y0 = view.cy - H / 2 * view.p, x1 = view.cx + W / 2 * view.p, y1 = view.cy + H / 2 * view.p;
  const nx = Math.ceil(L.shape[2] / TS), ny = Math.ceil(L.shape[1] / TS);
  const gx0 = clamp(Math.floor(x0 / sp[2] / TS), 0, nx - 1), gx1 = clamp(Math.floor(x1 / sp[2] / TS), 0, nx - 1), gy0 = clamp(Math.floor(y0 / sp[1] / TS), 0, ny - 1), gy1 = clamp(Math.floor(y1 / sp[1] / TS), 0, ny - 1);
  const zl = clamp(Math.floor(view.z / 2 ** r.level), 0, L.shape[0] - 1);
  // view-dependent normalization: one window from the whole FOV (all tiles and planes share it)
  let fovWin = null;
  {
    const by0 = clamp(Math.floor(y0 / sp[1]), 0, L.shape[1]), by1 = clamp(Math.ceil(y1 / sp[1]), by0 + 1, L.shape[1]), bx0 = clamp(Math.floor(x0 / sp[2]), 0, L.shape[2]), bx1 = clamp(Math.ceil(x1 / sp[2]), bx0 + 1, L.shape[2]);
    fovWin = auto() ? [rec.lo, rec.hi] : percentileWindow(await zarr.crop(r.level, zl, zl + 1, by0, by1, bx0, bx1));
    if (myGen !== gen) return;
    if (segWinUsed && (Math.abs(fovWin[0] - segWinUsed[0]) > 0.05 || Math.abs(fovWin[1] - segWinUsed[1]) > 0.05)) { segTiles.clear(); segStats.clear(); }
    segWinUsed = fovWin;
  }
  const todo = [];
  for (let gy = gy0; gy <= gy1; gy++) for (let gx = gx0; gx <= gx1; gx++) {
    if (segTiles.has(segKey(r.model, r.level, zl, gy, gx))) continue;
    const cx = (gx + 0.5) * TS * sp[2], cy = (gy + 0.5) * TS * sp[1];
    todo.push({ gy, gx, d: Math.hypot(cx - view.cx, cy - view.cy) });
  }
  todo.sort((a, b) => a.d - b.d);
  const total = (gx1 - gx0 + 1) * (gy1 - gy0 + 1), batch = todo.slice(0, MAX_TILES);
  let done = total - todo.length, t0 = performance.now();
  const bt = banner.start('infer', `${r.model} tiles · level ${r.level}`, batch.length);
  for (const t of batch) {
    if (myGen !== gen) { banner.finish(bt, true, 'superseded by a newer view'); return; }
    try {
      const c = await segTile(prog, r.model, r.level, zl, t.gy, t.gx, fovWin);
      if (myGen !== gen) { banner.finish(bt, true, 'superseded by a newer view'); return; }
      segTiles.set(segKey(r.model, r.level, zl, t.gy, t.gx), c); banner.update(bt, { done: banner_done(bt) }); if (segTiles.size > 400) segTiles.delete(segTiles.keys().next().value);
    } catch (e) { setStatus('Tile failed: ' + e.message, true); banner.finish(bt, false, e.message); return; }
    done++; draw(); setStatus(`${r.model}: ${done}/${total} tiles at level ${r.level} · ${((performance.now() - t0) / (done - (total - todo.length))).toFixed(0)} ms/tile` + (todo.length > MAX_TILES ? ` · ${todo.length - MAX_TILES} more tiles outside the per-refresh budget: pan or zoom to continue` : '') + (r.note ? ' · ' + r.note : ''));
  }
  banner.finish(bt, true, `${done}/${total} tiles in view`);
  if (myGen === gen) setStatus(`${r.model}: ${done}/${total} tiles at level ${r.level}` + (r.note ? ' · ' + r.note : ''), !!r.note);
}

function segSummary() {   // mean stats over the segmented tiles of the current route/plane (for automated evaluation)
  const r = curRoute; if (!r || r.model === 'none') return null;
  const zs = clamp(Math.floor(view.z / 2 ** r.level), 0, zarr.levels[r.level].shape[0] - 1), pre = `${zarr.base}|${r.model}|${r.level}|${zs}|`, acc = {}; let n = 0;
  if (fovSeg && fovSeg.model === r.model) return { model: r.model, level: fovSeg.level, tiles: 1, ...fovSeg.stats };
  for (const [k, st] of segStats) if (k.startsWith(pre)) { n++; for (const [a, b] of Object.entries(st)) acc[a] = (acc[a] || 0) + b; }
  for (const a in acc) acc[a] /= n; return { model: r.model, level: r.level, tiles: n, ...acc };
}
let _tileDone = new Map(); const banner_done = (id) => { const n = (_tileDone.get(id) || 0) + 1; _tileDone.set(id, n); return n; };
function setStatus(t, warn) { if (warn) banner.log(t); const s = $('status'); s.textContent = t; s.className = 'note' + (warn ? ' warn' : ''); }
function updateInfo(r) {
  const L = zarr.levels[r.level], prog = programs[r.model], np = prog ? nParams(prog) : null;
  const rows = [['Backend', backend], ['Pixel size', view.p.toFixed(2) + ' um/px'], ['Program', r.model === 'none' ? '-' : (MODEL_LABEL[r.family] || r.family)],
    ['Scale', r.model === 'none' || !r.sigmaUm ? '-' : `x${r.scale} (sigma ${r.sigmaUm.toFixed(1)} um): ${r.label}`],
    ['Exposure (model)', rec ? `window ${rec.lo.toFixed(2)}..${rec.hi.toFixed(2)} (log), gamma ${rec.gamma.toFixed(2)}` : '-'],
    ['Seg level', r.model === 'none' ? '-' : `${r.level} (${L.sp.map((s) => s.toFixed(1)).join(' x ')} um)`],
    ['Parameters', np ? `${np.nonzero} nonzero (${np.total} dense)` : '-'], ['Layers / heads', prog ? `${prog.layers.length} / ${prog.layers.reduce((a, b) => a + b.heads.length, 0)}` : '-'],
    ['Streamed', `${(zarr.net / 1e6).toFixed(1)} MB (cache ${(zarr.bytes / 1e6).toFixed(0)} MB)`]];
  $('info').innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
  $('legend').innerHTML = MULTI[r.kind] ? MULTI[r.kind].map((k) => `<span style="background:rgb(${MULTI_COLORS[k]})"></span>${k}`).join('') : r.kind === 'layers' ? ['bg', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'WM'].map((k) => `<span style="background:rgb(${COLORS[k]})"></span>${k}`).join('') : r.kind === 'tissue' ? `<span style="background:rgb(90,90,100)"></span>background<span style="background:rgb(255,160,40)"></span>gray<span style="background:rgb(60,140,255)"></span>white` : r.kind === 'vessels' ? '<span style="background:rgb(40,220,255)"></span>vessel' : r.kind === 'cells' ? '<span style="background:rgb(255,60,200)"></span>cell' : '';
}

// ------------------------------------------------------------------------------------------ asset handling
function fitView() {   // frame the whole volume (stitched slab or strip)
  const L = zarr.levels[0], wum = L.shape[2] * L.sp[2], hum = L.shape[1] * L.sp[1];
  view.p = Math.max(wum / cv.width, hum / cv.height) * 1.04; view.cx = wum / 2; view.cy = hum / 2;
  draw(); schedule();
}
async function jumpToTissue() {
  const bt = banner.start('scan', 'Scanning the coarsest level for tissue');
  const top = zarr.top, L = zarr.levels[top], a = await zarr.crop(top, 0, L.shape[0], 0, L.shape[1], 0, L.shape[2]);
  const [Z, Y, X] = L.shape, k = 3; let best = -1, bi = [0, 0, 0];
  const sm = new Float32Array(a.length), at = (z, y, x) => a[(z * Y + y) * X + x];
  for (let z = k; z < Z - k; z += 2) for (let y = k; y < Y - k; y += 2) for (let x = k; x < X - k; x += 2) {
    let s = 0; for (let dz = -k; dz <= k; dz += k) for (let dy = -k; dy <= k; dy += k) for (let dx = -k; dx <= k; dx += k) s += at(z + dz, y + dy, x + dx);
    if (s > best) { best = s; bi = [z, y, x]; } }
  const f = 2 ** top, sp0 = zarr.levels[0].sp;
  view.cx = bi[2] * f * sp0[2]; view.cy = bi[1] * f * sp0[1]; view.z = clamp(bi[0] * f, 0, zarr.levels[0].shape[0] - 1); view.p = 8;
  $('z').value = view.z; $('zval').textContent = view.z; banner.finish(bt, true, `brightest tissue at level ${top}, voxel ${bi.join(',')}`); draw(); schedule();
}

async function openAsset() {
  const id = $('chunk').value; if (!id) return;
  const members = id === 'slab' ? sel().sort((a, b) => a.chunk - b.chunk).map((a) => a.zarr) : null;
  setStatus('Opening zarr (metadata only)...'); gen++; rawTiles.clear(); segTiles.clear();
  const bt = banner.start('open', 'Opening ' + (members ? `the stitched slab (${members.length} strips)` : 'one strip') + ' · ' + $('stain').value + ' sample ' + $('sample').value);
  try {
    const nz = members ? new Slab(members) : new Zarr(id); await nz.init(); zarr = nz; stain = $('stain').value; curRoute = null;
    $('z').max = zarr.levels[0].shape[0] - 1;
    winByLevel = {};
    banner.finish(bt, true, `${zarr.levels.length} pyramid levels, level-0 shape ${zarr.levels[0].shape.join('x')}`);
    if (zarr.dropped && zarr.dropped.length) banner.log(`stitching: ${zarr.dropped.length} strip(s) cover an earlier strip's region (re-acquisition) and are left out; pick "strip N only" to view them`);
  } catch (e) { banner.finish(bt, false, e.message); throw e; }
  await jumpToTissue();
  if (members) fitView();     // stitched slabs open at overview zoom
}

fitCanvas();
fillSamples();
const H = new URLSearchParams(location.hash.slice(1));
const pick = assets.find((a) => a.zarr === H.get('id')) || assets.find((a) => a.stain === 'YO' && a.sample === '90' && a.chunk === 5) || assets[0];
function selectAsset(a, stitched) { $('stain').value = a.stain; fillSamples(); $('sample').value = a.sample; fillSessions(); $('session').value = a.session; fillChunks(); $('chunk').value = stitched ? 'slab' : a.zarr; }
selectAsset(pick, H.get('slab') !== '0');
$('z').value = view.z; $('zval').textContent = view.z;
await openAsset();
if (H.get('p')) {   // deep link from the slab gallery: #id=<zarr>&p=<um/px>&cx=<um>&cy=<um>&z=<level-0 plane>
  Object.assign(view, { p: +H.get('p'), cx: +H.get('cx'), cy: +H.get('cy'), z: clamp(+H.get('z'), 0, zarr.levels[0].shape[0] - 1) });
  $('z').value = view.z; $('zval').textContent = view.z; draw(); schedule();
}
window.__hw = { get zarr() { return zarr; }, fitView, counts, dbn, setMaxTiles: (n) => { MAX_TILES = n; }, resetSeg: () => { segTiles.clear(); segStats.clear(); fovSeg = null; document.getElementById('status').textContent = ''; }, segSummary, adj, repaintRaw, segTiles, segStats, view, draw, schedule, get status() { return document.getElementById('status').textContent; }, get route() { return curRoute; } };

// ------------------------------------------------------------------------------------------ 3D volume view
let vv = null, last3d = null;
try { vv = new VolumeView($('v3cv')); vv.attachMouse(); } catch (e) { $('v3status').textContent = '3D view unavailable: ' + e.message; $('v3go').disabled = true; }
const v3par = () => vv && vv.set({ mode: +$('v3mode').value, thr: +$('v3thr').value, op: +$('v3op').value, steps: +$('v3steps').value, segOp: +$('v3segop').value, seg: true });
for (const id of ['v3mode', 'v3thr', 'v3op', 'v3steps', 'v3segop']) $(id).oninput = $(id).onchange = v3par;
$('v3reset').onclick = () => vv && vv.set({ az: 0.6, el: 0.45, zoom: 1 });
function quantize3d() {
  const { crop, win } = last3d, d8 = new Uint8Array(crop.length), r = win[1] - win[0];
  for (let i = 0; i < crop.length; i++) d8[i] = Math.round(adjustU(clamp((crop[i] - win[0]) / r, 0, 1), effAdj()) * 255);
  return d8;
}
function reupload3d() { if (!vv || !last3d) return; vv.setVolume(quantize3d(), last3d.dims, last3d.spacing, last3d.labels, last3d.palette); v3par(); }
for (const id of ['bri', 'con', 'gam']) $(id).addEventListener('input', reupload3d);
$('adjreset').addEventListener('click', reupload3d);

async function build3D() {
  const bt = banner.start('3d', '3D volume from this view');
  try { await build3DInner(bt); banner.finish(bt, true); } catch (e) { banner.finish(bt, false, e.message); throw e; }
}
async function build3DInner(bt) {
  if (!vv || !zarr) return;
  const cap = +$('v3size').value, nzp = +$('v3depth').value, W = cv.width, H = cv.height;
  const ux0 = view.cx - W / 2 * view.p, ux1 = view.cx + W / 2 * view.p, uy0 = view.cy - H / 2 * view.p, uy1 = view.cy + H / 2 * view.p;
  const wantSeg = $('v3seg').checked, rt = wantSeg ? route() : null, fit = $('v3fit').value === 'fit';
  let l = wantSeg && rt.model !== 'none' ? rt.level : rawLevel(), box;      // with segmentation: the level the program's scale calls for
  for (;;) {
    const L = zarr.levels[l], sp = L.sp;
    box = [clamp(Math.floor(uy0 / sp[1]), 0, L.shape[1]), clamp(Math.ceil(uy1 / sp[1]), 0, L.shape[1]), clamp(Math.floor(ux0 / sp[2]), 0, L.shape[2]), clamp(Math.ceil(ux1 / sp[2]), 0, L.shape[2])];
    if (!fit || Math.max(box[1] - box[0], box[3] - box[2]) <= cap || l >= zarr.top) break; l++;
  }
  for (const [a, b, n] of [[0, 1, cap], [2, 3, cap]]) if (box[b] - box[a] > n) { const c = (box[a] + box[b]) >> 1; box[a] = c - (n >> 1); box[b] = box[a] + n; }
  const L = zarr.levels[l], sp = L.sp, zl = clamp(Math.floor(view.z / 2 ** l), 0, L.shape[0] - 1);
  const z0 = clamp(zl - (nzp >> 1), 0, Math.max(0, L.shape[0] - nzp)), z1 = Math.min(L.shape[0], z0 + nzp);
  if (box[1] <= box[0] || box[3] <= box[2]) { $('v3status').textContent = 'FOV outside the volume'; return; }
  $('v3status').textContent = `Streaming ${box[3] - box[2]}x${box[1] - box[0]}x${z1 - z0} voxels at level ${l}...`;
  banner.update(bt, { detail: "streaming the volume" });
  const crop = await zarr.crop(l, z0, z1, box[0], box[1], box[2], box[3]), dims = [z1 - z0, box[1] - box[0], box[3] - box[2]];
  const win = viewWin(crop); last3d = { crop, win, dims, spacing: [sp[0], sp[1], sp[2]], labels: null, palette: null };
  let note = '';
  if ($('v3seg').checked) {
    const r = rt, m = r.model;
    if (m === 'none') note = ' · no model selected';
    else {
      const prog = await getProgram(m), kind = modelIndex[m].kind, [hz] = haloFor(prog, sp), tokens = (dims[0] + 2 * hz) * dims[1] * dims[2];
      if (engine && tokens > engine.maxTokens(prog.d) * 0.6 || !engine && tokens > 150000) note = ' · segmentation skipped (volume too large for the device)';
      else {
        $('v3status').textContent = `Running ${m} on the volume (${dims.join('x')})...`;
        const ea = effAdj();
        const sv = await segVolume(zarr, engine, prog, l, z0, z1, box[0], box[1], box[2], box[3], (v) => adjustLogVol(v, win[0], win[1], ea), { halo: false });
        last3d.labels = labelVolume(kind, sv); last3d.palette = paletteFor(kind, sv.out);
        note = ` · ${m} on level ${l}` + (r.level !== l ? ` (its native level is ${r.level}: scale mismatch)` : '');
      }
    }
  }
  reupload3d();
  $('v3status').textContent = `3D: ${dims[2]}x${dims[1]}x${dims[0]} voxels at level ${l} (${sp.map((x) => x.toFixed(1)).join(' x ')} um)${note}. Drag to rotate, wheel to zoom.`;
}
$('v3go').onclick = () => build3D().catch((e) => { $('v3status').textContent = '3D failed: ' + e.message; });
window.__hw.build3D = build3D; window.__hw.vv = vv;
