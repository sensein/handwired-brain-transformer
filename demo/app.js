import { parseProgram, nParams, haloFor, GPUEngine, runCPU } from './engine.js';
import { Zarr, clamp, S3 } from './zarr.js';
import { segRegion, renderSegCanvas, toCanvas, COLORS } from './seg.js';

let TS = 128;                   // segmentation tile size (in-plane voxels); per-program via models/index.json
const MAX_TILES = 48;           // tiles segmented per refresh (nearest the view center first)
const $ = (id) => document.getElementById(id);

// ------------------------------------------------------------------------------------------ state
const assets = await (await fetch('assets.json')).json();
const modelIndex = await (await fetch('models/index.json')).json();
const MODEL_LABEL = { tissue: 'Tissue (bg / gray / white)', layers: 'Cortical layers I-VI (coarse)', cells: 'Cells (nuclei / somata)', vessels_13dir: 'Vessels, 13 directions', vessels_26dir: 'Vessels, 26 directions' };
const programs = {};
async function getProgram(name) { return programs[name] ||= parseProgram(await (await fetch(`models/${modelIndex[name].file}`)).text()); }

let engine = null, backend = 'none';
try { engine = await GPUEngine.create(); backend = 'WebGPU'; } catch (e) { backend = 'CPU reference (no WebGPU: slow, small tiles only)'; }

const view = { cx: 0, cy: 0, p: 100, z: 1024 };       // center in um (x,y); um per screen pixel; level-0 z index
let zarr = null, stain = '', winByLevel = {}, gen = 0;
const rawTiles = new Map(), segTiles = new Map();
const cv = $('cv'), ctx = cv.getContext('2d');

function fitCanvas() { const r = cv.getBoundingClientRect(), d = Math.min(devicePixelRatio || 1, 2); cv.width = Math.round(r.width * d); cv.height = Math.round(r.height * d); view.dpr = d; }
addEventListener('resize', () => { fitCanvas(); schedule(); });

// ------------------------------------------------------------------------------------------ UI wiring
for (const s of [...new Set(assets.map((a) => a.stain))].sort((a, b) => (['YO', 'NN', 'LEC'].indexOf(b) + 1) - (['YO', 'NN', 'LEC'].indexOf(a) + 1) || a.localeCompare(b)))
  $('stain').add(new Option(s, s));
$('model').add(new Option('Auto (follows zoom)', 'auto'));
for (const k of Object.keys(modelIndex)) $('model').add(new Option(`${MODEL_LABEL[k] || k} - ${modelIndex[k].nonzero} params`, k));
$('model').add(new Option('None (raw only)', 'none'));
function fillSamples() { const ss = [...new Set(assets.filter((a) => a.stain === $('stain').value).map((a) => a.sample))].sort((a, b) => +a - +b);
  $('sample').innerHTML = ''; ss.forEach((s) => $('sample').add(new Option('sample ' + s, s))); fillChunks(); }
function fillChunks() { const cs = assets.filter((a) => a.stain === $('stain').value && a.sample === $('sample').value).sort((a, b) => a.chunk - b.chunk);
  $('chunk').innerHTML = ''; cs.forEach((a) => $('chunk').add(new Option('chunk ' + a.chunk, a.zarr))); }
$('stain').onchange = () => { fillSamples(); openAsset(); };
$('sample').onchange = () => { fillChunks(); openAsset(); };
$('chunk').onchange = () => openAsset();
$('model').onchange = () => { segTiles.clear(); schedule(); };
$('opacity').oninput = () => draw();
$('z').oninput = () => { view.z = +$('z').value; $('zval').textContent = view.z; schedule(); };
$('reset').onclick = () => jumpToTissue();

let dragging = null;
cv.addEventListener('pointerdown', (e) => { cv.setPointerCapture(e.pointerId); dragging = { x: e.clientX, y: e.clientY, cx: view.cx, cy: view.cy }; });
cv.addEventListener('pointermove', (e) => { if (!dragging) return; const k = view.p * (cv.width / cv.getBoundingClientRect().width);
  view.cx = dragging.cx - (e.clientX - dragging.x) * k; view.cy = dragging.cy - (e.clientY - dragging.y) * k; draw(); schedule(); });
cv.addEventListener('pointerup', () => { dragging = null; });
cv.addEventListener('wheel', (e) => { e.preventDefault(); const r = cv.getBoundingClientRect(), k = cv.width / r.width;
  const mx = (e.clientX - r.left) * k, my = (e.clientY - r.top) * k, f = Math.exp(e.deltaY * 0.0015);
  const np = clamp(view.p * f, 0.4 * zarr.levels[0].sp[2] / view.dpr, 400);
  view.cx += (mx - cv.width / 2) * (view.p - np); view.cy += (my - cv.height / 2) * (view.p - np); view.p = np; draw(); schedule(); }, { passive: false });

// ------------------------------------------------------------------------------------------ routing by zoom
// p = um per screen pixel. Fine programs run at level 0 (their sigmas are physical and were checked there);
// tissue was checked at level 4 and is run at level 4/5.
function route() {
  const sel = $('model').value;
  const sp0 = zarr.levels[0].sp;
  let model = sel, note = '';
  if (sel === 'auto') {
    if (view.p >= 8.5) model = view.p < 30 ? 'layers' : 'tissue';
    else model = ['LEC', 'CD31'].includes(stain) ? 'vessels_26dir' : 'cells';
  }
  if (model === 'none') return { model, level: 0 };
  const kind = modelIndex[model].kind;
  TS = modelIndex[model].tile || 128;
  let level = 0;
  if (kind === 'layers') level = Math.min(5, zarr.top);
  else if (kind === 'tissue') level = clamp(Math.ceil(Math.log2(view.p / sp0[2])), Math.min(4, zarr.top), Math.min(5, zarr.top));
  else if (view.p > 14) note = 'zoom in for ' + kind + ' (needs about ' + sp0[2].toFixed(1) + ' um voxels)';
  return { model, level, kind, note };
}

// ------------------------------------------------------------------------------------------ rendering
function rawLevel() { let l = 0; for (let i = 0; i < zarr.levels.length; i++) if (zarr.levels[i].sp[2] <= view.p * 1.0) l = i; return l; }

function rawTile(l, zl, cy, cx) {
  const key = `${zarr.base}|${l}|${zl}|${cy}|${cx}`, hit = rawTiles.get(key);
  if (hit) return hit;
  if (hit === null) return null;
  rawTiles.set(key, null);
  const L = zarr.levels[l], [cz, cyS, cxS] = L.chunks;
  zarr.chunk(l, Math.floor(zl / cz), cy, cx).then((data) => {
    const base = ((zl % cz) * cyS) * cxS;
    if (!winByLevel[l]) {   // display window from the first chunk plane streamed at this level (2nd..99.7th percentile of log I)
      const v = new Float32Array(cxS * cyS); for (let i = 0; i < v.length; i++) v[i] = Math.log(data[base + i] + 1); v.sort();
      const lo = v[Math.floor(v.length * 0.02)], hi = v[Math.floor(v.length * 0.997)]; winByLevel[l] = [lo, Math.max(hi, lo + 1)];
    }
    const win = winByLevel[l];
    const c = toCanvas(null, cxS, cyS, (px) => { for (let i = 0; i < cxS * cyS; i++) {
      const v = clamp((Math.log(data[base + i] + 1) - win[0]) / (win[1] - win[0]), 0, 1) * 255; px[4 * i] = px[4 * i + 1] = px[4 * i + 2] = v; px[4 * i + 3] = 255; } });
    rawTiles.set(key, c); if (rawTiles.size > 900) rawTiles.delete(rawTiles.keys().next().value); draw();
  });
  return null;
}

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

let curRoute = null;
function draw() {
  if (!zarr) return;
  const W = cv.width, H = cv.height;
  ctx.fillStyle = '#000'; ctx.fillRect(0, 0, W, H); ctx.imageSmoothingEnabled = false;
  const l = rawLevel(), zl = clamp(Math.floor(view.z / 2 ** l), 0, zarr.levels[l].shape[0] - 1);
  drawGrid(l, (gy, gx) => rawTile(l, zl, gy, gx), 1, zarr.levels[l].chunks[1], zarr.levels[l].chunks[2]);
  const r = curRoute;
  if (r && r.model !== 'none') {
    const zs = clamp(Math.floor(view.z / 2 ** r.level), 0, zarr.levels[r.level].shape[0] - 1);
    drawGrid(r.level, (gy, gx) => segTiles.get(segKey(r.model, r.level, zs, gy, gx)) || null, +$('opacity').value / 100, TS, TS);
  }
  $('hud').textContent = `${view.p.toFixed(2)} um/px · level ${l} (raw) · z ${view.z}`;
  const barPx = 100 * view.dpr, um = barPx * view.p, nice = [1, 2, 5].map((m) => m * 10 ** Math.floor(Math.log10(um))).reduce((a, b) => Math.abs(b - um) < Math.abs(a - um) ? b : a);
  const sb = $('scale'); sb.firstElementChild.style.width = (nice / view.p / view.dpr) + 'px'; sb.lastElementChild.textContent = nice >= 1000 ? (nice / 1000) + ' mm' : nice + ' um';
}

// ------------------------------------------------------------------------------------------ segmentation tiles
const segKey = (m, l, z, gy, gx) => `${zarr.base}|${m}|${l}|${z}|${gy}|${gx}`;
const COLORS = { bg: [90, 90, 100], gray: [255, 160, 40], white: [60, 140, 255] };
let timer = null;
function schedule() { clearTimeout(timer); timer = setTimeout(refresh, 180); }

async function segTile(prog, name, l, zl, gy, gx) {
  const L = zarr.levels[l], y0 = gy * TS, x0 = gx * TS, y1 = Math.min(y0 + TS, L.shape[1]), x1 = Math.min(x0 + TS, L.shape[2]);
  const r = await segRegion(zarr, engine, prog, l, zl, y0, y1, x0, x1);
  return renderSegCanvas(modelIndex[name].kind, r);
}

async function refresh() {
  if (!zarr) return;
  const myGen = ++gen, r = route(); curRoute = r;
  const L = zarr.levels[r.level], sp = L.sp, W = cv.width, H = cv.height;
  updateInfo(r);
  if (r.model === 'none' || (r.kind && !['tissue', 'layers'].includes(r.kind) && view.p > 14)) { draw(); setStatus(r.model === 'none' ? 'Raw only.' : r.note, true); return; }
  const prog = await getProgram(r.model); updateInfo(r);
  const x0 = view.cx - W / 2 * view.p, y0 = view.cy - H / 2 * view.p, x1 = view.cx + W / 2 * view.p, y1 = view.cy + H / 2 * view.p;
  const nx = Math.ceil(L.shape[2] / TS), ny = Math.ceil(L.shape[1] / TS);
  const gx0 = clamp(Math.floor(x0 / sp[2] / TS), 0, nx - 1), gx1 = clamp(Math.floor(x1 / sp[2] / TS), 0, nx - 1), gy0 = clamp(Math.floor(y0 / sp[1] / TS), 0, ny - 1), gy1 = clamp(Math.floor(y1 / sp[1] / TS), 0, ny - 1);
  const zl = clamp(Math.floor(view.z / 2 ** r.level), 0, L.shape[0] - 1);
  const todo = [];
  for (let gy = gy0; gy <= gy1; gy++) for (let gx = gx0; gx <= gx1; gx++) {
    if (segTiles.has(segKey(r.model, r.level, zl, gy, gx))) continue;
    const cx = (gx + 0.5) * TS * sp[2], cy = (gy + 0.5) * TS * sp[1];
    todo.push({ gy, gx, d: Math.hypot(cx - view.cx, cy - view.cy) });
  }
  todo.sort((a, b) => a.d - b.d);
  const total = (gx1 - gx0 + 1) * (gy1 - gy0 + 1), batch = todo.slice(0, MAX_TILES);
  let done = total - todo.length, t0 = performance.now();
  for (const t of batch) {
    if (myGen !== gen) return;
    try {
      const c = await segTile(prog, r.model, r.level, zl, t.gy, t.gx);
      if (myGen !== gen) return;
      segTiles.set(segKey(r.model, r.level, zl, t.gy, t.gx), c); if (segTiles.size > 400) segTiles.delete(segTiles.keys().next().value);
    } catch (e) { setStatus('Tile failed: ' + e.message, true); return; }
    done++; draw(); setStatus(`${r.model}: ${done}/${total} tiles at level ${r.level} · ${((performance.now() - t0) / (done - (total - todo.length))).toFixed(0)} ms/tile` + (todo.length > MAX_TILES ? ` · ${todo.length - MAX_TILES} more tiles outside the per-refresh budget: pan or zoom to continue` : '') + (r.note ? ' · ' + r.note : ''));
  }
  if (myGen === gen) setStatus(`${r.model}: ${done}/${total} tiles at level ${r.level}` + (r.note ? ' · ' + r.note : ''), !!r.note);
}

function setStatus(t, warn) { const s = $('status'); s.textContent = t; s.className = 'note' + (warn ? ' warn' : ''); }
function updateInfo(r) {
  const L = zarr.levels[r.level], prog = programs[r.model], np = prog ? nParams(prog) : null;
  const rows = [['Backend', backend], ['Pixel size', view.p.toFixed(2) + ' um/px'], ['Program', r.model === 'none' ? '-' : (MODEL_LABEL[r.model] || r.model)],
    ['Seg level', r.model === 'none' ? '-' : `${r.level} (${L.sp.map((s) => s.toFixed(1)).join(' x ')} um)`],
    ['Parameters', np ? `${np.nonzero} nonzero (${np.total} dense)` : '-'], ['Layers / heads', prog ? `${prog.layers.length} / ${prog.layers.reduce((a, b) => a + b.heads.length, 0)}` : '-'],
    ['Streamed', `${(zarr.net / 1e6).toFixed(1)} MB (cache ${(zarr.bytes / 1e6).toFixed(0)} MB)`]];
  $('info').innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
  $('legend').innerHTML = r.kind === 'layers' ? ['bg', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'WM'].map((k) => `<span style="background:rgb(${COLORS[k]})"></span>${k}`).join('') : r.kind === 'tissue' ? `<span style="background:rgb(90,90,100)"></span>background<span style="background:rgb(255,160,40)"></span>gray<span style="background:rgb(60,140,255)"></span>white` : r.kind === 'vessels' ? '<span style="background:rgb(40,220,255)"></span>vessel' : r.kind === 'cells' ? '<span style="background:rgb(255,60,200)"></span>cell' : '';
}

// ------------------------------------------------------------------------------------------ asset handling
async function jumpToTissue() {
  const top = zarr.top, L = zarr.levels[top], a = await zarr.crop(top, 0, L.shape[0], 0, L.shape[1], 0, L.shape[2]);
  const [Z, Y, X] = L.shape, k = 3; let best = -1, bi = [0, 0, 0];
  const sm = new Float32Array(a.length), at = (z, y, x) => a[(z * Y + y) * X + x];
  for (let z = k; z < Z - k; z += 2) for (let y = k; y < Y - k; y += 2) for (let x = k; x < X - k; x += 2) {
    let s = 0; for (let dz = -k; dz <= k; dz += k) for (let dy = -k; dy <= k; dy += k) for (let dx = -k; dx <= k; dx += k) s += at(z + dz, y + dy, x + dx);
    if (s > best) { best = s; bi = [z, y, x]; } }
  const f = 2 ** top, sp0 = zarr.levels[0].sp;
  view.cx = bi[2] * f * sp0[2]; view.cy = bi[1] * f * sp0[1]; view.z = clamp(bi[0] * f, 0, zarr.levels[0].shape[0] - 1); view.p = 8;
  $('z').value = view.z; $('zval').textContent = view.z; draw(); schedule();
}

async function openAsset() {
  const id = $('chunk').value; if (!id) return;
  setStatus('Opening zarr (metadata only)...'); gen++; rawTiles.clear(); segTiles.clear();
  const nz = new Zarr(id); await nz.init(); zarr = nz; stain = $('stain').value; curRoute = null;
  $('z').max = zarr.levels[0].shape[0] - 1;
  winByLevel = {};
  await jumpToTissue();
}

fitCanvas();
fillSamples();
const H = new URLSearchParams(location.hash.slice(1));
const pick = assets.find((a) => a.zarr === H.get('id')) || assets.find((a) => a.stain === 'YO' && a.sample === '90' && a.chunk === 5) || assets[0];
$('stain').value = pick.stain; fillSamples(); $('sample').value = pick.sample; fillChunks(); $('chunk').value = pick.zarr;
$('z').value = view.z; $('zval').textContent = view.z;
await openAsset();
if (H.get('p')) {   // deep link from the slab gallery: #id=<zarr>&p=<um/px>&cx=<um>&cy=<um>&z=<level-0 plane>
  Object.assign(view, { p: +H.get('p'), cx: +H.get('cx'), cy: +H.get('cy'), z: clamp(+H.get('z'), 0, zarr.levels[0].shape[0] - 1) });
  $('z').value = view.z; $('zval').textContent = view.z; draw(); schedule();
}
window.__hw = { view, draw, schedule, get status() { return document.getElementById('status').textContent; }, get route() { return curRoute; } };
