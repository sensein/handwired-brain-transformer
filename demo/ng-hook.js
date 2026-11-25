// Neuroglancer bridge: runs inside our own Neuroglancer build (same origin on GitHub Pages). Reads the view from
// window.viewer, runs a weight-space-programmed model on exactly that field of view with the WebGPU engine, and writes the
// detections back as a Neuroglancer annotation layer. Triggered by a button, or automatically (debounced) as you navigate.
import { parseProgram, GPUEngine, haloFor } from './engine.js';
import { parseHier, isHier } from './hier.js';
import { Zarr, Slab, clamp } from './zarr.js';
import { segRegion, segVolume, labelVolume, adjustLogVol, recommendExposure, IDENT, haloOf } from './seg.js';
import { scaledModel, FAMILIES } from './route.js';
import { components3 } from './instances.js';
import { REF, isRef, refTile, fovNorm, ndInfo } from './nd/nd-seg.js';

const LAYER = 'hwseg detections';
const axisIdx = (d) => { const k = Object.keys(d); return { x: k.indexOf('x'), y: k.indexOf('y'), z: k.indexOf('z') }; };   // dimension order differs between the zarr state (t,z,y,x) and the virtual volume (x,y,z)

// ---- virtual stitched-slab volume: this page computes the chunks the service worker (sw.js) is asked for
const slabCache = {};
const getSlab = (ids, seams) => { const key = ids.join('+') + (seams ? 's' : ''); return (slabCache[key] ||= (async () => { const z = new Slab(ids); await z.init(); if (seams) await z.setSeams(true); return z; })()); };
async function serveVirtual(idsStr, path) {
  const seams = idsStr.endsWith('~s'), ids = idsStr.replace(/~s$/, '').split('+'), z = await getSlab(ids, seams);
  if (path === 'info') {
    const scales = z.levels.map((L, l) => ({ key: String(l), size: [L.shape[2], L.shape[1], L.shape[0]], resolution: [L.sp[2] * 1000, L.sp[1] * 1000, L.sp[0] * 1000], chunk_sizes: [[128, 128, 32]], encoding: 'raw', voxel_offset: [0, 0, 0] }));
    return { type: 'application/json', body: new TextEncoder().encode(JSON.stringify({ '@type': 'neuroglancer_multiscale_volume', type: 'image', data_type: 'uint16', num_channels: 1, scales })).buffer };
  }
  const m = path.match(/^(\d+)\/(\d+)-(\d+)_(\d+)-(\d+)_(\d+)-(\d+)$/); if (!m) return { status: 404, type: 'text/plain', body: new ArrayBuffer(0) };
  const [l, x0, x1, y0, y1, z0, z1] = m.slice(1).map(Number), L = z.levels[l], X1 = Math.min(x1, L.shape[2]), Y1 = Math.min(y1, L.shape[1]), Z1 = Math.min(z1, L.shape[0]);
  const v = await z.crop(l, z0, Z1, y0, Y1, x0, X1), out = new Uint16Array(v.length);          // log(I+1) -> uint16 counts; (z,y,x) C-order = Fortran (x fastest)
  for (let i = 0; i < v.length; i++) out[i] = Math.max(0, Math.min(65535, Math.round(Math.exp(v[i]) - 1)));
  return { type: 'application/octet-stream', body: out.buffer };
}

// ---- segmentation volume: model output as a Neuroglancer `precomputed` segmentation layer, served by the service worker from memory
let segAlpha = 0.55;
const pyApproved = new Set(), pyPending = [];                                   // Python-carrying segmentations: approved ids, and ones waiting for the person's approval
const segStore = {}, segJobs = {}, segMetaC = {};
// The layer's source URL carries the run parameters (model, level, box, z slab, exposure, image source) as the volume id, so a
// shared link regenerates the segmentation on demand in whoever opens it; nothing has to be uploaded or stored.
const b64e = (o) => btoa(unescape(encodeURIComponent(JSON.stringify(o)))).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
const b64d = (t) => JSON.parse(decodeURIComponent(escape(atob(t.replace(/-/g, '+').replace(/_/g, '/')))));
async function segMeta(P) { const lvl = P.kind === 'py' ? P.view.level : P.level, src = await openSource(P.kind === 'py' ? P.view.src : P.src), L = src.z.levels[lvl], f = 2 ** lvl;
  return { size: [L.shape[2], L.shape[1], L.shape[0]], res: [L.sp[2] * 1000, L.sp[1] * 1000, L.sp[0] * 1000], vo: [Math.round(src.off[2] / f), Math.round(src.off[1] / f), Math.round(src.off[0] / f)] }; }
async function serveSeg(idFull, path) {
  const id = idFull.split('.')[0]; let P; try { P = b64d(id); } catch (e) { return { status: 404, type: 'text/plain', body: new ArrayBuffer(0) }; }
  const S = await (segMetaC[id] ||= segMeta(P));
  if (path === 'info') return { type: 'application/json', body: new TextEncoder().encode(JSON.stringify({ '@type': 'neuroglancer_multiscale_volume', type: 'segmentation', data_type: 'uint32', num_channels: 1,
    scales: [{ key: '0', size: S.size, resolution: S.res, chunk_sizes: [[64, 64, 16]], encoding: 'raw', voxel_offset: S.vo }] })).buffer };
  const m = path.match(/^0\/(-?\d+)-(-?\d+)_(-?\d+)-(-?\d+)_(-?\d+)-(-?\d+)$/); if (!m) return { status: 404, type: 'text/plain', body: new ArrayBuffer(0) };
  const [x0, x1, y0, y1, z0, z1] = m.slice(1).map(Number), W = x1 - x0, H = y1 - y0, out = new Uint32Array(W * H * (z1 - z0));    // x fastest
  if (P.kind === 'py' && !pyApproved.has(id)) {                                 // a link carrying Python code: never run it without the person's say-so
    if (!pyPending.some((p) => p.id === id)) { pyPending.push({ id, name: P.name, code: P.code }); say('this link contains Python code for a segmentation layer: review and approve it to run it'); }
    return { type: 'application/octet-stream', body: out.buffer };
  }
  if (P.kind === 'py' && !segJobs[id]) say('python: seg request ' + path + ' -> starting regeneration');
  const blocks = await (segJobs[id] ||= computeSeg(P).then((r) => r.blocks));      // regenerated on first request when this page did not compute it
  for (const b of blocks) {                                                      // blocks hold local voxel indices of the level
    const gx0 = Math.max(x0, b.x0 + S.vo[0]), gx1 = Math.min(x1, b.x1 + S.vo[0]), gy0 = Math.max(y0, b.y0 + S.vo[1]), gy1 = Math.min(y1, b.y1 + S.vo[1]), gz0 = Math.max(z0, b.z0 + S.vo[2]), gz1 = Math.min(z1, b.z1 + S.vo[2]);
    if (gx0 >= gx1 || gy0 >= gy1 || gz0 >= gz1) continue; const bw = b.x1 - b.x0, bh = b.y1 - b.y0;
    for (let z = gz0; z < gz1; z++) for (let y = gy0; y < gy1; y++) { let o = ((z - z0) * H + (y - y0)) * W + (gx0 - x0), q = ((z - S.vo[2] - b.z0) * bh + (y - S.vo[1] - b.y0)) * bw + (gx0 - S.vo[0] - b.x0); for (let x = gx0; x < gx1; x++) out[o++] = b.lab[q++]; }
  }
  return { type: 'application/octet-stream', body: out.buffer };
}
function putSeg(P, blocks) {                                                   // register a freshly computed segmentation and return its layer source URL
  const id = b64e(P); const keep = Object.keys(segJobs); if (keep.length > 4) { delete segJobs[keep[0]]; delete segMetaC[keep[0]]; }
  segJobs[id] = Promise.resolve(blocks);
  return 'precomputed://' + new URL('../virtual/seg/' + id, import.meta.url).href;
}
const segLayer = (name, source, extra = {}) => ({ type: 'segmentation', name, source, selectedAlpha: segAlpha, notSelectedAlpha: 0, tab: 'rendering', ...extra });
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.addEventListener('message', async (ev) => {
    const m = ev.data; if (!m || m.kind !== 'virtual') return; const port = ev.ports[0];
    try { const r = m.which === 'seg' ? await serveSeg(m.ids, m.path) : await serveVirtual(m.ids, m.path); port.postMessage(r, [r.body]); }
    catch (e) { say('virtual volume request failed: ' + String(e.message || e).slice(0, 300)); port.postMessage({ status: 500, type: 'text/plain', body: new TextEncoder().encode(String(e)).buffer }); }
  });
  navigator.serviceWorker.register(new URL('sw.js', import.meta.url).href).then(async () => {
    await navigator.serviceWorker.ready;   // a shared link opened directly: the first load is not controlled yet, so reload once
    if (!navigator.serviceWorker.controller && /virtual\/(slab|seg)/.test(location.href) && !sessionStorage.getItem('hw-sw-reload')) { sessionStorage.setItem('hw-sw-reload', '1'); location.reload(); }
  }).catch(() => {});
}
const el = (t, css, html) => { const e = document.createElement(t); if (css) e.style.cssText = css; if (html) e.innerHTML = html; return e; };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
while (!window.viewer) await sleep(100);
const viewer = window.viewer;

// ---- small UI
const box = el('div', 'position:fixed;left:8px;bottom:8px;z-index:9999;background:rgba(20,21,26,.92);color:#e8e8e4;font:12px/1.4 system-ui;padding:8px 10px;border:1px solid #444;border-radius:8px;max-width:340px');
box.innerHTML = '<b>hwseg</b> programmed model in the loop<br><select id="hw-m"><option value="fable_blobs_v2">cells (nuclei / somata, hand-written v2)</option><option value="fable_vessels_v2">vessels (LEC, hand-written v2)</option><option value="cells">cells (builder)</option><option value="vessels_26dir">vessels (builder)</option><option value="regions">tissue outline + cortex / white-matter boundary (contours)</option><option value="hier_cells">cells (hierarchical, any zoom)</option><option value="hier_vessels">vessels (hierarchical, any zoom, LEC)</option><option value="hier_regions">tissue / depth bands / white matter (hierarchical, any zoom)</option><optgroup label="Neurodesk reference models"><option value="nd_cellpose">Cellpose 3 nuclei (reference)</option><option value="nd_stardist">StarDist 2D fluo (reference)</option><option value="nd_cellfinder">cellfinder candidates (reference)</option><option value="nd_vesselboost">VesselBoost (reference, experimental)</option></optgroup></select> <select id="hw-nz" title="planes in z the detector runs on (3-D); 1 = displayed plane only"><option value="1">1 plane (2-D)</option><option value="5">5 planes</option><option value="9" selected>9 planes</option><option value="15">15 planes</option><option value="25">25 planes</option></select> <button id="hw-run">Detect in this view</button> <label><input type="checkbox" id="hw-auto"> auto</label><div id="hw-s" style="color:#9a9a94;margin-top:4px">idle</div>';
const EMBEDDED = new URLSearchParams(location.search).has('embedded');    // the shell page drives us through window.__hwNg
document.body.append(box); if (EMBEDDED) box.style.display = 'none';
window.__hwNg = { status: '', runs: 0, busy: false, events: [], last: null };
const say = (t) => { document.getElementById('hw-s').textContent = t; window.__hwNg.status = t; window.__hwNg.events.push({ t: Date.now(), msg: t }); if (window.__hwNg.events.length > 200) window.__hwNg.events.shift(); };

let engine = null; try { engine = await GPUEngine.create(); } catch (e) { say('no WebGPU: detection unavailable'); }
const HERE = (p) => new URL(p, import.meta.url).href;                     // the page is /ng/index.html; models live next to this module
const index = await (await fetch(HERE('models/index.json'))).json(), programs = {};
const getProgram = async (k) => programs[k] ||= (async () => { const t = await (await fetch(HERE(`models/${index[k].file}`))).text(); return isHier(t) ? parseHier(t) : parseProgram(t); })();
const sources = {};

// ---- read the view: field of view recovered from the data coordinates Neuroglancer reports under the panel corners
// Field of view from the view's own state: position (centre), crossSectionScale (voxels of the first displayed dimension per CSS pixel) and the
// panel size. No mouse events are needed (the mouse-hover probe it replaces was flaky: synthetic events did not always update the mouse position).
async function probeFov() {
  const panel = [...viewer.display.panels].find((p) => p.element && p.element.getBoundingClientRect().width > 100); if (!panel) throw new Error('no cross-section panel');
  const r = panel.element.getBoundingClientRect(), j = viewer.state.toJSON(), d = j.dimensions || {}, keys = Object.keys(d), s = j.crossSectionScale, pos = j.position;
  const ix = keys.indexOf('x'), iy = keys.indexOf('y'); const dispOk = Array.isArray(j.displayDimensions) ? j.displayDimensions.slice(0, 2).join() === 'x,y' : true;
  if (pos && s > 0 && ix >= 0 && iy >= 0 && dispOk && keys.length === pos.length) {
    const ex = s * r.width, ey = s * (d.x[0] / d.y[0]) * r.height;                 // extents in voxels of x and y (physical scale per pixel is isotropic)
    const tl = pos.slice(), br = pos.slice(); tl[ix] = pos[ix] - ex / 2; br[ix] = pos[ix] + ex / 2; tl[iy] = pos[iy] - ey / 2; br[iy] = pos[iy] + ey / 2;
    return { tl, br, c: pos.slice(), w: r.width, h: r.height };
  }
  // fallback: hover just inside the panel corners and extrapolate
  const el_ = panel.element;
  const at = async (fx, fy) => { el_.dispatchEvent(new MouseEvent('mousemove', { clientX: r.left + fx * r.width, clientY: r.top + fy * r.height, bubbles: true })); await sleep(120); return Array.from(viewer.mouseState.position); };
  const c = await at(0.5, 0.5), a = await at(0.02, 0.02), b = await at(0.98, 0.98);
  const tl = c.map((v, i) => v - (v - a[i]) / 0.96), br = c.map((v, i) => v + (b[i] - v) / 0.96);
  return { tl, br, c, w: r.width, h: r.height };
}
function imageSource(j) {
  const L = (j.layers || []).find((l) => l.type === 'image'); if (!L) throw new Error('no image layer');
  const txt = JSON.stringify(L.source), v = txt.match(/virtual\/slab\/([0-9a-f+-]+)(~s)?/);
  if (v) return { ids: v[1].split('+'), virtual: true, seams: !!v[2] };
  const ids = [...new Set([...txt.matchAll(/dandiarchive\.s3\.amazonaws\.com\/zarr\/([0-9a-f-]{36})/g)].map((m) => m[1]))];
  if (!ids.length) throw new Error('image layer is not a DANDI zarr source'); return { ids, virtual: false, seams: false };
}
const openedOnce = new Set();
async function openSource({ ids, virtual, seams }) {
  const okey = ids.join('+') + (virtual ? 'v' : ''); if (!openedOnce.has(okey)) { openedOnce.add(okey); say(`opening ${ids.length > 1 || virtual ? `stitched slab (${ids.length} strips)` : 'strip'}...`); }   // not on every chunk request
  const z = virtual ? await getSlab(ids, seams) : (ids.length > 1 ? await getSlab(ids, false) : (sources[ids[0]] ||= await (async () => { const q = new Zarr(ids[0]); await q.init(); return q; })()));
  // zarr sources: Neuroglancer places each zarr at its OME translation (global voxel = local index + translation / voxel size);
  // the virtual volume's own grid starts at 0.
  const L0 = z.levels[0], off = virtual ? [0, 0, 0] : (ids.length > 1 ? L0.origin : L0.tr).map((o, a) => o / L0.sp[a]);
  return Object.assign(Object.create(null), { z, off });
}

// Marching squares on a binary mask -> line segments (in mask pixel coordinates). Neuroglancer annotations have no region type,
// so class boundaries (tissue vs background, gray vs white) are drawn as contour lines.
function contourSegments(mask, w, h) {
  const segs = [], at = (x, y) => (x < 0 || y < 0 || x >= w || y >= h ? 0 : mask[y * w + x]);
  for (let y = -1; y < h; y++) for (let x = -1; x < w; x++) {
    const c = at(x, y) | (at(x + 1, y) << 1) | (at(x + 1, y + 1) << 2) | (at(x, y + 1) << 3); if (c === 0 || c === 15) continue;
    const T = [x + 1, y + 0.5], R = [x + 1.5, y + 1], B = [x + 1, y + 1.5], L = [x + 0.5, y + 1];   // midpoints of the cell edges (cell centre at x+1,y+1)
    const tbl = { 1: [[L, T]], 2: [[T, R]], 3: [[L, R]], 4: [[R, B]], 5: [[L, T], [R, B]], 6: [[T, B]], 7: [[L, B]], 8: [[L, B]], 9: [[T, B]], 10: [[T, R], [L, B]], 11: [[R, B]], 12: [[L, R]], 13: [[T, R]], 14: [[L, T]] };
    for (const [a, b] of tbl[c]) segs.push([a, b]);
  }
  return segs;
}
// The run covers the visible field of view plus a 15% margin on every side: the viewport can change once the layer panel opens or the
// window is resized, and the probe reads the view at the moment of the click.
const padRange = ([a, b]) => { const m = 0.15 * (b - a); return [a - m, b + m]; };
function setSegAlpha(a) {                                                      // Neuroglancer's own selected-segment opacity, on every segmentation layer
  segAlpha = Math.max(0, Math.min(1, +a)); let n = 0;
  for (const ml of viewer.layerManager.managedLayers) { const ds = ml.layer && ml.layer.displayState; if (ds && ds.selectedAlpha) { ds.selectedAlpha.value = segAlpha; n++; } }
  return n;
}
// Neuroglancer reads every layer's hover (selection) state whenever the mouse is active over the view; a layer that was just created has none yet
// ('Cannot read properties of undefined (reading 'generation')'). Deactivate the mouse state before changing layers (it reactivates on the next move).
function quietMouse() { try { viewer.mouseState.setActive(false); } catch (e) { /* older builds */ } }
// Layer-list update without touching view, layout or display dimensions: rebuild only the layer list through the layer specification
// (viewer.state.restoreState also re-restores layout/position and was the source of the broken-state errors).
function setLayerList(mutate) {
  quietMouse();
  const cur = JSON.parse(JSON.stringify(viewer.state.toJSON().layers || []));
  viewer.layerSpecification.restoreState(mutate(cur) || cur);
}
function applyHwLayers(specs, rec) {
  try {
    setLayerList((cur) => { const out = cur.filter((l) => !/^hwseg /.test(l.name));
      if (rec) { const im = out.find((l) => l.type === 'image'); if (im && /uicontrol float lo/.test(String(im.shader || ''))) im.shaderControls = { ...(im.shaderControls || {}), lo: +rec.lo.toFixed(2), hi: +rec.hi.toFixed(2), gamma: +rec.gamma.toFixed(2) }; }
      return [...out, ...specs]; });
    return true;
  } catch (e) { console.warn('layer-list update failed, falling back to restoreState:', e); return false; }
}
// ---- Python bridge: Pyodide runs inside this page; `ng` (demo/py/ng_prelude.py) reads the current view and writes back layers
const PYO = 'https://cdn.jsdelivr.net/pyodide/v314.0.8/full/';
let pyP = null, pyCtx = null, pyCode = '', pyQ = Promise.resolve(), lastPyView = null;
window.__hwNg.pendingPy = pyPending;
function getPy() {
  return pyP ||= (async () => {
    say('python: loading Pyodide + numpy (first use, a few seconds)...');
    await new Promise((res, rej) => { const sc = document.createElement('script'); sc.src = PYO + 'pyodide.js'; sc.onload = res; sc.onerror = () => rej(new Error('could not load Pyodide from the CDN')); document.head.append(sc); });
    const py = await window.loadPyodide({ indexURL: PYO }); await py.loadPackage('numpy');
    py.setStdout({ batched: (t) => { window.__hwNg.pyOut = (window.__hwNg.pyOut || '') + t + '\n'; say('python: ' + t); } });
    py.setStderr({ batched: (t) => { window.__hwNg.pyOut = (window.__hwNg.pyOut || '') + t + '\n'; } });
    py.runPython(await (await fetch(HERE('py/ng_prelude.py'))).text());
    say('python ready'); return py;
  })();
}
async function pyRead(level, depth, margin) {
  let view;
  if (pyCtx && pyCtx.forced) view = pyCtx.forced;                                // regeneration: the saved view, not the live one
  else {
    const j = viewer.state.toJSON(), fov = await probeFov(), d = j.dimensions || {}, A = axisIdx(d), srcSpec = imageSource(j), src = await openSource(srcSpec), zarr = src.z, off = src.off, L0 = zarr.levels[0];
    const gx = padRange([fov.tl[A.x], fov.br[A.x]].sort((a, b) => a - b).map((v) => v)), gy = padRange([fov.tl[A.y], fov.br[A.y]].sort((a, b) => a - b)), umPerPx = ((fov.br[A.x] - fov.tl[A.x]) * L0.sp[2]) / fov.w;
    let lv = level == null ? 0 : Math.max(0, Math.min(zarr.top, Math.round(level)));
    if (level == null) for (let l = 0; l <= zarr.top; l++) if (zarr.levels[l].sp[2] <= Math.abs(umPerPx) * 1.0) lv = l;
    const mk = (l) => { const f = 2 ** l, L = zarr.levels[l];
      return { f, L, box: [clamp(Math.floor((gy[0] - off[1]) / f), 0, L.shape[1] - 1), clamp(Math.ceil((gy[1] - off[1]) / f), 1, L.shape[1]), clamp(Math.floor((gx[0] - off[2]) / f), 0, L.shape[2] - 1), clamp(Math.ceil((gx[1] - off[2]) / f), 1, L.shape[2])] }; };
    let m = mk(lv); const nzw = Math.max(1, Math.round(depth || 1));
    while (nzw * (m.box[1] - m.box[0]) * (m.box[3] - m.box[2]) > 8e6 && lv < zarr.top) m = mk(++lv);       // keep the crop within memory
    const zl = clamp(Math.round((fov.c[A.z] - off[0]) / m.f), 0, m.L.shape[0] - 1), zA = clamp(zl - (nzw >> 1), 0, Math.max(0, m.L.shape[0] - nzw)), zB = Math.min(m.L.shape[0], zA + nzw);
    view = { level: lv, box: m.box, zA, zB, zl, src: srcSpec, umPerPx: Math.abs(umPerPx) };
  }
  const src = await openSource(view.src), L = src.z.levels[view.level], [y0, y1, x0, x1] = view.box;
  say(`python: reading ${view.zB - view.zA} x ${y1 - y0} x ${x1 - x0} voxels at level ${view.level}...`);
  const data = await src.z.crop(view.level, view.zA, view.zB, y0, y1, x0, x1);
  if (!pyCtx || !pyCtx.forced) lastPyView = view;
  return { data, shape: [view.zB - view.zA, y1 - y0, x1 - x0], level: view.level, spacing: [...L.sp], origin: [view.zA, y0, x0], plane: view.zl - view.zA, umPerPx: view.umPerPx ?? 0 };
}
function pyBlock(flat, shape, view) {
  const [nz, h, w] = shape; return { x0: view.box[2], x1: view.box[2] + w, y0: view.box[0], y1: view.box[0] + h, z0: view.zA, z1: view.zA + nz, lab: new Uint32Array(flat) };   // copy: `flat` can be a view into Pyodide's memory, which is reused after the run
}
function addLayer(spec) { setLayerList((cur) => [...cur.filter((l) => l.name !== spec.name), spec]); }
window.__hwPy = {
  read: (level, depth, margin) => pyRead(level, depth == null ? 9 : depth, margin),
  async segmentation(flat, shape, name) {
    const view = pyCtx && pyCtx.forced ? pyCtx.forced : lastPyView; if (!view) throw new Error('call ng.read() first');
    const block = pyBlock(flat, shape.toJs ? shape.toJs() : shape, view);
    if (pyCtx && pyCtx.capture) { pyCtx.capture.push(block); return; }
    { let nz = 0, mx = 0; for (const q of block.lab) { if (q) nz++; if (q > mx) mx = q; } say(`python: segmentation ${nz} labeled voxels, max id ${mx}, view ${JSON.stringify(view.box)} z ${view.zA}-${view.zB}`); }
    const P = { kind: 'py', name, code: pyCode, view: { level: view.level, box: view.box, zA: view.zA, zB: view.zB, zl: view.zl, src: view.src, umPerPx: view.umPerPx } }, id = b64e(P);
    pyApproved.add(id);                                                          // we ran this code ourselves
    addLayer(segLayer('py: ' + name, putSeg(P, [block])));
    say(`python: segmentation layer "py: ${name}" (${shape[0] ?? ''} planes)`);
  },
  async annotations(pts, rad, name) {
    if (pyCtx && pyCtx.forced) return;                                           // annotations are not part of a regenerated volume
    const view = lastPyView; if (!view) throw new Error('call ng.read() first');
    const j = viewer.state.toJSON(), d = j.dimensions || {}, keys = Object.keys(d), pos = j.position || [], src = await openSource(view.src), L = src.z.levels[view.level], L0 = src.z.levels[0], f = 2 ** view.level;
    const P = Array.from(pts), R = Array.from(rad), ann = [];
    for (let i = 0; i < R.length; i++) {
      const g = { z: (view.zA + P[3 * i] + 0.5) * f + src.off[0], y: (view.box[0] + P[3 * i + 1] + 0.5) * f + src.off[1], x: (view.box[2] + P[3 * i + 2] + 0.5) * f + src.off[2] };
      const r = { z: Math.max(1, R[i] / L0.sp[0]), y: Math.max(1, R[i] / L0.sp[1]), x: Math.max(1, R[i] / L0.sp[2]) };
      ann.push({ type: 'ellipsoid', id: 'p' + i, center: keys.map((kk, q) => (g[kk] !== undefined ? g[kk] : pos[q] || 0)), radii: keys.map((kk) => r[kk] || 1) });
    }
    addLayer({ type: 'annotation', name: 'py: ' + name, annotationColor: '#3cffb4', source: { url: 'local://annotations', transform: { outputDimensions: Object.fromEntries(keys.map((kk) => [kk, d[kk]])) } }, annotations: ann });
    say(`python: ${ann.length} annotations "py: ${name}"`);
  },
};
async function runPyCode(code, ctx) {                                            // one Python job at a time
  const job = pyQ.then(async () => {
    const py = await getPy(); await py.loadPackagesFromImports(code);
    const prev = pyCtx; pyCtx = ctx || null; if (!ctx) { pyCode = code; window.__hwNg.pyOut = ''; }
    try { const g = py.toPy({}); g.set('ng', py.pyimport('ng')); await py.runPythonAsync(code, { globals: g }); }
    finally { pyCtx = prev; }
  });
  pyQ = job.catch(() => {}); return job;
}
async function computePy(P) {                                                    // regenerate a Python segmentation from its saved code + view
  const capture = []; await runPyCode(P.code, { forced: P.view, capture });
  if (!capture.length) throw new Error('the Python code produced no segmentation');
  { let nz = 0, mx = 0; for (const q of capture[0].lab) { if (q) nz++; if (q > mx) mx = q; } say(`python: regenerated segmentation, ${nz} labeled voxels, max id ${mx}, view ${JSON.stringify(P.view.box)} z ${P.view.zA}-${P.view.zB}`); }
  return { blocks: capture };
}
window.__hwNg.runPython = async (code) => { window.__hwNg.busy = true; try { await runPyCode(code, null); return { ok: true, out: window.__hwNg.pyOut || '' }; } catch (e) { return { ok: false, out: (window.__hwNg.pyOut || '') + String(e.message || e) }; } finally { window.__hwNg.busy = false; } };
window.__hwNg.approvePy = async (id) => {                                        // the person reviewed the code in a link and allowed it
  pyApproved.add(id); const i = pyPending.findIndex((p) => p.id === id); if (i >= 0) pyPending.splice(i, 1);
  return window.__hwNg.regenSeg(id);
};

// ---- compute a segmentation from its parameters (also used to regenerate one from a shared link)
async function computeSeg(P) {
  if (P.kind === 'py') return computePy(P);
  for (let i = 0; i < 1800 && !window.__hwNg.ready; i++) await sleep(100);
  if (isRef(P.kind)) return computeRef(P, (await openSource(P.src)).z);         // Neurodesk reference models: onnxruntime-web, no WebGPU needed
  if (!engine) throw new Error('no WebGPU: segmentation unavailable');
  const src = await openSource(P.src), zarr = src.z, L = zarr.levels[P.level], adjust = P.rec ? (v) => adjustLogVol(v, P.rec.lo, P.rec.hi, { b: 0, c: 1, g: P.rec.gamma }) : null;   // hier programs: raw log(I+1), no exposure
  if (P.kind === 'hregions') {                                                   // hierarchical regions: one pass over the FOV, labels 1-6 = depth bands, 7 = WM, 8 = tissue (no cortex decision)
    const prog = await getProgram('hier_regions'), [y0, y1, x0, x1] = P.box;
    say('regions (hierarchical) on WebGPU...'); const r = await segRegion(zarr, engine, prog, P.level, P.zl, y0, y1, x0, x1, null, { halo: false });
    const w = r.w, h = r.h, lab = new Uint32Array(w * h), names = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'WM'];
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { const i = (r.oy + y) * r.dims[2] + r.ox + x; let k = 0;
      for (let n = 0; n < names.length; n++) if (r.out[names[n]][i] > 0.5) { k = n + 1; break; }
      if (!k && r.out.tissue[i] > 0.5) k = 8; lab[y * w + x] = k; }
    return { blocks: [{ x0, x1, y0, y1, z0: P.zl, z1: P.zl + 1, lab }], w, h };
  }
  if (P.kind === 'regions') {
    const progT = await getProgram('fable_tissue'), progC = await getProgram('fable_cortex_wm'), [y0, y1, x0, x1] = P.box;
    say('regions: tissue program on WebGPU...'); const rT = await segRegion(zarr, engine, progT, P.level, P.zl, y0, y1, x0, x1, adjust, { halo: false });
    say('regions: cortex / white-matter program on WebGPU...'); const rC = await segRegion(zarr, engine, progC, P.level, P.zl, y0, y1, x0, x1, adjust, { halo: false });
    const w = rT.w, h = rT.h, tis = new Uint8Array(w * h), wm = new Uint8Array(w * h), lab = new Uint32Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { const q = y * w + x; tis[q] = rT.out.tissue[(rT.oy + y) * rT.dims[2] + rT.ox + x] > 0.5 ? 1 : 0; wm[q] = rC.out.wm[(rC.oy + y) * rC.dims[2] + rC.ox + x] > 0.5 ? 1 : 0; lab[q] = tis[q] ? (wm[q] ? 2 : 1) : 0; }   // 1 = gray matter, 2 = white matter
    return { blocks: [{ x0, x1, y0, y1, z0: P.zl, z1: P.zl + 1, lab }], tis, wm, w, h };
  }
  const isVessel = P.kind === 'vessels', prog = await getProgram(P.model), box_ = P.box, { zA, zB, zA2, zB2 } = P, nz = zB - zA, hal = 32;
  const [hz] = haloOf(prog, L.sp), Zin = Math.min(L.shape[0], (zB2 - zA2) + 2 * hz);
  const side = Math.max(96 + 2 * hal, Math.min(512, Math.floor(Math.sqrt(engine.maxTokens(prog.hier ? prog.d[0] + 2 : prog.d) * 0.8 / Zin)))), core = side - 2 * hal;   // tile side that keeps (slab + z halo) tokens within the device buffer
  const cy = (box_[0] + box_[1]) >> 1, cx = (box_[2] + box_[3]) >> 1;
  const tiles = []; for (let ty = box_[0]; ty < box_[1]; ty += core) for (let tx = box_[2]; tx < box_[3]; tx += core) tiles.push([ty, Math.min(ty + core, box_[1]), tx, Math.min(tx + core, box_[3])]);
  tiles.sort((p, q) => Math.hypot((p[0] + p[1]) / 2 - cy, (p[2] + p[3]) / 2 - cx) - Math.hypot((q[0] + q[1]) / 2 - cy, (q[2] + q[3]) / 2 - cx));   // centre first
  const MAXT = 100, use = tiles.slice(0, MAXT), comps = [], blocks = []; let idBase = 0;
  for (let ti = 0; ti < use.length; ti++) {
    const [cy_0, cy_1, cx_0, cx_1] = use[ti], Y0 = Math.max(0, cy_0 - hal), Y1 = Math.min(L.shape[1], cy_1 + hal), X0 = Math.max(0, cx_0 - hal), X1 = Math.min(L.shape[2], cx_1 + hal);
    say(`${P.model}: tile ${ti + 1}/${use.length} (${X1 - X0}x${Y1 - Y0}x${nz} planes, level ${P.level}) on WebGPU...`);
    const r = await segVolume(zarr, engine, prog, P.level, zA2, zB2, Y0, Y1, X0, X1, adjust, { halo: false });
    const lx0 = cx_0 - X0, lx1 = cx_1 - X0, ly0 = cy_0 - Y0, ly1 = cy_1 - Y0, cw = lx1 - lx0, chh = ly1 - ly0, plane = r.h * r.w; let full;
    if (isVessel) full = labelVolume(FAMILIES[P.family].kind, r);                 // class labels from the program's own channels (1 = capillary, 2 = vessel, ...)
    else { const t3 = components3(r.out.seg, r.dims, r.ox, r.oy, r.w, r.h, r.nz, idBase); idBase += t3.n; full = t3.lab;   // one id per linked cell
      for (const c of t3.objs) { const gy_ = Y0 + c.y, gx_ = X0 + c.x, gz_ = zA2 + c.z; if (gy_ >= cy_0 && gy_ < cy_1 && gx_ >= cx_0 && gx_ < cx_1 && gz_ >= zA - 0.5 && gz_ < zB - 0.5) comps.push({ ...c, y: gy_, x: gx_, z: gz_ }); } }
    const lab = new Uint32Array(cw * chh * r.nz);                                  // keep only this tile's core: tiles abut without overlap
    for (let z = 0; z < r.nz; z++) for (let y = 0; y < chh; y++) { const sx = z * plane + (ly0 + y) * r.w + lx0, dst = (z * chh + y) * cw; for (let x = 0; x < cw; x++) lab[dst + x] = full[sx + x]; }
    blocks.push({ x0: cx_0, x1: cx_1, y0: cy_0, y1: cy_1, z0: zA2, z1: zA2 + r.nz, lab });
  }
  return { blocks, comps, nTiles: tiles.length, nUsed: use.length };
}
// ---- Neurodesk reference models on tiles of the field of view (2-D models per plane, linked across z like our programs).
// Tiles abut (each keeps its core); a detection belongs to the tile whose core holds its centre, as for the hand-written programs.
const REF_MAXT = 36;
async function computeRef(P, zarr) {
  const M = REF[P.kind], L = zarr.levels[P.level], box_ = P.box, ves = M.kind === 'vessels', hal = ves ? 0 : 16, core = ves ? 128 : M.dims === 2 ? 160 : 192;
  const cy = (box_[0] + box_[1]) >> 1, cx = (box_[2] + box_[3]) >> 1, tiles = [];
  for (let ty = box_[0]; ty < box_[1]; ty += core) for (let tx = box_[2]; tx < box_[3]; tx += core) tiles.push([ty, Math.min(ty + core, box_[1]), tx, Math.min(tx + core, box_[3])]);
  tiles.sort((p, q) => Math.hypot((p[0] + p[1]) / 2 - cy, (p[2] + p[3]) / 2 - cx) - Math.hypot((q[0] + q[1]) / 2 - cy, (q[2] + q[3]) / 2 - cx));
  const use = tiles.slice(0, REF_MAXT), blocks = [], comps = [], Z = P.zB2 - P.zA2, t0 = performance.now(); let idBase = 0;
  for (let ti = 0; ti < use.length; ti++) {
    const [cy_0, cy_1, cx_0, cx_1] = use[ti], Y0 = Math.max(0, cy_0 - hal), Y1 = Math.min(L.shape[1], cy_1 + hal), X0 = Math.max(0, cx_0 - hal), X1 = Math.min(L.shape[2], cx_1 + hal), H = Y1 - Y0, W = X1 - X0;
    say(`${P.kind}: tile ${ti + 1}/${use.length} (${W}x${H}x${Z} planes, level ${P.level})...`);
    const vol = await zarr.crop(P.level, P.zA2, P.zB2, Y0, Y1, X0, X1);
    const r = await refTile(P.kind, vol, [Z, H, W], L.sp, { diamUm: P.diamUm, norm: P.norm, idBase, say: (t) => say(`${t} (tile ${ti + 1}/${use.length})`), onPlane: (k, n) => say(`${P.kind}: tile ${ti + 1}/${use.length}, plane ${k}/${n}`) });
    const full = ves ? r.cls : r.lab; if (!ves) { idBase += r.n;
      for (const c of r.objs) { const gy_ = Y0 + c.y, gx_ = X0 + c.x, gz_ = P.zA2 + c.z; if (gy_ >= cy_0 && gy_ < cy_1 && gx_ >= cx_0 && gx_ < cx_1 && gz_ >= P.zA - 0.5 && gz_ < P.zB - 0.5) comps.push({ ...c, y: gy_, x: gx_, z: gz_ }); } }
    const lx0 = cx_0 - X0, ly0 = cy_0 - Y0, cw = cx_1 - cx_0, chh = cy_1 - cy_0, lab = new Uint32Array(cw * chh * Z);
    for (let z = 0; z < Z; z++) for (let y = 0; y < chh; y++) { const sx = (z * H + ly0 + y) * W + lx0, dst = (z * chh + y) * cw; for (let x = 0; x < cw; x++) lab[dst + x] = full[sx + x]; }
    blocks.push({ x0: cx_0, x1: cx_1, y0: cy_0, y1: cy_1, z0: P.zA2, z1: P.zB2, lab });
  }
  return { blocks, comps, nTiles: tiles.length, nUsed: use.length, seconds: (performance.now() - t0) / 1000 };
}
async function detectRegions(j, fov, zarr, off, d, umPerPx, sp0) {
  const keys = Object.keys(d), pos = j.position || [];
  const m0 = scaledModel('fable_cortex_wm', umPerPx, zarr), progT = await getProgram('fable_tissue'), progC = await getProgram('fable_cortex_wm');
  const A = axisIdx(d), gx = padRange([fov.tl[A.x], fov.br[A.x]].sort((a, b) => a - b)), gy = padRange([fov.tl[A.y], fov.br[A.y]].sort((a, b) => a - b)), lz = fov.c[A.z] - off[0];
  const dmax = Math.max(progT.d, progC.d); let level = m0.level, box_;
  for (;;) {                                                                       // coarsen until the whole FOV fits one pass
    const f = 2 ** level, L = zarr.levels[level], hz = Math.max(haloFor(progT, L.sp)[0], haloFor(progC, L.sp)[0]);
    box_ = [clamp(Math.floor((gy[0] - off[1]) / f), 0, L.shape[1] - 1), clamp(Math.ceil((gy[1] - off[1]) / f), 1, L.shape[1]), clamp(Math.floor((gx[0] - off[2]) / f), 0, L.shape[2] - 1), clamp(Math.ceil((gx[1] - off[2]) / f), 1, L.shape[2])];
    if ((2 * hz + 1) * (box_[1] - box_[0]) * (box_[3] - box_[2]) <= Math.floor(engine.maxTokens(dmax) * 0.6) || level >= zarr.top) break; level++;
  }
  const f = 2 ** level, L = zarr.levels[level], zl = clamp(Math.floor(lz / f), 0, L.shape[0] - 1);   // Neuroglancer voxel k spans [k, k+1): the displayed plane is floor, not round
  say(`regions on ${box_[3] - box_[2]}x${box_[1] - box_[0]} voxels at level ${level}: streaming...`);
  const rec = await recommendExposure(zarr, engine, await getProgram('exposure'), level, zl, box_[0], box_[1], box_[2], box_[3]);
  const adjust = (v) => adjustLogVol(v, rec.lo, rec.hi, { b: 0, c: 1, g: rec.gamma });
  const P = { kind: 'regions', level, zl, box: box_, rec: { lo: +rec.lo.toFixed(4), hi: +rec.hi.toFixed(4), gamma: +rec.gamma.toFixed(4) }, src: imageSource(j) };
  const R = await computeSeg(P), { tis, wm, w, h } = R;
  const toG = (p) => { const g = { z: zl * f + off[0], y: (box_[0] + p[1]) * f + off[1], x: (box_[2] + p[0]) * f + off[2] }; return keys.map((kk, q) => (g[kk] !== undefined ? g[kk] : pos[q] || 0)); };
  const mk = (segs, tag) => segs.map((s2, i) => ({ type: 'line', id: tag + i, pointA: toG(s2[0]), pointB: toG(s2[1]) }));
  const layers = [['hwseg tissue outline', '#ffd23c', mk(contourSegments(tis, w, h), 't')], ['hwseg cortex/white-matter boundary', '#3cd2ff', mk(contourSegments(wm, w, h), 'w')]];
  const LINE_SHADER = 'void main() { setColor(defaultColor()); setLineWidth(1.5); setEndpointMarkerSize(0.0); setEndpointMarkerBorderWidth(0.0); }';   // thin lines, no endpoint markers
  const specs = [segLayer('hwseg tissue / white-matter segmentation', putSeg(P, R.blocks), { segmentColors: { 1: '#ffa028', 2: '#3c8cff' } })];
  for (const [name, col, ann] of layers) specs.push({ type: 'annotation', name, annotationColor: col, shader: LINE_SHADER, source: { url: 'local://annotations', transform: { outputDimensions: Object.fromEntries(keys.map((kk) => [kk, d[kk]])) } }, annotations: ann });
  lastView = viewKey();
  if (!applyHwLayers(specs, rec)) {
    const k = viewer.state.toJSON(); k.layers = (k.layers || []).filter((l) => !/^hwseg /.test(l.name)).concat(specs);
    const img = k.layers.find((l) => l.type === 'image'); if (img && /uicontrol float lo/.test(String(img.shader || ''))) img.shaderControls = { ...(img.shaderControls || {}), lo: +rec.lo.toFixed(2), hi: +rec.hi.toFixed(2), gamma: +rec.gamma.toFixed(2) };
    k.displayDimensions = ['x', 'y', 'z']; k.layout = k.layout || 'xy'; quietMouse(); viewer.state.reset(); await sleep(60); viewer.state.restoreState(k);
  }
  const tp = 100 * tis.reduce((a, b) => a + b, 0) / (w * h), wp = 100 * wm.reduce((a, b) => a + b, 0) / (w * h);
  say(`${layers[0][2].length} outline + ${layers[1][2].length} cortex/WM segments (tissue ${tp.toFixed(0)}%, white matter ${wp.toFixed(0)}%, level ${level}, ${umPerPx.toFixed(0)} um/px, FOV ${w}x${h} voxels)`);
  window.__hwNg.last = { regions: true, outline: layers[0][2].length, boundary: layers[1][2].length, tissuePct: tp, wmPct: wp, level, fov: [w, h] };
}

// hierarchical regions: whole FOV in one pass at the coarsest level that fits the device buffer (no exposure, no level routing of the program)
async function detectRegionsHier(j, fov, zarr, off, d, umPerPx) {
  const prog = await getProgram('hier_regions'), A = axisIdx(d), m0 = scaledModel('hier_regions', umPerPx, zarr);
  const gx = padRange([fov.tl[A.x], fov.br[A.x]].sort((a, b) => a - b)), gy = padRange([fov.tl[A.y], fov.br[A.y]].sort((a, b) => a - b)), lz = fov.c[A.z] - off[0];
  let level = m0.level, box_;
  for (;;) {
    const f = 2 ** level, L = zarr.levels[level], hz = haloOf(prog, L.sp)[0];
    box_ = [clamp(Math.floor((gy[0] - off[1]) / f), 0, L.shape[1] - 1), clamp(Math.ceil((gy[1] - off[1]) / f), 1, L.shape[1]), clamp(Math.floor((gx[0] - off[2]) / f), 0, L.shape[2] - 1), clamp(Math.ceil((gx[1] - off[2]) / f), 1, L.shape[2])];
    if ((2 * hz + 1) * (box_[1] - box_[0]) * (box_[3] - box_[2]) <= Math.floor(engine.maxTokens(Math.max(...prog.d)) * 0.5) || level >= zarr.top) break; level++;
  }
  const f = 2 ** level, L = zarr.levels[level], zl = clamp(Math.round(lz / f), 0, L.shape[0] - 1);
  const P = { kind: 'hregions', level, zl, box: box_, src: imageSource(j) };
  const R = await computeSeg(P);
  const COL = { 1: '#fde725', 2: '#aadc32', 3: '#5ec962', 4: '#21918c', 5: '#3b528b', 6: '#6e2896', 7: '#d7d7eb', 8: '#5ac878' };
  const specs = [segLayer('hwseg regions (hierarchical): depth bands / white matter', putSeg(P, R.blocks), { segmentColors: COL })];
  lastView = viewKey();
  if (!applyHwLayers(specs, null)) { const k = viewer.state.toJSON(); k.layers = (k.layers || []).filter((l) => !/^hwseg /.test(l.name)).concat(specs); viewer.state.reset(); await sleep(60); viewer.state.restoreState(k); }
  const lab = R.blocks[0].lab, n = lab.length, cnt = new Array(9).fill(0); for (let i = 0; i < n; i++) cnt[lab[i]]++;
  say(`hierarchical regions at level ${level} (${umPerPx.toFixed(0)} um/px): tissue ${(100 * (n - cnt[0]) / n).toFixed(0)}%, white matter ${(100 * cnt[7] / n).toFixed(0)}%, bands I-VI ${(100 * cnt.slice(1, 7).reduce((a, b) => a + b, 0) / n).toFixed(0)}%`);
  window.__hwNg.last = { hregions: true, level, fov: [box_[3] - box_[2], box_[1] - box_[0]], counts: cnt };
}

async function detect() {
  const key = $('hw-m').value, ref = isRef(key); if (!engine && !ref) { say('no WebGPU: only the Neurodesk reference models (onnxruntime-web) can run here'); return; } window.__hwNg.runs++;
  const j = viewer.state.toJSON(), disp = j.displayDimensions, fov = await probeFov();
  const d = j.dimensions || {}, sp0 = { z: d.z && d.z[0], y: d.y && d.y[0], x: d.x && d.x[0] }; if (!sp0.x) throw new Error('state has no x/y/z dimensions');
  const src = await openSource(imageSource(j)), zarr = src.z, off = src.off, L0 = zarr.levels[0], A = axisIdx(d);
  const gx0_ = [fov.tl[A.x], fov.br[A.x]].sort((a, b) => a - b), gy0_ = [fov.tl[A.y], fov.br[A.y]].sort((a, b) => a - b), gx = padRange(gx0_), gy = padRange(gy0_), gz = fov.c[A.z];
  if (!(gx[1] - gx[0] > 1 && gy[1] - gy[0] > 1)) throw new Error(`show the x-y plane (displayDimensions x, y, z and layout xy) [dims ${Object.keys(d)} x ${gx} y ${gy} tl ${fov.tl} br ${fov.br}]`);
  // level-0 local voxel box of the FOV
  const lx = [gx[0] - off[2], gx[1] - off[2]], ly = [gy[0] - off[1], gy[1] - off[1]], lz = gz - off[0];
  const umPerPx = ((gx[1] - gx[0]) * L0.sp[2]) / fov.w;                       // physical size of a screen pixel
  if ($('hw-m').value === 'regions') return detectRegions(j, fov, zarr, off, d, umPerPx, sp0);
  if (!ref && $('hw-m').value === 'hier_regions') return detectRegionsHier(j, fov, zarr, off, d, umPerPx);
  const hierSel = !ref && !!(FAMILIES[key] && FAMILIES[key].hier);
  if (umPerPx > (ref ? 8 : 16) && !hierSel) { say(`zoom in first: ${ref ? 'the reference models run at full resolution and need a field of view of about 1-3 mm' : 'cells and vessels need a field of view of a few mm'} (this view is ${umPerPx.toFixed(0)} um/px)`); return; }
  // reference models: the coarsest pyramid level at which an object of the declared diameter still spans >= 4 voxels in x
  // (in practice level 0 for 10-16 um nuclei); 2-D models then resample each plane so the object spans the network's training size
  const family = key, diamUm = ref ? +window.__hwNg.diamUm || (REF[key].kind === 'vessels' ? 8 : 10) : null;
  let refLevel = 0; if (ref) for (let l = 0; l <= Math.min(2, zarr.top); l++) if (diamUm / zarr.levels[l].sp[2] >= 4) refLevel = l;
  const m = ref ? { model: key, level: refLevel } : scaledModel(family, umPerPx, zarr), f = 2 ** m.level, L = zarr.levels[m.level];
  const box_ = [clamp(Math.floor(ly[0] / f), 0, L.shape[1] - 1), clamp(Math.ceil(ly[1] / f), 1, L.shape[1]), clamp(Math.floor(lx[0] / f), 0, L.shape[2] - 1), clamp(Math.ceil(lx[1] / f), 1, L.shape[2])];
  const nzWant = Math.max(1, +$('hw-nz').value || 1), zl = clamp(Math.floor(lz / f), 0, L.shape[0] - 1);   // Neuroglancer voxel k spans [k, k+1): the displayed plane is floor, not round
  const zA = clamp(zl - (nzWant >> 1), 0, Math.max(0, L.shape[0] - nzWant)), zB = Math.min(L.shape[0], zA + nzWant), nz = zB - zA;   // slab [zA,zB) centred on the displayed plane
  const mz = nz > 1 || (ref && REF[key].dims === 3) ? Math.min(4, Math.ceil(10 / L.sp[0])) : 0; let zA2 = Math.max(0, zA - mz), zB2 = Math.min(L.shape[0], zB + mz);
  if (ref && REF[key].zCtx) { zA2 = Math.max(0, zA - REF[key].zCtx); zB2 = Math.min(L.shape[0], zB + REF[key].zCtx); }
  if (ref && REF[key].zPatch) { const need = REF[key].zPatch; zA2 = clamp(zl - (need >> 1), 0, Math.max(0, L.shape[0] - need)); zB2 = Math.min(L.shape[0], zA2 + need); }   // VesselBoost sees a 64-plane slab (its patch depth)   // margin planes so objects cut by the slab edge keep their full z extent; a detection belongs to the slab if its centre is inside [zA,zB)
  const cy = (box_[0] + box_[1]) >> 1, cx = (box_[2] + box_[3]) >> 1;
  const cy0 = Math.max(0, cy - 192), cy1 = Math.min(L.shape[1], cy + 192), cx0 = Math.max(0, cx - 192), cx1 = Math.min(L.shape[2], cx + 192);
  const hier = !ref && !!FAMILIES[family].hier;                               // hierarchical programs normalise contrast in-model: no exposure program
  if (!hier) say(`${m.model}: streaming exposure window...`);
  const rec = engine && !hier ? await recommendExposure(zarr, engine, await getProgram('exposure'), m.level, zl, box_[0], box_[1], box_[2], box_[3]) : null, isVessel = ref ? REF[key].kind === 'vessels' : /^vessels/.test(FAMILIES[family].kind);
  const rec4 = rec ? { lo: +rec.lo.toFixed(4), hi: +rec.hi.toFixed(4), gamma: +rec.gamma.toFixed(4) } : null;
  let P;
  if (ref) {   // the model's own intensity normalization (percentiles) over the displayed plane of the whole field of view, stored in the link
    say(`${key}: normalization window over the field of view...`);
    const norm = fovNorm(key, await zarr.crop(m.level, zl, zl + 1, box_[0], box_[1], box_[2], box_[3]));
    P = { kind: key, model: key, level: m.level, box: box_, zA, zB, zA2, zB2, diamUm, norm, rec: rec4, src: imageSource(j) };
  } else P = { kind: isVessel ? 'vessels' : 'cells', family, model: m.model, level: m.level, box: box_, zA, zB, zA2, zB2, ...(rec4 ? { rec: rec4 } : {}), src: imageSource(j) };
  const { blocks, comps, nTiles, nUsed, seconds } = await computeSeg(P), use = { length: nUsed }, tiles = { length: nTiles };
  if (tiles.length > use.length) say(`field of view needs ${tiles.length} tiles; the ${use.length} nearest the centre were run - zoom in for full coverage`);
  const sL = L.sp;
  // voxel (level m.level, local) -> Neuroglancer global voxel (level 0 + translation); the radii are the measured 3-D extents
  // Annotation space = the viewer's own dimensions in the same order (t, z, y, x): any other space reorders the global
  // dimensions and resets the view. Dimensions we do not segment (t) take the current position and radius 1.
  const keys = Object.keys(d), pos = j.position || [], dimJSON = Object.fromEntries(keys.map((kk) => [kk, d[kk]]));
  const ann = comps.map((c, i) => { const g = { z: (c.z + 0.5) * f + off[0], y: (c.y + 0.5) * f + off[1], x: (c.x + 0.5) * f + off[2] };
    const rad = { z: Math.max(1, c.rz * sL[0] / L0.sp[0]), y: Math.max(1, c.ry * sL[1] / L0.sp[1]), x: Math.max(1, c.rx * sL[2] / L0.sp[2]) };
    return { type: 'ellipsoid', id: 'd' + i, center: keys.map((kk, q) => (g[kk] !== undefined ? g[kk] : pos[q] || 0)), radii: keys.map((kk) => rad[kk] || 1) }; });
  // the segmentation (voxel labels) is the result; the object centres / extents are an optional, hidden annotation layer
  const specs = [segLayer((isVessel ? 'hwseg vessel segmentation' : 'hwseg cell segmentation') + (ref ? ` (${key.replace('nd_', '')})` : ''), putSeg(P, blocks))];
  if (!isVessel) specs.push({ type: 'annotation', name: LAYER, visible: false, annotationColor: '#ff3cc8', source: { url: 'local://annotations', transform: { outputDimensions: dimJSON } }, annotations: ann });
  lastView = viewKey();
  if (!applyHwLayers(specs, rec)) {
    const k = viewer.state.toJSON(); k.layers = (k.layers || []).filter((l) => !/^hwseg /.test(l.name)).concat(specs);
    const img = k.layers.find((l) => l.type === 'image'); if (rec && img && /uicontrol float lo/.test(String(img.shader || ''))) img.shaderControls = { ...(img.shaderControls || {}), lo: +rec.lo.toFixed(2), hi: +rec.hi.toFixed(2), gamma: +rec.gamma.toFixed(2) };
    k.displayDimensions = ['x', 'y', 'z']; k.layout = k.layout || 'xy'; quietMouse(); viewer.state.reset(); await sleep(60); viewer.state.restoreState(k);
  }
  say(`segmentation: ${isVessel ? 'vessel mask' : ann.length + ' cells'} over ${ref ? zB2 - zA2 : nz} planes (${m.model}, level ${m.level}, ${(umPerPx).toFixed(1)} um/px, ${ref ? `object diameter ${diamUm} um, ${seconds.toFixed(0)} s, ${JSON.stringify(ndInfo().ep)}` : (rec ? `exposure gamma ${rec.gamma.toFixed(2)}` : 'in-model contrast normalisation')})`);
  const boxG = [[box_[2] * f + off[2], box_[3] * f + off[2]], [box_[0] * f + off[1], box_[1] * f + off[1]]];
  window.__hwNg.last = { fovG: [gx0_, gy0_], boxG, covers: boxG[0][0] <= gx0_[0] && boxG[0][1] >= gx0_[1] && boxG[1][0] <= gy0_[0] && boxG[1][1] >= gy0_[1], n: ann.length, model: m.model, level: m.level, planes: nz, tiles: use.length, fov: [box_[3] - box_[2], box_[1] - box_[0]], zspan: ann.length ? [Math.min(...comps.map((c) => c.z)), Math.max(...comps.map((c) => c.z))] : null, meanRz: comps.length ? comps.reduce((a_, c) => a_ + c.rz, 0) / comps.length : 0, edgeZ: comps.filter((c) => c.edgeZ).length, ...(ref ? { ref: { key, diamUm, seconds, nTiles, ...ndInfo() } } : {}) };
}
const $ = (id) => document.getElementById(id);
const viewKey = () => { const j = viewer.state.toJSON(); return JSON.stringify([j.position, j.crossSectionScale, j.layout, (j.layers || []).filter((l) => !/^hwseg /.test(l.name)).map((l) => l.name)]); };
let lastView = viewKey(), timer = null, busy = Promise.resolve();
const run = () => { window.__hwNg.busy = true; busy = busy.then(() => detect()).catch((e) => say('error: ' + e.message)).finally(() => { window.__hwNg.busy = false; }); return busy; };
$('hw-run').onclick = run;
viewer.state.changed.add(() => {                                              // debounced: only after the view settles, and never for our own annotation write
  if (!$('hw-auto').checked || viewKey() === lastView) return; lastView = viewKey(); clearTimeout(timer); say('view changed: detecting after it settles...'); timer = setTimeout(run, 600);
});
window.__hwNg.run = run; window.__hwNg.runModel = (key) => { $('hw-m').value = key; return run(); };
window.__hwNg.regenSeg = async (only = null) => {                            // drop cached volumes and give segmentation layers a fresh source so Neuroglancer refetches (and we recompute)
  const stamp = Date.now().toString(36); let n = 0;
  for (const id of Object.keys(segJobs)) if (!only || id === only) delete segJobs[id]; for (const id of Object.keys(segMetaC)) if (!only || id === only) delete segMetaC[id];
  setLayerList((cur) => { for (const l of cur) if (l.type === 'segmentation') { const u = typeof l.source === 'string' ? l.source : l.source && l.source.url;
      if (u && /virtual\/seg\//.test(u) && (!only || u.includes(only))) { const nu = u.replace(/(virtual\/seg\/[A-Za-z0-9_-]+)(\.[a-z0-9]+)?/, '$1.' + stamp); if (typeof l.source === 'string') l.source = nu; else l.source = { ...l.source, url: nu }; n++; } } return cur; });
  say(n ? `regenerating ${n} segmentation layer(s) from their parameters...` : 'no segmentation layers in this view'); return n;
};
window.__hwNg.probe = probeFov;
window.__hwNg.setSegAlpha = setSegAlpha; window.__hwNg.getSegAlpha = () => segAlpha;
window.__hwNg.setDepth = (n) => { $('hw-nz').value = String(n); };
window.__hwNg.diamUm = 10; window.__hwNg.setDiam = (um) => { window.__hwNg.diamUm = +um || 10; };   // declared object diameter for the reference models
window.__hwNg.setAuto = (on, key) => { if (key) $('hw-m').value = key; $('hw-auto').checked = !!on; if (on) run(); };
window.__hwNg.ready = true; say('ready');
