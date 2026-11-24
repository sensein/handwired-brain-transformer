// Neuroglancer bridge: runs inside our own Neuroglancer build (same origin on GitHub Pages). Reads the view from
// window.viewer, runs a weight-space-programmed model on exactly that field of view with the WebGPU engine, and writes the
// detections back as a Neuroglancer annotation layer. Triggered by a button, or automatically (debounced) as you navigate.
import { parseProgram, GPUEngine, haloFor } from './engine.js';
import { Zarr, Slab, clamp } from './zarr.js';
import { segRegion, adjustLogVol, recommendExposure, IDENT } from './seg.js';
import { scaledModel, FAMILIES } from './route.js';

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
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.addEventListener('message', async (ev) => {
    const m = ev.data; if (!m || m.kind !== 'virtual') return; const port = ev.ports[0];
    try { const r = await serveVirtual(m.ids, m.path); port.postMessage(r, [r.body]); }
    catch (e) { port.postMessage({ status: 500, type: 'text/plain', body: new TextEncoder().encode(String(e)).buffer }); }
  });
  navigator.serviceWorker.register(new URL('sw.js', import.meta.url).href).then(async () => {
    await navigator.serviceWorker.ready;   // a shared link opened directly: the first load is not controlled yet, so reload once
    if (!navigator.serviceWorker.controller && /virtual\/slab/.test(location.href) && !sessionStorage.getItem('hw-sw-reload')) { sessionStorage.setItem('hw-sw-reload', '1'); location.reload(); }
  }).catch(() => {});
}
const el = (t, css, html) => { const e = document.createElement(t); if (css) e.style.cssText = css; if (html) e.innerHTML = html; return e; };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
while (!window.viewer) await sleep(100);
const viewer = window.viewer;

// ---- small UI
const box = el('div', 'position:fixed;left:8px;bottom:8px;z-index:9999;background:rgba(20,21,26,.92);color:#e8e8e4;font:12px/1.4 system-ui;padding:8px 10px;border:1px solid #444;border-radius:8px;max-width:340px');
box.innerHTML = '<b>hwseg</b> programmed model in the loop<br><select id="hw-m"><option value="fable_blobs_v2">cells (nuclei / somata, hand-written v2)</option><option value="fable_vessels_v2">vessels (LEC, hand-written v2)</option><option value="cells">cells (builder)</option><option value="vessels_26dir">vessels (builder)</option><option value="regions">tissue outline + cortex / white-matter boundary (contours)</option></select> <button id="hw-run">Detect in this view</button> <label><input type="checkbox" id="hw-auto"> auto</label><div id="hw-s" style="color:#9a9a94;margin-top:4px">idle</div>';
const EMBEDDED = new URLSearchParams(location.search).has('embedded');    // the shell page drives us through window.__hwNg
document.body.append(box); if (EMBEDDED) box.style.display = 'none';
window.__hwNg = { status: '', runs: 0, busy: false, events: [], last: null };
const say = (t) => { document.getElementById('hw-s').textContent = t; window.__hwNg.status = t; window.__hwNg.events.push({ t: Date.now(), msg: t }); if (window.__hwNg.events.length > 200) window.__hwNg.events.shift(); };

let engine = null; try { engine = await GPUEngine.create(); } catch (e) { say('no WebGPU: detection unavailable'); }
const HERE = (p) => new URL(p, import.meta.url).href;                     // the page is /ng/index.html; models live next to this module
const index = await (await fetch(HERE('models/index.json'))).json(), programs = {};
const getProgram = async (k) => programs[k] ||= parseProgram(await (await fetch(HERE(`models/${index[k].file}`))).text());
const sources = {};

// ---- read the view: field of view recovered from the data coordinates Neuroglancer reports under the panel corners
async function probeFov() {
  const panel = [...viewer.display.panels].find((p) => p.element && p.element.getBoundingClientRect().width > 100); if (!panel) throw new Error('no cross-section panel');
  const el_ = panel.element, r = el_.getBoundingClientRect();
  const at = async (fx, fy) => { el_.dispatchEvent(new MouseEvent('mousemove', { clientX: r.left + fx * r.width, clientY: r.top + fy * r.height, bubbles: true })); await sleep(120); return Array.from(viewer.mouseState.position); };
  const c = await at(0.5, 0.5), a = await at(0.02, 0.02), b = await at(0.98, 0.98);   // hover just inside the edges, then extrapolate to the true corners
  const tl = c.map((v, i) => v - (v - a[i]) / 0.96), br = c.map((v, i) => v + (b[i] - v) / 0.96);   // the probe is 0.48 of the width from the centre; the corner is 0.5
  return { tl, br, c, w: r.width, h: r.height };
}
function imageSource(j) {
  const L = (j.layers || []).find((l) => l.type === 'image'); if (!L) throw new Error('no image layer');
  const txt = JSON.stringify(L.source), v = txt.match(/virtual\/slab\/([0-9a-f+-]+)(~s)?/);
  if (v) return { ids: v[1].split('+'), virtual: true, seams: !!v[2] };
  const ids = [...new Set([...txt.matchAll(/dandiarchive\.s3\.amazonaws\.com\/zarr\/([0-9a-f-]{36})/g)].map((m) => m[1]))];
  if (!ids.length) throw new Error('image layer is not a DANDI zarr source'); return { ids, virtual: false, seams: false };
}
async function openSource({ ids, virtual, seams }) {
  say(`opening ${ids.length > 1 || virtual ? `stitched slab (${ids.length} strips)` : 'strip'}...`);
  const z = virtual ? await getSlab(ids, seams) : (ids.length > 1 ? await getSlab(ids, false) : (sources[ids[0]] ||= await (async () => { const q = new Zarr(ids[0]); await q.init(); return q; })()));
  // zarr sources: Neuroglancer places each zarr at its OME translation (global voxel = local index + translation / voxel size);
  // the virtual volume's own grid starts at 0.
  const L0 = z.levels[0], off = virtual ? [0, 0, 0] : (ids.length > 1 ? L0.origin : L0.tr).map((o, a) => o / L0.sp[a]);
  return Object.assign(Object.create(null), { z, off });
}

// ---- 2-D connected components on the segmentation plane -> ellipsoid annotations
function components(seg, dims, ox, oy, w, h) {
  const mask = new Uint8Array(w * h), seen = new Uint8Array(w * h), out = [];
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) mask[y * w + x] = seg[(oy + y) * dims[2] + ox + x] > 0.5 ? 1 : 0;
  for (let i = 0; i < w * h; i++) if (mask[i] && !seen[i]) {
    let n = 0, sx = 0, sy = 0; const st = [i]; seen[i] = 1;
    while (st.length) { const j = st.pop(), x = j % w, y = (j / w) | 0; n++; sx += x; sy += y;
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) { const xx = x + dx, yy = y + dy; if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue; const k = yy * w + xx; if (mask[k] && !seen[k]) { seen[k] = 1; st.push(k); } } }
    if (n >= 3) out.push({ x: sx / n, y: sy / n, r: Math.sqrt(n / Math.PI) });
  }
  return out;
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
async function detectRegions(j, fov, zarr, off, d, umPerPx, sp0) {
  const keys = Object.keys(d), pos = j.position || [];
  const m0 = scaledModel('fable_cortex_wm', umPerPx, zarr), progT = await getProgram('fable_tissue'), progC = await getProgram('fable_cortex_wm');
  const A = axisIdx(d), gx = [fov.tl[A.x], fov.br[A.x]].sort((a, b) => a - b), gy = [fov.tl[A.y], fov.br[A.y]].sort((a, b) => a - b), lz = fov.c[A.z] - off[0];
  const dmax = Math.max(progT.d, progC.d); let level = m0.level, box_;
  for (;;) {                                                                       // coarsen until the whole FOV fits one pass
    const f = 2 ** level, L = zarr.levels[level], hz = Math.max(haloFor(progT, L.sp)[0], haloFor(progC, L.sp)[0]);
    box_ = [clamp(Math.floor((gy[0] - off[1]) / f), 0, L.shape[1] - 1), clamp(Math.ceil((gy[1] - off[1]) / f), 1, L.shape[1]), clamp(Math.floor((gx[0] - off[2]) / f), 0, L.shape[2] - 1), clamp(Math.ceil((gx[1] - off[2]) / f), 1, L.shape[2])];
    if ((2 * hz + 1) * (box_[1] - box_[0]) * (box_[3] - box_[2]) <= Math.floor(engine.maxTokens(dmax) * 0.6) || level >= zarr.top) break; level++;
  }
  const f = 2 ** level, L = zarr.levels[level], zl = clamp(Math.round(lz / f), 0, L.shape[0] - 1);
  say(`regions on ${box_[3] - box_[2]}x${box_[1] - box_[0]} voxels at level ${level}: streaming...`);
  const rec = await recommendExposure(zarr, engine, await getProgram('exposure'), level, zl, box_[0], box_[1], box_[2], box_[3]);
  const adjust = (v) => adjustLogVol(v, rec.lo, rec.hi, { b: 0, c: 1, g: rec.gamma });
  say('regions: tissue program on WebGPU...');
  const rT = await segRegion(zarr, engine, progT, level, zl, box_[0], box_[1], box_[2], box_[3], adjust, { halo: false });
  say('regions: cortex / white-matter program on WebGPU...');
  const rC = await segRegion(zarr, engine, progC, level, zl, box_[0], box_[1], box_[2], box_[3], adjust, { halo: false });
  const w = rT.w, h = rT.h, tis = new Uint8Array(w * h), wm = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { const q = y * w + x; tis[q] = rT.out.tissue[(rT.oy + y) * rT.dims[2] + rT.ox + x] > 0.5 ? 1 : 0; wm[q] = rC.out.wm[(rC.oy + y) * rC.dims[2] + rC.ox + x] > 0.5 ? 1 : 0; }
  const toG = (p) => { const g = { z: zl * f + off[0], y: (box_[0] + p[1]) * f + off[1], x: (box_[2] + p[0]) * f + off[2] }; return keys.map((kk, q) => (g[kk] !== undefined ? g[kk] : pos[q] || 0)); };
  const mk = (segs, tag) => segs.map((s2, i) => ({ type: 'line', id: tag + i, pointA: toG(s2[0]), pointB: toG(s2[1]) }));
  const layers = [['hwseg tissue outline', '#ffd23c', mk(contourSegments(tis, w, h), 't')], ['hwseg cortex/white-matter boundary', '#3cd2ff', mk(contourSegments(wm, w, h), 'w')]];
  const k = viewer.state.toJSON(); k.layers = (k.layers || []).filter((l) => !/^hwseg /.test(l.name));
  const LINE_SHADER = 'void main() { setColor(defaultColor()); setLineWidth(1.5); setEndpointMarkerSize(0.0); setEndpointMarkerBorderWidth(0.0); }';   // thin lines, no endpoint markers
  for (const [name, col, ann] of layers) k.layers.push({ type: 'annotation', name, annotationColor: col, shader: LINE_SHADER, source: { url: 'local://annotations', transform: { outputDimensions: Object.fromEntries(keys.map((kk) => [kk, d[kk]])) } }, annotations: ann });
  const img = k.layers.find((l) => l.type === 'image');
  if (img && /uicontrol float lo/.test(String(img.shader || ''))) img.shaderControls = { ...(img.shaderControls || {}), lo: +rec.lo.toFixed(2), hi: +rec.hi.toFixed(2), gamma: +rec.gamma.toFixed(2) };
  k.displayDimensions = ['x', 'y', 'z']; k.layout = k.layout || 'xy';
  lastView = viewKey(); viewer.state.reset(); await sleep(60); viewer.state.restoreState(k);
  const tp = 100 * tis.reduce((a, b) => a + b, 0) / (w * h), wp = 100 * wm.reduce((a, b) => a + b, 0) / (w * h);
  say(`${layers[0][2].length} outline + ${layers[1][2].length} cortex/WM segments (tissue ${tp.toFixed(0)}%, white matter ${wp.toFixed(0)}%, level ${level}, ${umPerPx.toFixed(0)} um/px, FOV ${w}x${h} voxels)`);
  window.__hwNg.last = { regions: true, outline: layers[0][2].length, boundary: layers[1][2].length, tissuePct: tp, wmPct: wp, level, fov: [w, h] };
}

async function detect() {
  if (!engine) return; window.__hwNg.runs++;
  const j = viewer.state.toJSON(), disp = j.displayDimensions, fov = await probeFov();
  const d = j.dimensions || {}, sp0 = { z: d.z && d.z[0], y: d.y && d.y[0], x: d.x && d.x[0] }; if (!sp0.x) throw new Error('state has no x/y/z dimensions');
  const src = await openSource(imageSource(j)), zarr = src.z, off = src.off, L0 = zarr.levels[0], A = axisIdx(d);
  const gx = [fov.tl[A.x], fov.br[A.x]].sort((a, b) => a - b), gy = [fov.tl[A.y], fov.br[A.y]].sort((a, b) => a - b), gz = fov.c[A.z];
  if (!(gx[1] - gx[0] > 1 && gy[1] - gy[0] > 1)) throw new Error('show the x-y plane (displayDimensions x, y, z and layout xy)');
  // level-0 local voxel box of the FOV
  const lx = [gx[0] - off[2], gx[1] - off[2]], ly = [gy[0] - off[1], gy[1] - off[1]], lz = gz - off[0];
  const umPerPx = ((gx[1] - gx[0]) * L0.sp[2]) / fov.w;                       // physical size of a screen pixel
  if ($('hw-m').value === 'regions') return detectRegions(j, fov, zarr, off, d, umPerPx, sp0);
  if (umPerPx > 16) { say(`zoom in first: cells and vessels need a field of view of a few mm (this view is ${umPerPx.toFixed(0)} um/px)`); return; }
  const family = $('hw-m').value, m = scaledModel(family, umPerPx, zarr), prog = await getProgram(m.model), f = 2 ** m.level, L = zarr.levels[m.level];
  const box_ = [clamp(Math.floor(ly[0] / f), 0, L.shape[1] - 1), clamp(Math.ceil(ly[1] / f), 1, L.shape[1]), clamp(Math.floor(lx[0] / f), 0, L.shape[2] - 1), clamp(Math.ceil(lx[1] / f), 1, L.shape[2])];
  const cap = 384, core = 320, hal = (cap - core) >> 1, zl = clamp(Math.round(lz / f), 0, L.shape[0] - 1);
  const cy = (box_[0] + box_[1]) >> 1, cx = (box_[2] + box_[3]) >> 1;
  const cy0 = Math.max(0, cy - (cap >> 1)), cy1 = Math.min(L.shape[1], cy + (cap >> 1)), cx0 = Math.max(0, cx - (cap >> 1)), cx1 = Math.min(L.shape[2], cx + (cap >> 1));
  say(`${m.model}: streaming exposure window...`);
  const rec = await recommendExposure(zarr, engine, await getProgram('exposure'), m.level, zl, cy0, cy1, cx0, cx1);
  // the whole field of view is covered by overlapping tiles (core + halo, one GPU pass each); a detection belongs to the tile whose core holds its centre
  const tiles = []; for (let ty = box_[0]; ty < box_[1]; ty += core) for (let tx = box_[2]; tx < box_[3]; tx += core) tiles.push([ty, Math.min(ty + core, box_[1]), tx, Math.min(tx + core, box_[3])]);
  tiles.sort((p, q) => Math.hypot((p[0] + p[1]) / 2 - cy, (p[2] + p[3]) / 2 - cx) - Math.hypot((q[0] + q[1]) / 2 - cy, (q[2] + q[3]) / 2 - cx));   // centre first
  const MAXT = 100, use = tiles.slice(0, MAXT); let sL = L.sp, R = (c) => c.r * sL[2], comps = [], y0 = box_[0], x0 = box_[2], y1 = box_[1], x1 = box_[3];
  for (let ti = 0; ti < use.length; ti++) {
    const [cy_0, cy_1, cx_0, cx_1] = use[ti], Y0 = Math.max(0, cy_0 - hal), Y1 = Math.min(L.shape[1], cy_1 + hal), X0 = Math.max(0, cx_0 - hal), X1 = Math.min(L.shape[2], cx_1 + hal);
    say(`${m.model}: tile ${ti + 1}/${use.length} (${X1 - X0}x${Y1 - Y0} voxels, level ${m.level}) on WebGPU...`);
    const r = await segRegion(zarr, engine, prog, m.level, zl, Y0, Y1, X0, X1, (v) => adjustLogVol(v, rec.lo, rec.hi, { b: 0, c: 1, g: rec.gamma }), { halo: false });
    for (const c of components(r.out.seg, r.dims, r.ox, r.oy, r.w, r.h)) { const gy_ = Y0 + c.y, gx_ = X0 + c.x; if (gy_ >= cy_0 && gy_ < cy_1 && gx_ >= cx_0 && gx_ < cx_1) comps.push({ ...c, y: gy_ - 0, x: gx_ - 0 }); }
  }
  y0 = 0; x0 = 0; y1 = box_[1]; x1 = box_[3];           // detections are already in level-local voxel coordinates
  if (tiles.length > use.length) say(`field of view needs ${tiles.length} tiles; the ${use.length} nearest the centre were run - zoom in for full coverage`);
  // voxel (level m.level, local) -> Neuroglancer global voxel (level 0 + translation)
  // Annotation space = the viewer's own dimensions in the same order (t, z, y, x): any other space reorders the global
  // dimensions and resets the view. Dimensions we do not segment (t) take the current position and radius 1.
  const keys = Object.keys(d), pos = j.position || [], dimJSON = Object.fromEntries(keys.map((kk) => [kk, d[kk]]));
  const ann = comps.map((c, i) => { const g = { z: zl * f + off[0], y: (y0 + c.y) * f + off[1], x: (x0 + c.x) * f + off[2] };
    const rad = { z: Math.max(1, R(c) / L0.sp[0]), y: Math.max(1, R(c) / L0.sp[1]), x: Math.max(1, R(c) / L0.sp[2]) };
    return { type: 'ellipsoid', id: 'd' + i, center: keys.map((kk, q) => (g[kk] !== undefined ? g[kk] : pos[q] || 0)), radii: keys.map((kk) => rad[kk] || 1) }; });
  const k = viewer.state.toJSON(); k.layers = (k.layers || []).filter((l) => l.name !== LAYER && !/^hwseg (tissue|gray|cortex)/.test(l.name));
  k.layers.push({ type: 'annotation', name: LAYER, annotationColor: '#ff3cc8', source: { url: 'local://annotations', transform: { outputDimensions: dimJSON } }, annotations: ann });
  // the model's exposure becomes Neuroglancer's own window: log-domain lo / hi sliders of the layer shader
  const img = k.layers.find((l) => l.type === 'image' && l.shaderControls !== undefined || l.type === 'image');
  if (img && /uicontrol float lo/.test(String(img.shader || ''))) img.shaderControls = { ...(img.shaderControls || {}), lo: +rec.lo.toFixed(2), hi: +rec.hi.toFixed(2), gamma: +rec.gamma.toFixed(2) };
  k.displayDimensions = ['x', 'y', 'z']; k.layout = k.layout || 'xy';        // restoreState drops the displayed dimensions otherwise; the hook needs the x-y plane
  lastView = viewKey(); viewer.state.reset(); await sleep(60); viewer.state.restoreState(k);   // reset first: a plain restoreState keeps a broken display plane
  say(`${ann.length} detections (${m.model}, level ${m.level}, ${(umPerPx).toFixed(1)} um/px, exposure gamma ${rec.gamma.toFixed(2)})`);
  window.__hwNg.last = { n: ann.length, model: m.model, level: m.level, fov: [x1 - x0, y1 - y0] };
}
const $ = (id) => document.getElementById(id);
const viewKey = () => { const j = viewer.state.toJSON(); return JSON.stringify([j.position, j.crossSectionScale, j.layout, (j.layers || []).filter((l) => l.name !== LAYER).map((l) => l.name)]); };
let lastView = viewKey(), timer = null, busy = Promise.resolve();
const run = () => { window.__hwNg.busy = true; busy = busy.then(() => detect()).catch((e) => say('error: ' + e.message)).finally(() => { window.__hwNg.busy = false; }); return busy; };
$('hw-run').onclick = run;
viewer.state.changed.add(() => {                                              // debounced: only after the view settles, and never for our own annotation write
  if (!$('hw-auto').checked || viewKey() === lastView) return; lastView = viewKey(); clearTimeout(timer); say('view changed: detecting after it settles...'); timer = setTimeout(run, 600);
});
window.__hwNg.run = run; window.__hwNg.runModel = (key) => { $('hw-m').value = key; return run(); };
window.__hwNg.setAuto = (on, key) => { if (key) $('hw-m').value = key; $('hw-auto').checked = !!on; if (on) run(); };
window.__hwNg.ready = true; say('ready');
