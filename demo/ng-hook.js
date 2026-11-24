// Neuroglancer bridge: runs inside our own Neuroglancer build (same origin on GitHub Pages). Reads the view from
// window.viewer, runs a weight-space-programmed model on exactly that field of view with the WebGPU engine, and writes the
// detections back as a Neuroglancer annotation layer. Triggered by a button, or automatically (debounced) as you navigate.
import { parseProgram, GPUEngine } from './engine.js';
import { Zarr, Slab, clamp } from './zarr.js';
import { segRegion, adjustLogVol, recommendExposure, IDENT } from './seg.js';
import { scaledModel, FAMILIES } from './route.js';

const LAYER = 'hwseg detections';
const el = (t, css, html) => { const e = document.createElement(t); if (css) e.style.cssText = css; if (html) e.innerHTML = html; return e; };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
while (!window.viewer) await sleep(100);
const viewer = window.viewer;

// ---- small UI
const box = el('div', 'position:fixed;left:8px;bottom:8px;z-index:9999;background:rgba(20,21,26,.92);color:#e8e8e4;font:12px/1.4 system-ui;padding:8px 10px;border:1px solid #444;border-radius:8px;max-width:340px');
box.innerHTML = '<b>hwseg</b> programmed model in the loop<br><select id="hw-m"><option value="cells">cells (nuclei)</option><option value="vessels_26dir">vessels</option></select> <button id="hw-run">Detect in this view</button> <label><input type="checkbox" id="hw-auto"> auto</label><div id="hw-s" style="color:#9a9a94;margin-top:4px">idle</div>';
document.body.append(box);
const say = (t) => { document.getElementById('hw-s').textContent = t; window.__hwNg.status = t; };
window.__hwNg = { status: '', runs: 0 };

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
  const txt = JSON.stringify(L.source); const ids = [...new Set([...txt.matchAll(/dandiarchive\.s3\.amazonaws\.com\/zarr\/([0-9a-f-]{36})/g)].map((m) => m[1]))];
  if (!ids.length) throw new Error('image layer is not a DANDI zarr source'); return ids;
}
async function openSource(ids) {
  const key = ids.join('+'); if (sources[key]) return sources[key];
  say(`opening ${ids.length > 1 ? `stitched slab (${ids.length} strips)` : 'strip'}...`);
  const z = ids.length > 1 ? new Slab(ids) : new Zarr(ids[0]); await z.init();
  // Neuroglancer places each zarr at its OME translation: global voxel = local index + translation / voxel size
  const L0 = z.levels[0], off = ids.length > 1 ? L0.origin : L0.tr; z.__off = [0, 1, 2].map((a) => off[a] / L0.sp[a]); return (sources[key] = z);
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

async function detect() {
  if (!engine) return; window.__hwNg.runs++;
  const j = viewer.state.toJSON(), disp = j.displayDimensions, fov = await probeFov();
  const d = j.dimensions || {}, sp0 = { z: d.z && d.z[0], y: d.y && d.y[0], x: d.x && d.x[0] }; if (!sp0.x) throw new Error('state has no x/y/z dimensions');
  const zarr = await openSource(imageSource(j)), off = zarr.__off, L0 = zarr.levels[0];
  const gx = [fov.tl[3], fov.br[3]].sort((a, b) => a - b), gy = [fov.tl[2], fov.br[2]].sort((a, b) => a - b), gz = fov.c[1];
  if (!(gx[1] - gx[0] > 1 && gy[1] - gy[0] > 1)) throw new Error('show the x-y plane (displayDimensions x, y, z and layout xy)');
  // level-0 local voxel box of the FOV
  const lx = [gx[0] - off[2], gx[1] - off[2]], ly = [gy[0] - off[1], gy[1] - off[1]], lz = gz - off[0];
  const umPerPx = ((gx[1] - gx[0]) * L0.sp[2]) / fov.w;                       // physical size of a screen pixel
  const family = $('hw-m').value, m = scaledModel(family, umPerPx, zarr), prog = await getProgram(m.model), f = 2 ** m.level, L = zarr.levels[m.level];
  const box_ = [clamp(Math.floor(ly[0] / f), 0, L.shape[1] - 1), clamp(Math.ceil(ly[1] / f), 1, L.shape[1]), clamp(Math.floor(lx[0] / f), 0, L.shape[2] - 1), clamp(Math.ceil(lx[1] / f), 1, L.shape[2])];
  const cap = 384, cy = (box_[0] + box_[1]) >> 1, cx = (box_[2] + box_[3]) >> 1;                 // keep one pass within device memory: centre crop
  const y0 = Math.max(0, cy - Math.min(cap >> 1, (box_[1] - box_[0]) >> 1)), y1 = Math.min(L.shape[1], cy + Math.min(cap >> 1, (box_[1] - box_[0]) >> 1)), x0 = Math.max(0, cx - Math.min(cap >> 1, (box_[3] - box_[2]) >> 1)), x1 = Math.min(L.shape[2], cx + Math.min(cap >> 1, (box_[3] - box_[2]) >> 1));
  const zl = clamp(Math.round(lz / f), 0, L.shape[0] - 1);
  say(`${m.model} on ${x1 - x0}x${y1 - y0} voxels at level ${m.level}: streaming...`);
  const rec = await recommendExposure(zarr, engine, await getProgram('exposure'), m.level, zl, y0, y1, x0, x1);
  say(`${m.model}: running on WebGPU...`);
  const r = await segRegion(zarr, engine, prog, m.level, zl, y0, y1, x0, x1, (v) => adjustLogVol(v, rec.lo, rec.hi, { b: 0, c: 1, g: rec.gamma }), { halo: false });
  const comps = components(r.out.seg, r.dims, r.ox, r.oy, r.w, r.h), sL = L.sp, R = (c) => c.r * sL[2];
  // voxel (level m.level, local) -> Neuroglancer global voxel (level 0 + translation)
  // Annotation space = the viewer's own dimensions in the same order (t, z, y, x): any other space reorders the global
  // dimensions and resets the view. Dimensions we do not segment (t) take the current position and radius 1.
  const keys = Object.keys(d), pos = j.position || [], dimJSON = Object.fromEntries(keys.map((kk) => [kk, d[kk]]));
  const ann = comps.map((c, i) => { const g = { z: zl * f + off[0], y: (y0 + c.y) * f + off[1], x: (x0 + c.x) * f + off[2] };
    const rad = { z: Math.max(1, R(c) / L0.sp[0]), y: Math.max(1, R(c) / L0.sp[1]), x: Math.max(1, R(c) / L0.sp[2]) };
    return { type: 'ellipsoid', id: 'd' + i, center: keys.map((kk, q) => (g[kk] !== undefined ? g[kk] : pos[q] || 0)), radii: keys.map((kk) => rad[kk] || 1) }; });
  const k = viewer.state.toJSON(); k.layers = (k.layers || []).filter((l) => l.name !== LAYER);
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
const run = () => { busy = busy.then(() => detect()).catch((e) => say('error: ' + e.message)); return busy; };
$('hw-run').onclick = run;
viewer.state.changed.add(() => {                                              // debounced: only after the view settles, and never for our own annotation write
  if (!$('hw-auto').checked || viewKey() === lastView) return; lastView = viewKey(); clearTimeout(timer); say('view changed: detecting after it settles...'); timer = setTimeout(run, 600);
});
window.__hwNg.run = run; say('ready');
