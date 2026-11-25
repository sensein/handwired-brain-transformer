import { runCPU, haloFor } from './engine.js';
import { clamp } from './zarr.js';

export function toCanvas(arr, w, h, f) { const c = document.createElement('canvas'); c.width = w; c.height = h; const g = c.getContext('2d'), im = g.createImageData(w, h); f(im.data); g.putImageData(im, 0, 0); return c; }

// Segment the in-plane region [y0,y1)x[x0,x1) of plane zl at pyramid `level` (with the program's attention halo).
// Returns the model output planes plus the raw log-intensity plane, all indexed on the padded crop (use ox/oy).
// Brightness / contrast / gamma on log-intensity normalized to a display window [lo, hi]:
//   u = clamp((v-lo)/(hi-lo)); u' = clamp((u-0.5)*contrast + 0.5 + brightness)^gamma
export const IDENT = { b: 0, c: 1, g: 1 };
export const isIdent = (a) => a.b === 0 && a.c === 1 && a.g === 1;
export function adjustU(u, a) { u = (u - 0.5) * a.c + 0.5 + a.b; u = u < 0 ? 0 : u > 1 ? 1 : u; return a.g === 1 ? u : Math.pow(u, a.g); }
// Model input = the display image u' in [0,1] (what the viewer shows): window -> contrast/brightness -> gamma.
export function adjustLogVol(vol, lo, hi, a) {
  const out = new Float32Array(vol.length), r = Math.max(hi - lo, 1e-6);
  for (let i = 0; i < vol.length; i++) { const u = (vol[i] - lo) / r; out[i] = adjustU(u < 0 ? 0 : u > 1 ? 1 : u, a); }
  return out;
}
export const percentileWindow = (v, p0 = 0.02, p1 = 0.997) => { const s = Float32Array.from(v).sort(); const lo = s[Math.floor(s.length * p0)], hi = s[Math.floor(s.length * p1)]; return [lo, Math.max(hi, lo + 1)]; };

// opts.halo=false: the model sees only the requested in-plane region (view-dependent: in-context statistics and
// attention are computed over the field of view and nothing outside it); depth context (z slab) is still included.
export async function segRegion(zarr, engine, prog, level, zl, y0, y1, x0, x1, adjustFn = null, opts = {}) {
  const L = zarr.levels[level], sp = L.sp; let [hz, hy, hx] = haloFor(prog, sp);
  if (opts.halo === false) { hy = 0; hx = 0; }
  if (opts.onStage) opts.onStage('fetch');
  const zs = Math.max(0, zl - hz), ze = Math.min(L.shape[0], zl + hz + 1), ys = Math.max(0, y0 - hy), ye = Math.min(L.shape[1], y1 + hy), xs = Math.max(0, x0 - hx), xe = Math.min(L.shape[2], x1 + hx);
  let vol = await zarr.crop(level, zs, ze, ys, ye, xs, xe);
  if (adjustFn) vol = adjustFn(vol);
  const dims = [ze - zs, ye - ys, xe - xs], zc = zl - zs, pl = dims[1] * dims[2];
  if (opts.onStage) opts.onStage('run');
  let out;
  if (engine) out = await engine.run(prog, vol, dims, sp, zc);
  else {
    if (dims[0] * dims[1] * dims[2] > 150000) throw new Error('tile too large for CPU reference');
    const o = runCPU(prog, vol, dims, sp); out = {}; for (const k in o) out[k] = o[k].subarray(zc * pl, (zc + 1) * pl);
  }
  return { out, raw: vol.subarray(zc * pl, (zc + 1) * pl), dims, ox: x0 - xs, oy: y0 - ys, w: x1 - x0, h: y1 - y0, level };
}

export const COLORS = { bg: [90, 90, 100], gray: [255, 160, 40], white: [60, 140, 255], tissue: [90, 200, 120], cortex: [255, 160, 40], wm: [60, 140, 255], sup: [255, 230, 60], mid: [110, 210, 110], deep: [40, 170, 170],
  L1: [253, 231, 37], L2: [170, 220, 50], L3: [94, 201, 98], L4: [33, 145, 140], L5: [59, 82, 139], L6: [110, 40, 150], WM: [215, 215, 235] };
export const CLASSES = { tissue: ['bg', 'gray', 'white'], layers: ['bg', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'WM'],
  // hand-written hard class maps: all channels zero = the model says nothing (dark / flat / unreadable view), drawn as no overlay
  tissue2: ['bg', 'tissue'], cortex_wm: ['bg', 'cortex', 'wm'], layers2: ['bg', 'sup', 'mid', 'deep', 'wm'] };
export const HARD = new Set(['tissue2', 'cortex_wm', 'layers2']);
// Scale-invariant detectors with size classes: 'seg' gates, the class channels colour by size / type.
export const MULTI = { blobs: ['nucleus', 'soma', 'cluster'], vessels_ms: ['capillary', 'vessel', 'large'] };
export const MULTI_COLORS = { nucleus: [255, 60, 200], soma: [255, 170, 40], cluster: [120, 200, 255], capillary: [40, 220, 255], vessel: [80, 255, 120], large: [255, 230, 60] };
export const multiNames = (kind, out) => MULTI[kind].filter((n) => out[n] !== undefined);   // v2 programs have no cluster / large channel
const argmaxAt = (out, names, i) => { let b = 0, bv = -Infinity; for (let k = 0; k < names.length; k++) { const v = out[names[k]][i]; if (v > bv) { bv = v; b = k; } } return b; };

export function renderSegCanvas(kind, r) {
  const { out, dims, ox, oy, w, h } = r;
  return toCanvas(null, w, h, (px) => {
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = (oy + y) * dims[2] + ox + x, o = 4 * (y * w + x);
      if (MULTI[kind]) { if (out.seg[i] > 0.5) { const mn = multiNames(kind, out), c = MULTI_COLORS[mn[argmaxAt(out, mn, i)]]; px[o] = c[0]; px[o + 1] = c[1]; px[o + 2] = c[2]; px[o + 3] = 230; } }
      else if (CLASSES[kind]) { const a = argmaxAt(out, CLASSES[kind], i); if (HARD.has(kind) && out[CLASSES[kind][a]][i] <= 0.5) continue; const c = COLORS[CLASSES[kind][a]]; px[o] = c[0]; px[o + 1] = c[1]; px[o + 2] = c[2]; px[o + 3] = a === 0 ? 150 : 200; }
      else { const v = clamp(out.seg[i], 0, 1), c = kind === 'vessels' ? [40, 220, 255] : [255, 60, 200]; px[o] = c[0]; px[o + 1] = c[1]; px[o + 2] = c[2]; px[o + 3] = v * 255; }
    }
  });
}

// Per-region summary statistics from the model outputs (on the unpadded region).
export function regionStats(kind, r, sp) {
  const { out, dims, ox, oy, w, h } = r, n = w * h, areaMm2 = (w * sp[2]) * (h * sp[1]) * 1e-6;
  if (CLASSES[kind]) {
    const names = CLASSES[kind], c = new Array(names.length).fill(0);
    let none = 0;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { const i = (oy + y) * dims[2] + ox + x, a = argmaxAt(out, names, i); if (HARD.has(kind) && out[names[a]][i] <= 0.5) none++; else c[a]++; }
    return { ...Object.fromEntries(names.map((nm, k) => [nm, c[k] / n])), ...(HARD.has(kind) ? { none: none / n } : {}) };
  }
  const mn = MULTI[kind] ? multiNames(kind, out) : null, cls = mn ? new Array(mn.length).fill(0) : null;
  let on = 0; const mask = new Uint8Array(n);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (out.seg[(oy + y) * dims[2] + ox + x] > 0.5) { mask[y * w + x] = 1; on++; if (cls) cls[argmaxAt(out, mn, (oy + y) * dims[2] + ox + x)]++; }
  if (kind === 'vessels') return { area_frac: on / n };
  if (kind === 'vessels_ms') return { area_frac: on / n, ...Object.fromEntries(mn.map((nm, k) => [nm, on ? cls[k] / on : 0])) };
  // 2D connected components (4-neighbour) -> cell count in this plane
  let cc = 0; const seen = new Uint8Array(n), st = [];
  for (let i = 0; i < n; i++) if (mask[i] && !seen[i]) {
    cc++; seen[i] = 1; st.push(i);
    while (st.length) { const j = st.pop(), x = j % w, y = (j / w) | 0;
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) { const xx = x + dx, yy = y + dy; if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue; const k = yy * w + xx; if (mask[k] && !seen[k]) { seen[k] = 1; st.push(k); } } }
  }
  return { cells_per_mm2: cc / areaMm2, area_frac: on / n, ...(cls ? Object.fromEntries(mn.map((nm, k) => [nm, on ? cls[k] / on : 0])) : {}) };
}

// Segment every plane of [z0,z1) over the region (view-dependent: halo=false means in-plane context is the FOV only).
export async function segVolume(zarr, engine, prog, level, z0, z1, y0, y1, x0, x1, adjustFn = null, opts = {}) {
  const L = zarr.levels[level], sp = L.sp; let [hz, hy, hx] = haloFor(prog, sp);
  if (opts.halo === false) { hy = 0; hx = 0; }
  const zs = Math.max(0, z0 - hz), ze = Math.min(L.shape[0], z1 + hz), ys = Math.max(0, y0 - hy), ye = Math.min(L.shape[1], y1 + hy), xs = Math.max(0, x0 - hx), xe = Math.min(L.shape[2], x1 + hx);
  let vol = await zarr.crop(level, zs, ze, ys, ye, xs, xe);
  if (adjustFn) vol = adjustFn(vol);
  const dims = [ze - zs, ye - ys, xe - xs], plane = dims[1] * dims[2], zc = z0 - zs, zhi = z1 - 1 - zs;
  let out;
  if (engine) out = await engine.run(prog, vol, dims, sp, zc, { zhi });
  else { const o = runCPU(prog, vol, dims, sp); out = {}; for (const k in o) out[k] = o[k].subarray(zc * plane, (zhi + 1) * plane); }
  return { out, dims, ox: x0 - xs, oy: y0 - ys, w: x1 - x0, h: y1 - y0, nz: z1 - z0, plane };
}

// Uint8 label volume (z,y,x): 0 = nothing, k>0 = class k (tissue/layers: bg is 0) or 1 for binary segmentations.
export function labelVolume(kind, r) {
  const { out, dims, ox, oy, w, h, nz, plane } = r, lab = new Uint8Array(nz * h * w), names = CLASSES[kind];
  for (let z = 0; z < nz; z++) for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const i = z * plane + (oy + y) * dims[2] + ox + x, o = (z * h + y) * w + x;
    if (MULTI[kind]) lab[o] = out.seg[i] > 0.5 ? 1 + argmaxAt(out, multiNames(kind, out), i) : 0;
    else if (names) { let b = 0, bv = -Infinity; for (let k = 0; k < names.length; k++) { const v = out[names[k]][i]; if (v > bv) { bv = v; b = k; } } lab[o] = HARD.has(kind) && bv <= 0.5 ? 0 : b; }
    else lab[o] = out.seg[i] > 0.5 ? 1 : 0;
  }
  return lab;
}
export function paletteFor(kind, out = null) {
  if (MULTI[kind]) return [[0, 0, 0], ...(out ? multiNames(kind, out) : MULTI[kind]).map((n) => MULTI_COLORS[n])];
  if (CLASSES[kind]) return CLASSES[kind].map((n, k) => (k === 0 ? [0, 0, 0] : COLORS[n]));
  return [[0, 0, 0], kind === 'vessels' ? [40, 220, 255] : [255, 60, 200]];
}

// Model-recommended default exposure for a field of view: runs the `exposure` program (global heads over the FOV slab)
// and returns the display window [lo, hi] (log domain) and a gamma that maps the FOV mean to ~0.4 of the display range.
export async function recommendExposure(zarr, engine, prog, level, zl, y0, y1, x0, x1) {
  // The exposure program has only global heads (statistics over every token), so its output depends on the multiset of voxel values, not on
  // layout. It therefore reads a mosaic of 96x96 patches spread uniformly over the whole field of view (not a centre crop): 4x4 patches = 147k tokens.
  const L = zarr.levels[level], sp = L.sp, P = 96, G = 4, H = y1 - y0, W = x1 - x0;
  const gy = Math.min(G, Math.max(1, Math.floor(H / P))), gx = Math.min(G, Math.max(1, Math.floor(W / P)));
  const py = Math.min(P, H), px = Math.min(P, W), jobs = [];
  for (let i = 0; i < gy; i++) for (let j = 0; j < gx; j++) {
    const cy = y0 + Math.round(((i + 0.5) / gy) * H), cx = x0 + Math.round(((j + 0.5) / gx) * W);        // patch centres on a uniform grid over the FOV
    const ys = clamp(cy - (py >> 1), y0, y1 - py), xs = clamp(cx - (px >> 1), x0, x1 - px);
    jobs.push(zarr.crop(level, zl, zl + 1, ys, ys + py, xs, xs + px).then((v) => ({ v, i, j })));
  }
  const parts = await Promise.all(jobs), MH = gy * py, MW = gx * px, vol = new Float32Array(MH * MW);
  for (const { v, i, j } of parts) for (let y = 0; y < py; y++) vol.set(v.subarray(y * px, (y + 1) * px), (i * py + y) * MW + j * px);
  const dims = [1, MH, MW];
  let o;
  if (engine) o = await engine.run(prog, vol, dims, sp, 0);
  else { const r = runCPU(prog, vol, dims, sp); o = {}; for (const k in r) o[k] = r[k]; }
  const lo = o.lo[0], hi = Math.max(o.hi[0], o.lo[0] + 0.5), mean = o.mean[0], um = clamp((mean - lo) / (hi - lo), 0.05, 0.95);
  return { lo, hi, mean, gamma: clamp(Math.log(0.4) / Math.log(um), 0.4, 2.5) };
}
