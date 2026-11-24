import { runCPU, haloFor } from './engine.js';
import { clamp } from './zarr.js';

export function toCanvas(arr, w, h, f) { const c = document.createElement('canvas'); c.width = w; c.height = h; const g = c.getContext('2d'), im = g.createImageData(w, h); f(im.data); g.putImageData(im, 0, 0); return c; }

// Segment the in-plane region [y0,y1)x[x0,x1) of plane zl at pyramid `level` (with the program's attention halo).
// Returns the model output planes plus the raw log-intensity plane, all indexed on the padded crop (use ox/oy).
export async function segRegion(zarr, engine, prog, level, zl, y0, y1, x0, x1) {
  const L = zarr.levels[level], sp = L.sp, [hz, hy, hx] = haloFor(prog, sp);
  const zs = Math.max(0, zl - hz), ze = Math.min(L.shape[0], zl + hz + 1), ys = Math.max(0, y0 - hy), ye = Math.min(L.shape[1], y1 + hy), xs = Math.max(0, x0 - hx), xe = Math.min(L.shape[2], x1 + hx);
  const vol = await zarr.crop(level, zs, ze, ys, ye, xs, xe), dims = [ze - zs, ye - ys, xe - xs], zc = zl - zs, pl = dims[1] * dims[2];
  let out;
  if (engine) out = await engine.run(prog, vol, dims, sp, zc);
  else {
    if (dims[0] * dims[1] * dims[2] > 150000) throw new Error('tile too large for CPU reference');
    const o = runCPU(prog, vol, dims, sp); out = {}; for (const k in o) out[k] = o[k].subarray(zc * pl, (zc + 1) * pl);
  }
  return { out, raw: vol.subarray(zc * pl, (zc + 1) * pl), dims, ox: x0 - xs, oy: y0 - ys, w: x1 - x0, h: y1 - y0, level };
}

export const COLORS = { bg: [90, 90, 100], gray: [255, 160, 40], white: [60, 140, 255],
  L1: [253, 231, 37], L2: [170, 220, 50], L3: [94, 201, 98], L4: [33, 145, 140], L5: [59, 82, 139], L6: [110, 40, 150], WM: [215, 215, 235] };
export const CLASSES = { tissue: ['bg', 'gray', 'white'], layers: ['bg', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'WM'] };
const argmaxAt = (out, names, i) => { let b = 0, bv = -Infinity; for (let k = 0; k < names.length; k++) { const v = out[names[k]][i]; if (v > bv) { bv = v; b = k; } } return b; };

export function renderSegCanvas(kind, r) {
  const { out, dims, ox, oy, w, h } = r;
  return toCanvas(null, w, h, (px) => {
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = (oy + y) * dims[2] + ox + x, o = 4 * (y * w + x);
      if (CLASSES[kind]) { const a = argmaxAt(out, CLASSES[kind], i), c = COLORS[CLASSES[kind][a]]; px[o] = c[0]; px[o + 1] = c[1]; px[o + 2] = c[2]; px[o + 3] = a === 0 ? 150 : 200; }
      else { const v = clamp(out.seg[i], 0, 1), c = kind === 'vessels' ? [40, 220, 255] : [255, 60, 200]; px[o] = c[0]; px[o + 1] = c[1]; px[o + 2] = c[2]; px[o + 3] = v * 255; }
    }
  });
}

// Per-region summary statistics from the model outputs (on the unpadded region).
export function regionStats(kind, r, sp) {
  const { out, dims, ox, oy, w, h } = r, n = w * h, areaMm2 = (w * sp[2]) * (h * sp[1]) * 1e-6;
  if (CLASSES[kind]) {
    const names = CLASSES[kind], c = new Array(names.length).fill(0);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) c[argmaxAt(out, names, (oy + y) * dims[2] + ox + x)]++;
    return Object.fromEntries(names.map((nm, k) => [nm, c[k] / n]));
  }
  let on = 0; const mask = new Uint8Array(n);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (out.seg[(oy + y) * dims[2] + ox + x] > 0.5) { mask[y * w + x] = 1; on++; }
  if (kind === 'vessels') return { area_frac: on / n };
  // 2D connected components (4-neighbour) -> cell count in this plane
  let cc = 0; const seen = new Uint8Array(n), st = [];
  for (let i = 0; i < n; i++) if (mask[i] && !seen[i]) {
    cc++; seen[i] = 1; st.push(i);
    while (st.length) { const j = st.pop(), x = j % w, y = (j / w) | 0;
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) { const xx = x + dx, yy = y + dy; if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue; const k = yy * w + xx; if (mask[k] && !seen[k]) { seen[k] = 1; st.push(k); } } }
  }
  return { cells_per_mm2: cc / areaMm2, area_frac: on / n };
}
