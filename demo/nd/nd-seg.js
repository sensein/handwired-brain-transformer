// Neurodesk reference models (github.com/neurodesk/webapps) run on tiles of our streamed light-sheet volumes.
// The pre/post-processing ports (cellpose-core.js, stardist-core.js, cellfinder-detect.js) are vendored from the Neurodesk apps;
// the network weights are NOT in this repository: they are fetched at run time from the Neurodesk Hugging Face asset store
// (CORS-enabled, pinned revisions, checked against the published sha256) and executed with onnxruntime-web from a pinned CDN.
//
// Every runner takes a tile of the pyramid as log(I+1) (z, y, x), the voxel spacing in um and a declared object diameter in um,
// and returns either linked 3-D instances (same linking rule as our own programs: demo/instances.js) or a class mask.
import { resizePlane2D, runNet2D, followFlows2D, clusterMasks, removeBadFlowMasks2D, fillHolesAndRemoveSmallMasks } from './cellpose-core.js';
import * as SD from './stardist-core.js';
import { createDetector } from './cellfinder-detect.js';
import { labelComponents, linkTracks } from '../instances.js';

export const ORT_VERSION = '1.29.0';
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const CLIPPER_URL = 'https://cdn.jsdelivr.net/npm/clipper-lib@6.4.2/clipper.js';
const HF = 'https://huggingface.co/datasets/';

// Model cards. `px` = the object diameter in pixels the network was trained for: each plane is resampled (anisotropically:
// DANDI 000108 voxels are 2.56 x 3.63 um in x / y at level 0) so that an object of the declared diameter spans `px` pixels.
export const REF = {
  nd_cellpose: {
    label: 'Cellpose 3 nuclei (Neurodesk reference)', kind: 'cells', dims: 2, px: 17,
    url: HF + 'neurodeskorg/webapps/resolve/bc1de48f8c0857d30f2330e986eef29ec77603b3/cellpose/models/cellpose-nuclei.onnx',
    bytes: 26489378, sha256: 'e6ebcede6fca35ae50916257952fe8a7af085ec1a8ffad732cc0ad16a4e8d82a',
    licence: 'BSD-3-Clause code and weights (HHMI); training data include CC-BY-NC sets: check upstream terms before commercial use',
    paper: 'Stringer et al., Nat Methods 2021; Stringer & Pachitariu, Nat Methods 2025',
    cfg: { bsize: 224, tileOverlap: 0.1, divBy: 16, extraPad: 1, flowThresh: 0.4, cellprobThresh: 0, minSize: 15, maxSizeFraction: 0.4, lower: 1, upper: 99 },
  },
  nd_stardist: {
    label: 'StarDist 2D versatile fluorescence (Neurodesk reference)', kind: 'cells', dims: 2, px: 16,
    url: HF + 'neurodeskorg/webapps/resolve/2cc5042a0fad34811857bbdf89b1172bda338af4/stardist/models/stardist-2d-versatile-fluo.onnx',
    bytes: 5715335, sha256: 'cc5cb72c2b47ebdf55296fdb03cca87a512f458ee1defb0169cfe2506de5f0d3',
    licence: 'BSD-3-Clause (StarDist models, Weigert & Schmidt); clipper-lib 6.4.2 (Boost) for polygon NMS',
    paper: 'Schmidt et al., MICCAI 2018',
    cfg: { channels: 1, grid: [2, 2], divBy: [16, 16], tileOverlap: [94, 94], nRays: 32, thresholds: { prob: 0.479071463157368, nms: 0.3 }, lower: 1, upper: 99.8 },
  },
  nd_cellfinder: {
    label: 'cellfinder candidate detection (Neurodesk JS port, classical)', kind: 'cells', dims: 3, px: null, url: null, zCtx: 8,   // + 8 planes each side: the 15 um ball filter needs depth before its first valid plane
    licence: 'BSD-3-Clause (cellfinder 1.10.1, Tyson et al.); no network weights',
    paper: 'Tyson et al., PLOS Comput Biol 2021',
    // threshold mean + 3 SD of the filtered plane: cellfinder's default 10 SD (sparse labels in mouse whole-brain light-sheet) found no
    // candidates at all in dense pan-nuclear YO windows (0 of ~400; 5 SD: 44, 3 SD: 390 on the same 208 x 208 window)
    cfg: { nSdsAboveMeanThresh: 3 },
  },
  nd_vesselboost: {
    label: 'VesselBoost 3-D U-Net (Neurodesk reference, MRA-trained, experimental)', kind: 'vessels', dims: 3, px: null, zPatch: 64,
    url: HF + 'sbollmann/neurodesk-webapps-assets/resolve/a8cdbf8c2874e1a2f617ecc6695244a0810eac11/vesselboost/vesselboost.onnx',
    bytes: 26334432, sha256: '8ebf3547bd2318963e7080af74244b2478cd1938a35cd8eace5ddd8edff460d9',
    licence: 'VesselBoost code MIT; the ONNX weights carry no licence statement (NOASSERTION in the Neurodesk manifest): fetched from the Neurodesk host, not redistributed',
    paper: 'Xu et al., Aperture Neuro 2024',
    cfg: { patch: 64, threshold: 0.1, minComponent: 10, sigma: 8 },
  },
};
export const isRef = (k) => Object.prototype.hasOwnProperty.call(REF, k);

// ---- runtime: onnxruntime-web (WebGPU execution provider, WASM fallback) and clipper-lib, both from pinned CDN versions
let ortP = null, clipP = null; const sessions = {}, epUsed = {};
async function getOrt() {
  return (ortP ||= (async () => { const ort = await import(/* @vite-ignore */ ORT_BASE + 'ort.webgpu.min.mjs'); ort.env.wasm.wasmPaths = ORT_BASE; ort.env.wasm.numThreads = (globalThis.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1); return ort; })());
}
async function getClipper() {
  return (clipP ||= (async () => { const src = await (await fetch(CLIPPER_URL)).text(), m = { exports: {} }; new Function('module', 'exports', src)(m, m.exports); SD.setClipper(m.exports); return m.exports; })());
}
async function sha256hex(buf) { const d = await crypto.subtle.digest('SHA-256', buf); return [...new Uint8Array(d)].map((b) => b.toString(16).padStart(2, '0')).join(''); }
// The execution provider: globalThis.__hwNdEP = 'wasm' | 'webgpu' forces one; default WebGPU first, WASM if session creation fails.
// 3-D convolutions (VesselBoost) run on WASM, as in the Neurodesk app.
export async function getSession(key, say = () => {}) {
  if (sessions[key]) return sessions[key];
  return (sessions[key] = (async () => {
    const M = REF[key], ort = await getOrt(); say(`${key}: downloading ${(M.bytes / 1e6).toFixed(1)} MB weights from the Neurodesk asset store...`);
    const r = await fetch(M.url); if (!r.ok) throw new Error(`${key}: weights fetch failed (${r.status})`);
    const buf = await r.arrayBuffer(); if (M.bytes && buf.byteLength !== M.bytes) throw new Error(`${key}: expected ${M.bytes} bytes, got ${buf.byteLength}`);
    if (M.sha256) { const h = await sha256hex(buf); if (h !== M.sha256) throw new Error(`${key}: sha256 mismatch`); }
    const force = globalThis.__hwNdEP, order = force ? [force] : (M.dims === 3 || !('gpu' in navigator) ? ['wasm'] : ['webgpu', 'wasm']);
    let last = null;
    for (const ep of order) { try { const s = await ort.InferenceSession.create(buf, { executionProviders: [ep], graphOptimizationLevel: 'all' }); epUsed[key] = ep; say(`${key}: onnxruntime-web ${ORT_VERSION} on ${ep}`); return s; } catch (e) { last = e; } }
    throw new Error(`${key}: could not create an ONNX session (${last && last.message})`);
  })().catch((e) => { delete sessions[key]; throw e; }));
}
export const ndInfo = () => ({ ort: ORT_VERSION, ep: { ...epUsed }, threads: ortP ? (globalThis.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1) : null });

// ---- helpers
const counts = (v) => { const o = new Float32Array(v.length); for (let i = 0; i < v.length; i++) o[i] = Math.expm1(v[i]); return o; };   // log(I+1) -> intensity
export function percentiles(values, lo, hi) {           // numpy-style linear percentiles on a strided sample (<= 400k values)
  const step = Math.max(1, Math.floor(values.length / 400000)), s = []; for (let i = 0; i < values.length; i += step) s.push(values[i]);
  const a = Float64Array.from(s).sort(), q = (p) => { const t = (p / 100) * (a.length - 1), b = Math.floor(t), c = Math.min(b + 1, a.length - 1); return a[b] + (a[c] - a[b]) * (t - b); };
  return [q(lo), q(hi)];
}
// Normalization window of a model over the field of view (intensity domain), from a plane or slab crop as log(I+1).
export function fovNorm(key, logVals) { const c = REF[key].cfg; if (!c.lower) return null; const [a, b] = percentiles(counts(logVals), c.lower, c.upper); return { lo: +a.toFixed(3), hi: +b.toFixed(3) }; }
const normPlane = (I, n) => { const o = new Float32Array(I.length), r = Math.max(n.hi - n.lo, 1e-3); for (let i = 0; i < I.length; i++) o[i] = (I[i] - n.lo) / r; return o; };
// resampled grid for a plane of H x W voxels with in-plane spacing (sy, sx) so that an object of diamUm spans px pixels
function resampleDims(H, W, sy, sx, diamUm, px) { const f = px / diamUm; return { RH: Math.max(16, Math.round(H * sy * f)), RW: Math.max(16, Math.round(W * sx * f)) }; }
function labelsBack(lab, RH, RW, H, W) {                // nearest-neighbour from the resampled grid back to the voxel grid
  const out = new Uint32Array(H * W), fy = RH / H, fx = RW / W;
  for (let y = 0; y < H; y++) { const ry = Math.min(RH - 1, Math.floor((y + 0.5) * fy)); for (let x = 0; x < W; x++) out[y * W + x] = lab[ry * RW + Math.min(RW - 1, Math.floor((x + 0.5) * fx))]; }
  return out;
}

// ---- 2-D models, one plane: normalized intensities on the resampled grid -> instance labels on the resampled grid
async function cellposePlane(img, RH, RW, session) {
  const c = REF.nd_cellpose.cfg, ort = await getOrt();
  const runBatch = async (data, b, h, w) => (await session.run({ input: new ort.Tensor('float32', data, [b, 2, h, w]) })).flows.data;
  const net = await runNet2D(img, null, RH, RW, { bsize: c.bsize, tileOverlap: c.tileOverlap, div: c.divBy, extra: c.extraPad, runBatch });
  for (const a of [net.dY, net.dX, net.cellprob]) for (let i = 0; i < a.length; i += 997) if (!Number.isFinite(a[i])) throw new Error('Cellpose produced non-finite flows on this execution provider');
  const { inds, pts } = followFlows2D(net.dY, net.dX, net.cellprob, RH, RW, c.cellprobThresh, 200);
  let lab = clusterMasks(pts, inds, [RH, RW], { maxSizeFraction: c.maxSizeFraction });
  if (c.flowThresh > 0) lab = removeBadFlowMasks2D(lab, net.dY, net.dX, RH, RW, c.flowThresh);
  return fillHolesAndRemoveSmallMasks(lab, [RH, RW], c.minSize);
}
async function stardistPlane(img, RH, RW, session) {
  const c = REF.nd_stardist.cfg, ort = await getOrt(); await getClipper();
  const model = { channels: 1, grid: c.grid, divBy: c.divBy, tileOverlap: c.tileOverlap, nRays: c.nRays };
  const run = async (tile, h, w) => { const o = await session.run({ input: new ort.Tensor('float32', tile, [1, h, w, 1]) }); return { prob: o.prob.data, dist: o.dist.data }; };
  const cand = await SD.predictCandidates(img, RH, RW, model, { probThresh: c.thresholds.prob, nTiles: SD.automaticTiles(RH, RW), run });
  const order = SD.descendingOrder(cand.prob), nR = c.nRays, dist = new Float32Array(order.length * nR), pts = new Int32Array(order.length * 2), prob = new Float32Array(order.length);
  order.forEach((f, t) => { dist.set(cand.dist.subarray(f * nR, (f + 1) * nR), t * nR); pts[2 * t] = cand.points[2 * f]; pts[2 * t + 1] = cand.points[2 * f + 1]; prob[t] = cand.prob[f]; });
  const kept = SD.suppress(dist, pts, nR, c.thresholds.nms), kd = new Float32Array(kept.length * nR), kp = new Int32Array(kept.length * 2), kpr = new Float32Array(kept.length);
  kept.forEach((f, t) => { kd.set(dist.subarray(f * nR, (f + 1) * nR), t * nR); kp[2 * t] = pts[2 * f]; kp[2 * t + 1] = pts[2 * f + 1]; kpr[t] = prob[f]; });
  return SD.renderLabels(SD.polygonCoordinates(kd, kp, nR), kpr, nR, RH, RW);
}

// ---- the tile runners. vol = log(I+1) (Z, H, W), sp = [z, y, x] um. opts: { diamUm, norm: {lo, hi} (intensity), say }
export async function refTile(key, vol, dims, sp, opts = {}) {
  const M = REF[key], [Z, H, W] = dims, plane = H * W, say = opts.say || (() => {}), t0 = performance.now();
  if (M.dims === 2) {
    const session = await getSession(key, say), { RH, RW } = resampleDims(H, W, sp[1], sp[2], opts.diamUm, M.px);
    const norm = opts.norm || fovNorm(key, vol), pcs = [];
    for (let z = 0; z < Z; z++) {
      const I = counts(vol.subarray(z * plane, (z + 1) * plane)), img = resizePlane2D(normPlane(I, norm), H, W, RH, RW);
      const lab = key === 'nd_cellpose' ? await cellposePlane(img, RH, RW, session) : await stardistPlane(img, RH, RW, session);
      pcs.push(labelComponents(labelsBack(lab, RH, RW, H, W), W, H));
      if (opts.onPlane) opts.onPlane(z + 1, Z);
    }
    const t = linkTracks(pcs, W, H, opts.idBase || 0);
    return { ...t, resampled: [RH, RW], norm, ms: performance.now() - t0 };
  }
  if (key === 'nd_cellfinder') {
    const det = createDetector({ height: H, width: W, voxelSizesUm: sp, dtype: 'uint16', options: { somaDiameterUm: opts.diamUm, ...M.cfg } });
    for (let z = 0; z < Z; z++) { const p = new Uint16Array(plane), v = vol.subarray(z * plane, (z + 1) * plane); for (let i = 0; i < plane; i++) p[i] = Math.min(65535, Math.round(Math.expm1(v[i]))); det.push(p); }
    const { cells } = det.finish(), lab = new Uint32Array(Z * plane), objs = [], R = opts.diamUm / 2, base = opts.idBase || 0;
    // candidates are points: each is drawn as an ellipsoid of the declared soma diameter (in um, so anisotropic in voxels)
    const rz = R / sp[0], ry = R / sp[1], rx = R / sp[2];
    cells.forEach(([cx, cy, cz], i) => { const id = base + i + 1;
      for (let z = Math.max(0, Math.floor(cz - rz)); z <= Math.min(Z - 1, Math.ceil(cz + rz)); z++) for (let y = Math.max(0, Math.floor(cy - ry)); y <= Math.min(H - 1, Math.ceil(cy + ry)); y++) for (let x = Math.max(0, Math.floor(cx - rx)); x <= Math.min(W - 1, Math.ceil(cx + rx)); x++) {
        const d = ((z - cz) / rz) ** 2 + ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2, q = (z * H + y) * W + x; if (d <= 1 && !lab[q]) lab[q] = id; }
      objs.push({ x: cx, y: cy, z: cz, rx, ry, rz, vox: (4 / 3) * Math.PI * rx * ry * rz, edgeZ: cz < 1 || cz > Z - 2, planes: Math.round(2 * rz) }); });
    return { objs, n: cells.length, lab, ms: performance.now() - t0 };
  }
  if (key === 'nd_vesselboost') return vesselboostTile(vol, dims, say, t0);
  throw new Error('unknown reference model ' + key);
}

// VesselBoost (packages/vesselboost/src/pipeline.js stepInference): z-score over the whole volume, 64^3 patches on a grid whose
// last patch is shifted inside the volume, logits -> sigmoid, Gaussian-weighted (sigma 8) accumulation, threshold 0.1, drop 3-D
// components under 10 voxels. A volume thinner than 64 voxels along an axis is edge-padded (the app pads by nearest neighbour).
async function vesselboostTile(vol, [Z, H, W], say, t0) {
  const c = REF.nd_vesselboost.cfg, P = c.patch, session = await getSession('nd_vesselboost', say), ort = await getOrt();
  const PZ = Math.max(Z, P), PH = Math.max(H, P), PW = Math.max(W, P), N = PZ * PH * PW, x = new Float32Array(N);
  let s = 0, s2 = 0;
  for (let z = 0; z < PZ; z++) for (let y = 0; y < PH; y++) for (let xx = 0; xx < PW; xx++) { const v = Math.expm1(vol[(Math.min(z, Z - 1) * H + Math.min(y, H - 1)) * W + Math.min(xx, W - 1)]); x[(z * PH + y) * PW + xx] = v; s += v; s2 += v * v; }
  const m = s / N, sd = Math.sqrt(Math.max(s2 / N - m * m, 1e-12)); for (let i = 0; i < N; i++) x[i] = (x[i] - m) / sd;
  const starts = (n) => { const o = []; for (let a = 0; a + P <= n; a += P) o.push(a); if (o[o.length - 1] + P < n) o.push(n - P); return o; };
  const gw = new Float32Array(P * P * P), h = (P - 1) / 2; for (let a = 0; a < P; a++) for (let b = 0; b < P; b++) for (let d = 0; d < P; d++) gw[(a * P + b) * P + d] = Math.exp(-((a - h) ** 2 + (b - h) ** 2 + (d - h) ** 2) / (2 * c.sigma * c.sigma));
  const acc = new Float32Array(N), wac = new Float32Array(N), patch = new Float32Array(P * P * P), pos = [];
  for (const z0 of starts(PZ)) for (const y0 of starts(PH)) for (const x0 of starts(PW)) pos.push([z0, y0, x0]);
  for (let k = 0; k < pos.length; k++) { const [z0, y0, x0] = pos[k];
    for (let a = 0; a < P; a++) for (let b = 0; b < P; b++) patch.set(x.subarray(((z0 + a) * PH + y0 + b) * PW + x0, ((z0 + a) * PH + y0 + b) * PW + x0 + P), (a * P + b) * P);
    const out = (await session.run({ input: new ort.Tensor('float32', patch, [1, 1, P, P, P]) })).output.data;
    for (let a = 0; a < P; a++) for (let b = 0; b < P; b++) for (let d = 0; d < P; d++) { const i = (a * P + b) * P + d, g = ((z0 + a) * PH + y0 + b) * PW + x0 + d, w = gw[i]; acc[g] += w / (1 + Math.exp(-out[i])); wac[g] += w; }
    say(`nd_vesselboost: patch ${k + 1}/${pos.length}`);
  }
  const cls = new Uint8Array(Z * H * W);
  for (let z = 0; z < Z; z++) for (let y = 0; y < H; y++) for (let xx = 0; xx < W; xx++) { const g = (z * PH + y) * PW + xx; if (wac[g] > 0 && acc[g] / wac[g] >= c.threshold) cls[(z * H + y) * W + xx] = 1; }
  // remove small 6-connected components
  const seen = new Uint8Array(cls.length), st = [], comp = [];
  for (let i = 0; i < cls.length; i++) if (cls[i] && !seen[i]) { comp.length = 0; st.push(i); seen[i] = 1;
    while (st.length) { const j = st.pop(); comp.push(j); const xx = j % W, y = ((j / W) | 0) % H, z = (j / (W * H)) | 0;
      for (const [dz, dy, dx] of [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]) { const zz = z + dz, yy = y + dy, x2 = xx + dx; if (zz < 0 || yy < 0 || x2 < 0 || zz >= Z || yy >= H || x2 >= W) continue; const q = (zz * H + yy) * W + x2; if (cls[q] && !seen[q]) { seen[q] = 1; st.push(q); } } }
    if (comp.length < c.minComponent) for (const j of comp) cls[j] = 0; }
  return { cls, patches: pos.length, ms: performance.now() - t0 };
}
