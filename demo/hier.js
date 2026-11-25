// Hierarchical (SegFormer-style) hand-programmed transformer, format "hier1": parser, geometry, CPU reference, WebGPU engine.
// Semantics: docs/hierarchical/SPEC.md and src/hwseg/hier.py / hier_model.py (the Python reference). Input = raw log(I+1).
// Spatial ops (attn, conv) are separable: per-axis matrices are built on the host in float64 (identical formulas to Python)
// and applied as banded 1-D passes. rank = step-kernel global attention against a strided key set; gpool = global softmax head.

export const isHier = (js) => (typeof js === 'string' ? JSON.parse(js) : js).format === 'hier1';

export function parseHier(js) {
  if (typeof js === 'string') js = JSON.parse(js);
  const dense = (m) => { const a = new Float32Array(m.shape.reduce((p, q) => p * q, 1)); for (let j = 0; j < m.i.length; j++) a[m.i[j]] = m.v[j]; return a; };
  const ops = js.ops.map((o) => (o.op === 'mlp' ? { ...o, W1: dense(o.W1), b1: dense(o.b1), W2: dense(o.W2), h: o.b1.shape[0], din: o.W1.shape[1] } : o));
  return { hier: true, name: js.name, S: js.n_stages, d: js.d, dg: js.dg, out: js.out, meta: js.meta, zpool: js.zpool !== false, ops };
}

// ------------------------------------------------------------------ geometry (mirrors src/hwseg/hier.py)
export const fac = (s, a, zpool) => (a === 0 && !zpool ? 1 : 2 ** s);
export function stageShapes(n0, S, zpool) {
  const sh = [n0.slice()];
  for (let s = 1; s < S; s++) sh.push(sh[s - 1].map((v, a) => (a > 0 || zpool ? (v + 1) >> 1 : v)));
  return sh;
}
const stagePos = (n0a, sp0a, f, n) => Float64Array.from({ length: n }, (_, k) => (k * f + (f - 1) / 2 - (n0a - 1) / 2) * sp0a);

export function attnMatrix(n0a, sp0a, srcF, dstF, nSrc, nDst, sigma) {
  const ps = stagePos(n0a, sp0a, srcF, nSrc), pd = stagePos(n0a, sp0a, dstF, nDst), h = sp0a * srcF, sig = Math.max(sigma, 1e-6);
  const lim = Math.max(3 * sig, h) + 1e-9, M = new Float64Array(nDst * nSrc);
  for (let i = 0; i < nDst; i++) {
    let mx = -Infinity;
    for (let j = 0; j < nSrc; j++) { const D = pd[i] - ps[j]; if (Math.abs(D) <= lim) mx = Math.max(mx, -0.5 * (D / sig) ** 2); }
    let sum = 0;
    if (mx === -Infinity) { let bj = 0; for (let j = 1; j < nSrc; j++) if (Math.abs(pd[i] - ps[j]) < Math.abs(pd[i] - ps[bj])) bj = j; M[i * nSrc + bj] = 1; continue; }
    for (let j = 0; j < nSrc; j++) { const D = pd[i] - ps[j]; if (Math.abs(D) <= lim) { const w = Math.exp(-0.5 * (D / sig) ** 2 - mx); M[i * nSrc + j] = w; sum += w; } }
    for (let j = 0; j < nSrc; j++) M[i * nSrc + j] /= sum;
  }
  return M;
}
export function derivKernel(h, sigma, order) {
  const sig = Math.max(sigma, 0.5 * h), R = Math.max(1, Math.ceil(3 * sig / h)), n = 2 * R + 1;
  const t = Float64Array.from({ length: n }, (_, k) => (k - R) * h), g = t.map((v) => Math.exp(-0.5 * (v / sig) ** 2));
  const gs = g.reduce((a, b) => a + b, 0); for (let k = 0; k < n; k++) g[k] /= gs;
  if (order === 0) return g;
  if (order === 1) { const k1 = t.map((v, i) => v * g[i]); const s = k1.reduce((a, v, i) => a + v * t[i], 0); return k1.map((v) => v / s * sig); }
  const k2 = t.map((v, i) => (v * v / (sig * sig) - 1) * g[i]); const m = k2.reduce((a, b) => a + b, 0) / n;
  for (let i = 0; i < n; i++) k2[i] -= m;
  const s = k2.reduce((a, v, i) => a + v * t[i] * t[i] / 2, 0); return k2.map((v) => v / s * sig * sig);
}
export function convMatrix(n, h, sigma, order) {
  const k = derivKernel(h, sigma, order), R = (k.length - 1) >> 1, M = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < k.length; j++) M[i * n + Math.min(Math.max(i + j - R, 0), n - 1)] += k[j];
  return M;
}
const sigmaUm = (op, sRef, sp0, zpool) => {
  const sg = Array.isArray(op.sigma) ? op.sigma : [op.sigma, op.sigma, op.sigma];
  return op.unit === 'vox' ? sg.map((v, a) => v * sp0[a] * fac(sRef, a, zpool)) : sg.slice();
};
export const rankStride = (N, kmax) => Math.max(1, Math.ceil(N / kmax));

// banded form of a dense (nd x ns) matrix: per row [start, len] and the weights (row-major, width W)
function band(M, nd, ns) {
  const st = new Int32Array(nd), ln = new Int32Array(nd); let W = 1;
  for (let i = 0; i < nd; i++) { let a = -1, b = -1; for (let j = 0; j < ns; j++) if (M[i * ns + j] !== 0) { if (a < 0) a = j; b = j; } if (a < 0) { a = 0; b = 0; } st[i] = a; ln[i] = b - a + 1; W = Math.max(W, ln[i]); }
  const w = new Float32Array(nd * W);
  for (let i = 0; i < nd; i++) for (let k = 0; k < ln[i]; k++) w[i * W + k] = M[i * ns + st[i] + k];
  return { st, ln, w, W, nd, ns };
}

// Per-op host plan (matrices) for a given input shape / spacing: shared by the CPU and GPU executors.
export function planHier(prog, n0, sp0) {
  const shapes = stageShapes(n0, prog.S, prog.zpool), steps = [];
  for (const op of prog.ops) {
    if (op.op === 'attn') {
      const sg = sigmaUm(op, op.src_s, sp0, prog.zpool);
      const Ms = [0, 1, 2].map((a) => band(attnMatrix(n0[a], sp0[a], fac(op.src_s, a, prog.zpool), fac(op.dst_s, a, prog.zpool), shapes[op.src_s][a], shapes[op.dst_s][a], sg[a]), shapes[op.dst_s][a], shapes[op.src_s][a]));
      steps.push({ op, Ms, items: op.src.map((c, i) => ({ src: c, dst: op.dst[i], gain: op.gain[i] })), sIn: op.src_s, sOut: op.dst_s });
    } else if (op.op === 'conv') {
      const s = op.s, hs = [0, 1, 2].map((a) => sp0[a] * fac(s, a, prog.zpool));
      const items = op.kern.map((k) => { const sg = k.sigma * (op.unit === 'vox' ? hs[2] : 1);
        return { src: k.src, dst: k.dst, gain: k.gain, Ms: [0, 1, 2].map((a) => band(convMatrix(shapes[s][a], hs[a], sg, k.order[a]), shapes[s][a], shapes[s][a])) }; });
      steps.push({ op, items, sIn: s, sOut: s });
    } else steps.push({ op });
  }
  return { shapes, steps };
}

// ------------------------------------------------------------------ CPU reference
function applyBand(src, shIn, B, axis) {           // src (shIn) -> dst with shIn[axis] replaced by B.nd
  const sh = shIn.slice(); sh[axis] = B.nd; const [Z, Y, X] = sh, [Zi, Yi, Xi] = shIn, out = new Float32Array(Z * Y * X);
  for (let z = 0; z < Z; z++) for (let y = 0; y < Y; y++) for (let x = 0; x < X; x++) {
    const i = [z, y, x][axis], s0 = B.st[i], L = B.ln[i]; let acc = 0;
    for (let k = 0; k < L; k++) { const j = s0 + k, zz = axis === 0 ? j : z, yy = axis === 1 ? j : y, xx = axis === 2 ? j : x; acc += B.w[i * B.W + k] * src[(zz * Yi + yy) * Xi + xx]; }
    out[(z * Y + y) * X + x] = acc;
  }
  return out;
}

export function runHierCPU(prog, logI, n0, sp0) {
  const { shapes, steps } = planHier(prog, n0, sp0), S = prog.S, N = shapes.map((s) => s[0] * s[1] * s[2]);
  const x = shapes.map((_, s) => new Float32Array(prog.d[s] * N[s])), g = new Float64Array(prog.dg);
  x[0].set(logI.subarray ? logI.subarray(0, N[0]) : logI, 0); x[0].fill(1, N[0], 2 * N[0]);
  for (let s = 1; s < S; s++) x[s].fill(1, 0, N[s]);
  g[0] = Math.log2(sp0[2]); g[1] = 1;
  const ch = (s, c) => x[s].subarray(c * N[s], (c + 1) * N[s]);
  for (const st of steps) {
    const op = st.op;
    if (op.op === 'rank') {
      const v = ch(op.s, op.src), m = rankStride(N[op.s], op.kmax), mk = op.kmask != null ? ch(op.s, op.kmask) : null, keys = [];
      for (let j = 0; j < N[op.s]; j += m) if (!mk || mk[j] > 0.5) keys.push(v[j]);
      const K = Float32Array.from(keys).sort(), nk = K.length, dst = ch(op.s, op.dst);
      for (let i = 0; i < N[op.s]; i++) {
        const q = v[i]; let lo = 0, hi = nk; while (lo < hi) { const md = (lo + hi) >> 1; if (K[md] < q) lo = md + 1; else hi = md; }
        let lo2 = lo, hi2 = nk; while (lo2 < hi2) { const md = (lo2 + hi2) >> 1; if (K[md] <= q) lo2 = md + 1; else hi2 = md; }
        dst[i] += (lo + 0.5 * (lo2 - lo)) / Math.max(nk, 1);
      }
    } else if (op.op === 'gpool') {
      const n = N[op.s], sc = op.score != null && op.beta !== 0 ? ch(op.s, op.score) : null; let mx = -Infinity;
      if (sc) for (let i = 0; i < n; i++) mx = Math.max(mx, op.beta * sc[i]);
      const w = new Float64Array(n); let ws = 0;
      for (let i = 0; i < n; i++) { w[i] = sc ? Math.exp(op.beta * sc[i] - mx) : 1; ws += w[i]; }
      op.src.forEach((c, k) => { const v = ch(op.s, c); let a = 0; for (let i = 0; i < n; i++) a += w[i] * v[i]; g[op.dst[k]] += op.gain[k] * a / ws; });
    } else if (op.op === 'mlp') {
      const { W1, b1, W2, h, din } = op;
      if (op.s < 0) {
        const hid = new Float64Array(h);
        for (let u = 0; u < h; u++) { let p = b1[u]; for (let c = 0; c < din; c++) p += W1[u * din + c] * g[c]; hid[u] = p > 0 ? p : 0; }
        for (let c = 0; c < prog.dg; c++) { let a = 0; for (let u = 0; u < h; u++) a += W2[c * h + u] * hid[u]; g[c] += a; }
      } else {
        const s = op.s, ds = prog.d[s], n = N[s], X = x[s], dl = new Float32Array(ds * n);
        for (let u = 0; u < h; u++) {
          let gb = b1[u]; for (let c = 0; c < prog.dg; c++) gb += W1[u * din + ds + c] * g[c];
          const pre = new Float32Array(n).fill(gb);
          for (let c = 0; c < ds; c++) { const a = W1[u * din + c]; if (a !== 0) for (let i = 0; i < n; i++) pre[i] += a * X[c * n + i]; }
          for (let c = 0; c < ds; c++) { const b = W2[c * h + u]; if (b !== 0) for (let i = 0; i < n; i++) if (pre[i] > 0) dl[c * n + i] += b * pre[i]; }
        }
        for (let i = 0; i < ds * n; i++) X[i] += dl[i];
      }
    } else if (op.op === 'attn' || op.op === 'conv') {
      for (const it of st.items) {
        const Ms = st.Ms || it.Ms; let v = ch(st.sIn, it.src), sh = shapes[st.sIn].slice();
        for (let a = 0; a < 3; a++) { v = applyBand(v, sh, Ms[a], a); sh[a] = Ms[a].nd; }
        const d = ch(st.sOut, it.dst); for (let i = 0; i < d.length; i++) d[i] += it.gain * v[i];
      }
    }
  }
  const out = {};
  for (const [name, [s, c]] of Object.entries(prog.out)) out[name] = s < 0 ? g[c] : ch(s, c);
  return out;
}

// Stage-0 voxel half-extent per axis that one output depends on through local ops (global ops see the whole tile).
export function haloHier(prog, sp0) {
  const tot = [0, 0, 0];
  for (const op of prog.ops) {
    if (op.op === 'attn') { const sg = sigmaUm(op, op.src_s, sp0, prog.zpool); for (let a = 0; a < 3; a++) tot[a] += Math.max(3 * sg[a], sp0[a] * fac(op.src_s, a, prog.zpool)) / sp0[a]; }
    else if (op.op === 'conv') { const h = sp0[2] * 2 ** op.s; let r = 0;
      for (const k of op.kern) { const sg = k.sigma * (op.unit === 'vox' ? h : 1); r = Math.max(r, Math.max(1, Math.ceil(3 * Math.max(sg, 0.5 * h) / h)) * 2 ** op.s); }
      tot[0] += prog.zpool ? r : r / 2 ** op.s; tot[1] += r; tot[2] += r; }
  }
  return tot.map(Math.ceil);
}

// ------------------------------------------------------------------ WebGPU engine
const WGSL = /* wgsl */ `
struct P { a0:u32, a1:u32, a2:u32, a3:u32, a4:u32, a5:u32, a6:u32, a7:u32, a8:u32, a9:u32, a10:u32, a11:u32, f0:f32, f1:f32, f2:f32, f3:f32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> A: array<f32>;      // source stage stream (or tmp)
@group(0) @binding(2) var<storage, read_write> Bf: array<f32>;     // destination stage stream (or tmp)
@group(0) @binding(3) var<storage, read> W: array<f32>;            // weights (band weights / mlp)
@group(0) @binding(4) var<storage, read> I: array<i32>;            // band starts / lens
@group(0) @binding(5) var<storage, read_write> G: array<f32>;      // global token (dg) + scratch (reductions) at offset 4096

fn tid(g: vec3<u32>) -> u32 { return g.x + g.y * 4194240u; }

// axis pass: dst[(z,y,x)] (+)= gain * sum_k W[i*Wd+k] * src[.. st[i]+k ..] along axis a.
// a0 = axis, a1..a3 = src shape (Z,Y,X), a4..a6 = dst shape, a7 = src offset (floats), a8 = dst offset, a9 = band width, a10 = accumulate, f0 = gain
@compute @workgroup_size(64) fn axis(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = tid(gid); let Z = p.a4; let Y = p.a5; let X = p.a6; if (t >= Z * Y * X) { return; }
  let z = t / (Y * X); let y = (t % (Y * X)) / X; let x = t % X;
  var i = z; if (p.a0 == 1u) { i = y; } if (p.a0 == 2u) { i = x; }
  let s0 = u32(I[i]); let L = u32(I[p.a4 * 0u + i + select(select(Z, Y, p.a0 == 1u), X, p.a0 == 2u)]);
  var acc = 0.0;
  for (var k = 0u; k < L; k++) {
    let j = s0 + k; var zz = z; var yy = y; var xx = x;
    if (p.a0 == 0u) { zz = j; } if (p.a0 == 1u) { yy = j; } if (p.a0 == 2u) { xx = j; }
    acc += W[i * p.a9 + k] * A[p.a7 + (zz * p.a2 + yy) * p.a3 + xx];
  }
  let o = p.a8 + t;
  if (p.a10 == 1u) { Bf[o] = Bf[o] + p.f0 * acc; } else { Bf[o] = acc; }
}

// pointwise MLP on a stage: a0 = N, a1 = d, a2 = h, a3 = din, a4 = dg ; W = [W1 (h x din) | b1 (h) | W2 (d x h)]
@compute @workgroup_size(64) fn mlp(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = tid(gid); let N = p.a0; if (t >= N) { return; }
  let d = p.a1; let h = p.a2; let din = p.a3;
  var xv: array<f32, 128>; var dl: array<f32, 128>;
  for (var c = 0u; c < d; c++) { xv[c] = Bf[c * N + t]; dl[c] = 0.0; }
  for (var u = 0u; u < h; u++) {
    var pre = W[h * din + u];
    for (var c = 0u; c < p.a4; c++) { let a = W[u * din + d + c]; if (a != 0.0) { pre += a * G[c]; } }
    for (var c = 0u; c < d; c++) { let a = W[u * din + c]; if (a != 0.0) { pre += a * xv[c]; } }
    if (pre > 0.0) { for (var c = 0u; c < d; c++) { let b = W[h * din + h + c * h + u]; if (b != 0.0) { dl[c] += b * pre; } } }
  }
  for (var c = 0u; c < d; c++) { Bf[c * N + t] = xv[c] + dl[c]; }
}

// MLP on the global token (one thread): a2 = h, a3 = din (= dg)
@compute @workgroup_size(1) fn mlpg() {
  let h = p.a2; let din = p.a3;
  var hid: array<f32, 1024>;
  for (var u = 0u; u < h; u++) { var pre = W[h * din + u]; for (var c = 0u; c < din; c++) { pre += W[u * din + c] * G[c]; } hid[u] = max(pre, 0.0); }
  for (var c = 0u; c < din; c++) { var a = 0.0; for (var u = 0u; u < h; u++) { a += W[h * din + h + c * h + u] * hid[u]; } G[c] = G[c] + a; }
}

// rank: a0 = N, a1 = src offset, a2 = dst offset, a3 = stride m, a4 = mask offset, a5 = has mask
@compute @workgroup_size(64) fn rank(@builtin(global_invocation_id) gid: vec3<u32>) {
  let t = tid(gid); let N = p.a0; if (t >= N) { return; }
  let q = Bf[p.a1 + t]; var lt = 0.0; var eq = 0.0; var nk = 0.0;
  for (var j = 0u; j < N; j += p.a3) {
    if (p.a5 == 1u && Bf[p.a4 + j] <= 0.5) { continue; }
    let v = Bf[p.a1 + j]; nk += 1.0; if (v < q) { lt += 1.0; } else if (v == q) { eq += 1.0; }
  }
  Bf[p.a2 + t] = Bf[p.a2 + t] + (lt + 0.5 * eq) / max(nk, 1.0);
}

// global softmax pooling, 3 passes. a0 = N, a1 = score offset, a2 = value offset, a3 = has score, f0 = beta
var<workgroup> sh: array<f32, 256>;
var<workgroup> sh2: array<f32, 256>;
@compute @workgroup_size(256) fn gmax(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  var m = -3.0e38; var t = wg.x * 256u + l.x;
  loop { if (t >= p.a0) { break; } if (p.a3 == 1u) { m = max(m, p.f0 * Bf[p.a1 + t]); } else { m = 0.0; } t += 65536u; }
  sh[l.x] = m; workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (l.x < s) { sh[l.x] = max(sh[l.x], sh[l.x + s]); } workgroupBarrier(); }
  if (l.x == 0u) { G[4096u + wg.x] = sh[0]; }
}
@compute @workgroup_size(1) fn gmaxfin() { var m = -3.0e38; for (var i = 0u; i < 256u; i++) { m = max(m, G[4096u + i]); } G[4096u + 1024u] = m; }
@compute @workgroup_size(256) fn gsum(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  let mx = G[4096u + 1024u]; var a = 0.0; var b = 0.0; var t = wg.x * 256u + l.x;
  loop { if (t >= p.a0) { break; } var w = 1.0; if (p.a3 == 1u) { w = exp(p.f0 * Bf[p.a1 + t] - mx); } a += w; b += w * Bf[p.a2 + t]; t += 65536u; }
  sh[l.x] = a; sh2[l.x] = b; workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (l.x < s) { sh[l.x] += sh[l.x + s]; sh2[l.x] += sh2[l.x + s]; } workgroupBarrier(); }
  if (l.x == 0u) { G[4096u + 2048u + wg.x] = sh[0]; G[4096u + 2048u + 256u + wg.x] = sh2[0]; }
}
// a4 = destination global channel, f1 = gain
@compute @workgroup_size(1) fn gfin() {
  var a = 0.0; var b = 0.0; for (var i = 0u; i < 256u; i++) { a += G[4096u + 2048u + i]; b += G[4096u + 2048u + 256u + i]; }
  G[p.a4] = G[p.a4] + p.f1 * b / a;
}
`;

export class HierGPU {
  static async create(device = null) {
    if (!device) {
      if (!navigator.gpu) throw new Error('WebGPU not available');
      const ad = await navigator.gpu.requestAdapter(); if (!ad) throw new Error('No WebGPU adapter');
      device = await ad.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: ad.limits.maxStorageBufferBindingSize, maxBufferSize: ad.limits.maxBufferSize } });
    }
    return new HierGPU(device);
  }
  constructor(device) {
    this.dev = device;
    const S = GPUShaderStage.COMPUTE, e = (b, type) => ({ binding: b, visibility: S, buffer: { type } });
    this.bgl = device.createBindGroupLayout({ entries: [e(0, 'uniform'), e(1, 'storage'), e(2, 'storage'), e(3, 'read-only-storage'), e(4, 'read-only-storage'), e(5, 'storage')] });
    const module = device.createShaderModule({ code: WGSL }), layout = device.createPipelineLayout({ bindGroupLayouts: [this.bgl] });
    this.pl = {};
    for (const n of ['axis', 'mlp', 'mlpg', 'rank', 'gmax', 'gmaxfin', 'gsum', 'gfin']) this.pl[n] = device.createComputePipeline({ layout, compute: { module, entryPoint: n } });
  }
  maxTokens(d) { return Math.floor(this.dev.limits.maxStorageBufferBindingSize / 4 / Math.max(d, 1)); }

  // Run a whole tile; returns {name: Float32Array (stage tokens) | number (global)} for the requested outputs.
  async run(prog, logI, n0, sp0, opts = {}) {
    const dev = this.dev, U = GPUBufferUsage, { shapes, steps } = planHier(prog, n0, sp0), S = prog.S;
    if (Math.max(...prog.d) > 128) throw new Error('d > 128 per stage unsupported');
    const N = shapes.map((s) => s[0] * s[1] * s[2]), maxN = Math.max(...N);
    const bufs = [], tmp = [0, 1].map(() => dev.createBuffer({ size: 4 * Math.max(16, maxN), usage: U.STORAGE | U.COPY_SRC | U.COPY_DST }));
    for (let s = 0; s < S; s++) {
      const init = new Float32Array(prog.d[s] * N[s]);
      if (s === 0) { init.set(logI.subarray ? logI.subarray(0, N[0]) : logI, 0); init.fill(1, N[0], 2 * N[0]); } else init.fill(1, 0, N[s]);
      const b = dev.createBuffer({ size: 4 * Math.max(4, init.length), usage: U.STORAGE | U.COPY_SRC | U.COPY_DST }); dev.queue.writeBuffer(b, 0, init); bufs.push(b);
    }
    const gInit = new Float32Array(4096 + 4096); gInit[0] = Math.log2(sp0[2]); gInit[1] = 1;
    const gb = dev.createBuffer({ size: 4 * gInit.length, usage: U.STORAGE | U.COPY_SRC | U.COPY_DST }); dev.queue.writeBuffer(gb, 0, gInit);
    const dummyF = dev.createBuffer({ size: 16, usage: U.STORAGE }), dummyB = dev.createBuffer({ size: 16, usage: U.STORAGE }), dummyI = dev.createBuffer({ size: 16, usage: U.STORAGE }), dummyW = dev.createBuffer({ size: 16, usage: U.STORAGE }), trash = [dummyB, dummyW];
    const mk = (arr) => { const b = dev.createBuffer({ size: Math.max(16, arr.byteLength), usage: U.STORAGE | U.COPY_DST }); dev.queue.writeBuffer(b, 0, arr); trash.push(b); return b; };
    const enc = dev.createCommandEncoder();
    const go = (name, a, b, w, iv, u, f, nThreads) => {
      const ub = new ArrayBuffer(64), ui = new Uint32Array(ub), uf = new Float32Array(ub);
      ui.set(u.slice(0, 12)); uf.set((f || []).slice(0, 4), 12);
      const ubuf = dev.createBuffer({ size: 64, usage: U.UNIFORM | U.COPY_DST }); dev.queue.writeBuffer(ubuf, 0, ub); trash.push(ubuf);
      const bg = dev.createBindGroup({ layout: this.bgl, entries: [{ binding: 0, resource: { buffer: ubuf } }, { binding: 1, resource: { buffer: a || dummyF } },
        { binding: 2, resource: { buffer: b || dummyB } }, { binding: 3, resource: { buffer: w || dummyW } }, { binding: 4, resource: { buffer: iv || dummyI } }, { binding: 5, resource: { buffer: gb } }] });
      const pass = enc.beginComputePass(); pass.setPipeline(this.pl[name]); pass.setBindGroup(0, bg);
      if (name === 'gmax' || name === 'gsum') pass.dispatchWorkgroups(256);
      else if (nThreads === 1) pass.dispatchWorkgroups(1);
      else { const nwg = Math.ceil(nThreads / 64); pass.dispatchWorkgroups(Math.min(nwg, 65535), Math.ceil(nwg / 65535)); }
      pass.end();
    };
    for (const st of steps) {
      const op = st.op;
      if (op.op === 'rank') {
        const s = op.s, n = N[s];
        go('rank', null, bufs[s], null, null, [n, op.src * n, op.dst * n, rankStride(n, op.kmax), (op.kmask ?? 0) * n, op.kmask != null ? 1 : 0], [], n);
      } else if (op.op === 'gpool') {
        const s = op.s, n = N[s], has = op.score != null && op.beta !== 0 ? 1 : 0;
        go('gmax', null, bufs[s], null, null, [n, (op.score ?? 0) * n, 0, has], [op.beta], 0);
        go('gmaxfin', null, null, null, null, [], [], 1);
        op.src.forEach((c, k) => {
          go('gsum', null, bufs[s], null, null, [n, (op.score ?? 0) * n, c * n, has], [op.beta], 0);
          go('gfin', null, null, null, null, [0, 0, 0, 0, op.dst[k]], [0, op.gain[k]], 1);
        });
      } else if (op.op === 'mlp') {
        const wcat = new Float32Array(op.W1.length + op.b1.length + op.W2.length); wcat.set(op.W1, 0); wcat.set(op.b1, op.W1.length); wcat.set(op.W2, op.W1.length + op.b1.length);
        const wb = mk(wcat);
        if (op.s < 0) go('mlpg', null, null, wb, null, [0, 0, op.h, op.din], [], 1);
        else go('mlp', null, bufs[op.s], wb, null, [N[op.s], prog.d[op.s], op.h, op.din, prog.dg], [], N[op.s]);
      } else {   // attn / conv: three banded passes src -> tmp0 -> tmp1 -> dst (accumulate)
        for (const it of st.items) {
          const Ms = st.Ms || it.Ms; let sh = shapes[st.sIn].slice(), src = bufs[st.sIn], off = it.src * N[st.sIn];
          for (let a = 0; a < 3; a++) {
            const B = Ms[a], dsh = sh.slice(); dsh[a] = B.nd;
            const iv = new Int32Array(2 * B.nd); iv.set(B.st, 0); iv.set(B.ln, B.nd);
            const last = a === 2, dst = last ? bufs[st.sOut] : tmp[a & 1], doff = last ? it.dst * N[st.sOut] : 0;
            go('axis', src, dst, mk(B.w), mk(iv), [a, sh[0], sh[1], sh[2], dsh[0], dsh[1], dsh[2], off, doff, B.W, last ? 1 : 0], [it.gain], dsh[0] * dsh[1] * dsh[2]);
            src = dst; off = 0; sh = dsh;
          }
        }
      }
    }
    const names = opts.names || Object.keys(prog.out), out = {}, reads = [];
    for (const nm of names) {
      const [s, c] = prog.out[nm], n = s < 0 ? 1 : (opts.planes ? opts.planes[1] - opts.planes[0] : shapes[s][0]) * shapes[s][1] * shapes[s][2];
      const off = s < 0 ? c : c * N[s] + (opts.planes && s >= 0 ? opts.planes[0] * shapes[s][1] * shapes[s][2] : 0);
      const stg = dev.createBuffer({ size: 4 * Math.max(4, n), usage: U.MAP_READ | U.COPY_DST });
      enc.copyBufferToBuffer(s < 0 ? gb : bufs[s], 4 * off, stg, 0, 4 * n); reads.push([nm, stg, n, s]);
    }
    dev.queue.submit([enc.finish()]);
    for (const [nm, stg, n, s] of reads) { await stg.mapAsync(GPUMapMode.READ); const a = new Float32Array(stg.getMappedRange().slice(0)); stg.unmap(); stg.destroy(); out[nm] = s < 0 ? a[0] : a; }
    [...bufs, ...tmp, gb, dummyF, dummyI, ...trash].forEach((b) => b.destroy());
    return out;
  }
}
