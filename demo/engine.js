// Weight-space-programmed 3D transformer: program loader, execution planner, CPU reference, WebGPU engine.
// Token = voxel. Channels 0..10 are the fixed embedding (logI, z,y,x, 1, z2,y2,x2, zy,zx,yx).
// Heads: Q=Wq x, K=Wk x (4 dims), V=Wv.x (scalar), local softmax attention in a physical-radius window,
// output Wo * attn added to the residual. MLP: x += W2 relu(W1 x + b1). Nothing is trained.

export function parseProgram(js) {
  if (typeof js === 'string') js = JSON.parse(js);
  const dense = (m) => {
    const a = new Float32Array(m.shape.reduce((p, q) => p * q, 1));
    for (let j = 0; j < m.i.length; j++) a[m.i[j]] = m.v[j];
    return a;
  };
  return {
    name: js.name, d: js.d, out: js.out, meta: js.meta,
    layers: js.layers.map((L) => ({
      heads: L.heads.map((h) => ({ Wq: dense(h.Wq), Wk: dense(h.Wk), Wv: dense(h.Wv), Wo: dense(h.Wo), radius_um: h.radius_um })),
      mlp: L.mlp ? { W1: dense(L.mlp.W1), b1: dense(L.mlp.b1), W2: dense(L.mlp.W2), h: L.mlp.b1.shape[0] } : null,
    })),
  };
}

export function nParams(prog) {
  let nz = 0, tot = 0;
  const cnt = (a) => { tot += a.length; for (const v of a) if (v !== 0) nz++; };
  for (const L of prog.layers) {
    for (const h of L.heads) [h.Wq, h.Wk, h.Wv, h.Wo].forEach(cnt);
    if (L.mlp) [L.mlp.W1, L.mlp.b1, L.mlp.W2].forEach(cnt);
  }
  return { total: tot, nonzero: nz };
}

const radiusVox = (h, sp) => h.radius_um ? h.radius_um.map((r, i) => Math.max(1, Math.ceil(r / sp[i]))) : null;

// Geometry needed to segment ONE output plane: in-plane halo (sum of radii), z half-thickness, and, for a given
// tile (dims) and plane index zc, the output z-range each layer must produce (computed backwards).
export function haloFor(prog, sp) {
  let hz = 0, hy = 0, hx = 0;
  for (const L of prog.layers) {
    let mz = 0, my = 0, mx = 0;
    for (const h of L.heads) {
      const r = radiusVox(h, sp);
      if (r) { mz = Math.max(mz, r[0]); my = Math.max(my, r[1]); mx = Math.max(mx, r[2]); }
    }
    hz += mz; hy += my; hx += mx;
  }
  return [hz, hy, hx];
}

export function planRanges(prog, sp, Z, zc) {
  const n = prog.layers.length, ranges = new Array(n), inRanges = new Array(n);
  let lo = zc, hi = zc;
  for (let l = n - 1; l >= 0; l--) {
    ranges[l] = [lo, hi];
    const rv = prog.layers[l].heads.map((h) => radiusVox(h, sp));
    const global = rv.some((r) => r === null);
    const rz = Math.max(0, ...rv.filter(Boolean).map((r) => r[0]));
    if (global) { lo = 0; hi = Z - 1; } else { lo = Math.max(0, lo - rz); hi = Math.min(Z - 1, hi + rz); }
    inRanges[l] = [lo, hi];
  }
  return { ranges, inRanges };
}

// ------------------------------------------------------------------ CPU reference (all tokens, dense attention)
export function runCPU(prog, logI, dims, sp) {
  const [Z, Y, X] = dims, N = Z * Y * X, d = prog.d;
  const x = new Float32Array(d * N);
  let t = 0;
  for (let z = 0; z < Z; z++) for (let y = 0; y < Y; y++) for (let xx = 0; xx < X; xx++, t++) {
    const pz = (z - (Z - 1) / 2) * sp[0], py = (y - (Y - 1) / 2) * sp[1], px = (xx - (X - 1) / 2) * sp[2];
    const v = [logI[t], pz, py, px, 1, pz * pz, py * py, px * px, pz * py, pz * px, py * px];
    for (let c = 0; c < 11; c++) x[c * N + t] = v[c];
  }
  for (const L of prog.layers) {
    const delta = new Float32Array(d * N);
    for (const h of L.heads) {
      const v = new Float32Array(N);
      for (let c = 0; c < d; c++) if (h.Wv[c] !== 0) for (let i = 0; i < N; i++) v[i] += h.Wv[c] * x[c * N + i];
      const o = new Float32Array(N);
      if (!h.radius_um) {
        let m = 0; for (let i = 0; i < N; i++) m += v[i]; m /= N; o.fill(m);
      } else {
        const q = new Float32Array(4 * N), k = new Float32Array(4 * N);
        for (let a = 0; a < 4; a++) for (let c = 0; c < d; c++) {
          const wq = h.Wq[a * d + c], wk = h.Wk[a * d + c];
          if (wq !== 0) for (let i = 0; i < N; i++) q[a * N + i] += wq * x[c * N + i];
          if (wk !== 0) for (let i = 0; i < N; i++) k[a * N + i] += wk * x[c * N + i];
        }
        const r = radiusVox(h, sp);
        for (let z = 0; z < Z; z++) for (let y = 0; y < Y; y++) for (let xx = 0; xx < X; xx++) {
          const i = (z * Y + y) * X + xx;
          let m = -Infinity, s = 0, acc = 0;
          for (let dz = -r[0]; dz <= r[0]; dz++) { const zz = z + dz; if (zz < 0 || zz >= Z) continue;
            for (let dy = -r[1]; dy <= r[1]; dy++) { const yy = y + dy; if (yy < 0 || yy >= Y) continue;
              for (let dx = -r[2]; dx <= r[2]; dx++) { const xq = xx + dx; if (xq < 0 || xq >= X) continue;
                const j = (zz * Y + yy) * X + xq;
                const sc = q[i] * k[j] + q[N + i] * k[N + j] + q[2 * N + i] * k[2 * N + j] + q[3 * N + i] * k[3 * N + j];
                if (sc > m) { const e = Math.exp(m - sc); s *= e; acc *= e; m = sc; }
                const w = Math.exp(sc - m); s += w; acc += w * v[j];
              } } }
          o[i] = acc / s;
        }
      }
      for (let c = 0; c < d; c++) if (h.Wo[c] !== 0) for (let i = 0; i < N; i++) delta[c * N + i] += h.Wo[c] * o[i];
    }
    for (let i = 0; i < d * N; i++) x[i] += delta[i];
    if (L.mlp) {
      const { W1, b1, W2, h } = L.mlp;
      const dl = new Float32Array(d * N);
      for (let u = 0; u < h; u++) {
        const pre = new Float32Array(N).fill(b1[u]);
        for (let c = 0; c < d; c++) { const a = W1[u * d + c]; if (a !== 0) for (let i = 0; i < N; i++) pre[i] += a * x[c * N + i]; }
        for (let c = 0; c < d; c++) { const b = W2[c * h + u]; if (b !== 0) for (let i = 0; i < N; i++) if (pre[i] > 0) dl[c * N + i] += b * pre[i]; }
      }
      for (let i = 0; i < d * N; i++) x[i] += dl[i];
    }
  }
  const out = {};
  for (const [name, c] of Object.entries(prog.out)) out[name] = x.subarray(c * N, (c + 1) * N);
  return out;
}

// ------------------------------------------------------------------ WebGPU engine
const WGSL = /* wgsl */ `
struct P { N:u32, Z:u32, Y:u32, X:u32, d:u32, rz:u32, ry:u32, rx:u32, olo:u32, ohi:u32, ilo:u32, ihi:u32, h:u32, nglob:u32, spz:f32, spy:f32, spx:f32, _p0:u32, _p1:u32, _p2:u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read_write> q: array<f32>;
@group(0) @binding(4) var<storage, read_write> k: array<f32>;
@group(0) @binding(5) var<storage, read_write> v: array<f32>;
@group(0) @binding(6) var<storage, read> logi: array<f32>;
@group(0) @binding(7) var<storage, read_write> part: array<f32>;

fn tok(g: vec3<u32>) -> u32 { return g.x + g.y * 4194240u; }

@compute @workgroup_size(64) fn embed(@builtin(global_invocation_id) g: vec3<u32>) {
  let t = tok(g); if (t >= p.N) { return; }
  let plane = p.Y * p.X;
  let z = t / plane; let y = (t % plane) / p.X; let xx = t % p.X;
  let pz = (f32(z) - f32(p.Z - 1u) * 0.5) * p.spz;
  let py = (f32(y) - f32(p.Y - 1u) * 0.5) * p.spy;
  let px = (f32(xx) - f32(p.X - 1u) * 0.5) * p.spx;
  let N = p.N;
  x[t] = logi[t]; x[N + t] = pz; x[2u*N + t] = py; x[3u*N + t] = px; x[4u*N + t] = 1.0;
  x[5u*N + t] = pz*pz; x[6u*N + t] = py*py; x[7u*N + t] = px*px;
  x[8u*N + t] = pz*py; x[9u*N + t] = pz*px; x[10u*N + t] = py*px;
}

@compute @workgroup_size(64) fn qkv(@builtin(global_invocation_id) g: vec3<u32>) {
  let t = tok(g); if (t >= p.N) { return; }
  let z = t / (p.Y * p.X); if (z < p.ilo || z > p.ihi) { return; }
  let d = p.d; let N = p.N;
  var qa = vec4<f32>(0.0); var ka = vec4<f32>(0.0); var vv = 0.0;
  for (var c = 0u; c < d; c++) {
    let xc = x[c * N + t];
    qa += vec4<f32>(w[c], w[d + c], w[2u*d + c], w[3u*d + c]) * xc;
    ka += vec4<f32>(w[4u*d + c], w[5u*d + c], w[6u*d + c], w[7u*d + c]) * xc;
    vv += w[8u*d + c] * xc;
  }
  q[t] = qa.x; q[N + t] = qa.y; q[2u*N + t] = qa.z; q[3u*N + t] = qa.w;
  k[t] = ka.x; k[N + t] = ka.y; k[2u*N + t] = ka.z; k[3u*N + t] = ka.w;
  v[t] = vv;
}

@compute @workgroup_size(64) fn attn(@builtin(global_invocation_id) g: vec3<u32>) {
  let t = tok(g); if (t >= p.N) { return; }
  let plane = p.Y * p.X;
  let z = i32(t / plane); let y = i32((t % plane) / p.X); let xx = i32(t % p.X);
  if (u32(z) < p.olo || u32(z) > p.ohi) { return; }
  let N = p.N; let d = p.d;
  let q0 = q[t]; let q1 = q[N + t]; let q2 = q[2u*N + t]; let q3 = q[3u*N + t];
  var m = -3.0e38; var s = 0.0; var acc = 0.0;
  let rz = i32(p.rz); let ry = i32(p.ry); let rx = i32(p.rx);
  for (var dz = -rz; dz <= rz; dz++) {
    let zz = z + dz; if (zz < 0 || zz >= i32(p.Z)) { continue; }
    for (var dy = -ry; dy <= ry; dy++) {
      let yy = y + dy; if (yy < 0 || yy >= i32(p.Y)) { continue; }
      for (var dx = -rx; dx <= rx; dx++) {
        let xq = xx + dx; if (xq < 0 || xq >= i32(p.X)) { continue; }
        let j = (u32(zz) * p.Y + u32(yy)) * p.X + u32(xq);
        let sc = q0 * k[j] + q1 * k[N + j] + q2 * k[2u*N + j] + q3 * k[3u*N + j];
        if (sc > m) { let e = exp(m - sc); s *= e; acc *= e; m = sc; }
        let wt = exp(sc - m); s += wt; acc += wt * v[j];
      }
    }
  }
  let res = acc / s;
  for (var c = 0u; c < d; c++) { let wo = w[9u*d + c]; if (wo != 0.0) { x[c * N + t] += wo * res; } }
}

var<workgroup> sh: array<f32, 256>;
@compute @workgroup_size(256) fn gsum(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
  var a = 0.0;
  var t = wg.x * 256u + l.x;
  let stride = 256u * 256u;
  loop { if (t >= p.N) { break; } a += v[t]; t += stride; }
  sh[l.x] = a; workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (l.x < s) { sh[l.x] += sh[l.x + s]; } workgroupBarrier(); }
  if (l.x == 0u) { part[wg.x] = sh[0]; }
}
@compute @workgroup_size(1) fn gfin() {
  var a = 0.0; for (var i = 0u; i < 256u; i++) { a += part[i]; }
  part[256] = a / f32(p.N);
}
@compute @workgroup_size(64) fn gapply(@builtin(global_invocation_id) g: vec3<u32>) {
  let t = tok(g); if (t >= p.N) { return; }
  let z = t / (p.Y * p.X); if (z < p.olo || z > p.ohi) { return; }
  let d = p.d; let N = p.N; let mean = part[256];
  for (var c = 0u; c < d; c++) { let wo = w[9u*d + c]; if (wo != 0.0) { x[c * N + t] += wo * mean; } }
}

@compute @workgroup_size(64) fn mlp(@builtin(global_invocation_id) g: vec3<u32>) {
  let t = tok(g); if (t >= p.N) { return; }
  let z = t / (p.Y * p.X); if (z < p.olo || z > p.ohi) { return; }
  let d = p.d; let N = p.N; let h = p.h;
  var xv: array<f32, 64>; var dl: array<f32, 64>;
  for (var c = 0u; c < d; c++) { xv[c] = x[c * N + t]; dl[c] = 0.0; }
  for (var i = 0u; i < h; i++) {
    var pre = w[h * d + i];
    for (var c = 0u; c < d; c++) { let a = w[i * d + c]; if (a != 0.0) { pre += a * xv[c]; } }
    if (pre > 0.0) {
      for (var c = 0u; c < d; c++) { let b = w[h * d + h + c * h + i]; if (b != 0.0) { dl[c] += b * pre; } }
    }
  }
  for (var c = 0u; c < d; c++) { x[c * N + t] = xv[c] + dl[c]; }
}
`;

export class GPUEngine {
  static async create() {
    if (!navigator.gpu) throw new Error('WebGPU not available');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('No WebGPU adapter');
    const limits = {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    };
    const device = await adapter.requestDevice({ requiredLimits: limits });
    return new GPUEngine(device, limits);
  }

  constructor(device, limits) {
    this.device = device; this.limits = limits; this.cache = {};
    const S = GPUShaderStage.COMPUTE;
    const e = (b, type) => ({ binding: b, visibility: S, buffer: { type } });
    this.bgl = device.createBindGroupLayout({ entries: [
      e(0, 'uniform'), e(1, 'storage'), e(2, 'read-only-storage'), e(3, 'storage'), e(4, 'storage'), e(5, 'storage'),
      e(6, 'read-only-storage'), e(7, 'storage')] });
    const module = device.createShaderModule({ code: WGSL });
    const layout = device.createPipelineLayout({ bindGroupLayouts: [this.bgl] });
    this.pl = {};
    for (const n of ['embed', 'qkv', 'attn', 'gsum', 'gfin', 'gapply', 'mlp'])
      this.pl[n] = device.createComputePipeline({ layout, compute: { module, entryPoint: n } });
    this.dummy = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE });
  }

  maxTokens(d) { return Math.floor(Math.min(this.limits.maxStorageBufferBindingSize, this.limits.maxBufferSize) / 4 / d); }

  _buf(data, usage) {
    const b = this.device.createBuffer({ size: Math.max(16, data.byteLength), usage: usage | GPUBufferUsage.COPY_DST, mappedAtCreation: false });
    this.device.queue.writeBuffer(b, 0, data);
    return b;
  }

  _weights(prog) {
    if (prog._gpu) return prog._gpu;
    const d = prog.d, g = prog.layers.map((L) => ({
      heads: L.heads.map((h) => {
        const a = new Float32Array(10 * d);
        a.set(h.Wq, 0); a.set(h.Wk, 4 * d); a.set(h.Wv, 8 * d); a.set(h.Wo, 9 * d);
        return this._buf(a, GPUBufferUsage.STORAGE);
      }),
      mlp: L.mlp ? (() => { const m = L.mlp, a = new Float32Array(m.h * d + m.h + d * m.h);
        a.set(m.W1, 0); a.set(m.b1, m.h * d); a.set(m.W2, m.h * d + m.h); return this._buf(a, GPUBufferUsage.STORAGE); })() : null,
    }));
    prog._gpu = g;
    return g;
  }

  // Run a tile; returns {name: Float32Array(Y*X)} for plane zc of every named output.
  async run(prog, logI, dims, sp, zc) {
    const dev = this.device, [Z, Y, X] = dims, N = Z * Y * X, d = prog.d;
    if (d > 64) throw new Error('d>64 unsupported');
    if (N > this.maxTokens(d)) throw new Error('tile too large for device buffer limits');
    const W = this._weights(prog), { ranges, inRanges } = planRanges(prog, sp, Z, zc);
    const U = GPUBufferUsage;
    const bx = dev.createBuffer({ size: 4 * d * N, usage: U.STORAGE | U.COPY_SRC | U.COPY_DST });
    const bq = dev.createBuffer({ size: 16 * N, usage: U.STORAGE });
    const bk = dev.createBuffer({ size: 16 * N, usage: U.STORAGE });
    const bv = dev.createBuffer({ size: 4 * N, usage: U.STORAGE });
    const bl = this._buf(logI, U.STORAGE);
    const bp = dev.createBuffer({ size: 4 * 257, usage: U.STORAGE });
    const enc = dev.createCommandEncoder();
    const uniforms = [];
    const dispatch = (name, wbuf, o) => {
      const ub = new ArrayBuffer(80), u = new Uint32Array(ub), f = new Float32Array(ub);
      u.set([N, Z, Y, X, d, o.rz || 0, o.ry || 0, o.rx || 0, o.olo, o.ohi, o.ilo, o.ihi, o.h || 0, 0]);
      f[14] = sp[0]; f[15] = sp[1]; f[16] = sp[2];
      const ubuf = dev.createBuffer({ size: 80, usage: U.UNIFORM | U.COPY_DST });
      dev.queue.writeBuffer(ubuf, 0, ub); uniforms.push(ubuf);
      const bg = dev.createBindGroup({ layout: this.bgl, entries: [
        { binding: 0, resource: { buffer: ubuf } }, { binding: 1, resource: { buffer: bx } },
        { binding: 2, resource: { buffer: wbuf || this.dummy } }, { binding: 3, resource: { buffer: bq } },
        { binding: 4, resource: { buffer: bk } }, { binding: 5, resource: { buffer: bv } },
        { binding: 6, resource: { buffer: bl } }, { binding: 7, resource: { buffer: bp } }] });
      const pass = enc.beginComputePass();
      pass.setPipeline(this.pl[name]); pass.setBindGroup(0, bg);
      if (name === 'gsum') pass.dispatchWorkgroups(256);
      else if (name === 'gfin') pass.dispatchWorkgroups(1);
      else { const nwg = Math.ceil(N / 64); pass.dispatchWorkgroups(Math.min(nwg, 65535), Math.ceil(nwg / 65535)); }
      pass.end();
    };
    dispatch('embed', null, { olo: 0, ohi: Z - 1, ilo: 0, ihi: Z - 1 });
    prog.layers.forEach((L, li) => {
      const [olo, ohi] = ranges[li], [ilo, ihi] = inRanges[li];
      L.heads.forEach((h, hi) => {
        const r = radiusVox(h, sp);
        if (r) {
          dispatch('qkv', W[li].heads[hi], { ilo, ihi, olo, ohi });
          dispatch('attn', W[li].heads[hi], { rz: r[0], ry: r[1], rx: r[2], olo, ohi, ilo, ihi });
        } else {
          dispatch('qkv', W[li].heads[hi], { ilo: 0, ihi: Z - 1, olo, ohi });
          dispatch('gsum', W[li].heads[hi], { olo, ohi, ilo: 0, ihi: Z - 1 });
          dispatch('gfin', W[li].heads[hi], { olo, ohi, ilo: 0, ihi: Z - 1 });
          dispatch('gapply', W[li].heads[hi], { olo, ohi, ilo: 0, ihi: Z - 1 });
        }
      });
      if (L.mlp) dispatch('mlp', W[li].mlp, { h: L.mlp.h, olo, ohi, ilo, ihi });
    });
    const names = Object.keys(prog.out), plane = Y * X;
    const stage = dev.createBuffer({ size: 4 * plane * names.length, usage: U.MAP_READ | U.COPY_DST });
    names.forEach((n, i) => enc.copyBufferToBuffer(bx, 4 * (prog.out[n] * N + zc * plane), stage, 4 * plane * i, 4 * plane));
    dev.queue.submit([enc.finish()]);
    await stage.mapAsync(GPUMapMode.READ);
    const all = new Float32Array(stage.getMappedRange().slice(0));
    stage.unmap();
    [bx, bq, bk, bv, bl, bp, stage, ...uniforms].forEach((b) => b.destroy());
    const out = {};
    names.forEach((n, i) => { out[n] = all.subarray(i * plane, (i + 1) * plane); });
    return out;
  }
}
