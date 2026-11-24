import { bloscDecode } from './blosc.js';
import * as fz from 'https://cdn.jsdelivr.net/npm/fzstd@0.1.1/esm/index.mjs';

export const S3 = 'https://dandiarchive.s3.amazonaws.com/zarr/';
export const clamp = (v, a, b) => Math.min(Math.max(v, a), b);

export class Limiter { constructor(n) { this.n = n; this.q = []; this.run = 0; }
  async go(fn) { if (this.run >= this.n) await new Promise((r) => this.q.push(r)); this.run++;
    try { return await fn(); } finally { this.run--; const r = this.q.shift(); if (r) r(); } } }

export class Zarr {
  constructor(id) { this.base = S3 + id; this.cache = new Map(); this.bytes = 0; this.net = 0; this.lim = new Limiter(8); this.inflight = new Map(); }
  async init() {
    const attrs = await (await fetch(this.base + '/.zattrs')).json();
    const ds = attrs.multiscales[0].datasets;
    this.levels = await Promise.all(ds.map(async (d) => {
      const za = await (await fetch(`${this.base}/${d.path}/.zarray`)).json();
      const sc = d.coordinateTransformations.find((t) => t.type === 'scale').scale.slice(-3);
      return { path: d.path, shape: za.shape.slice(-3), chunks: za.chunks.slice(-3), sp: sc };
    }));
    this.top = this.levels.length - 1;
  }
  chunk(l, cz, cy, cx) {
    const key = `${l}/${cz}/${cy}/${cx}`, hit = this.cache.get(key);
    if (hit) { this.cache.delete(key); this.cache.set(key, hit); return Promise.resolve(hit); }
    if (this.inflight.has(key)) return this.inflight.get(key);
    const L = this.levels[l], n = L.chunks[0] * L.chunks[1] * L.chunks[2];
    const p = this.lim.go(async () => {
      let arr;
      try {
        const r = await fetch(`${this.base}/${L.path}/0/0/${cz}/${cy}/${cx}`);
        if (!r.ok) arr = new Uint16Array(n);
        else {
          const buf = new Uint8Array(await r.arrayBuffer()); this.net += buf.length;
          const raw = bloscDecode(buf, (s, o) => fz.decompress(s, o));
          arr = new Uint16Array(raw.buffer, raw.byteOffset, raw.byteLength >> 1);
        }
      } catch (e) { arr = new Uint16Array(n); }
      this.cache.set(key, arr); this.bytes += arr.byteLength;
      for (const [k, v] of this.cache) { if (this.bytes < 700e6) break; this.cache.delete(k); this.bytes -= v.byteLength; }
      this.inflight.delete(key);
      return arr;
    });
    this.inflight.set(key, p);
    return p;
  }
  // log(I+1) crop as Float32Array [z0,z1)x[y0,y1)x[x0,x1)
  async crop(l, z0, z1, y0, y1, x0, x1) {
    const L = this.levels[l], [cz, cy, cx] = L.chunks, out = new Float32Array((z1 - z0) * (y1 - y0) * (x1 - x0));
    const jobs = [];
    for (let a = Math.floor(z0 / cz); a <= Math.floor((z1 - 1) / cz); a++)
      for (let b = Math.floor(y0 / cy); b <= Math.floor((y1 - 1) / cy); b++)
        for (let c = Math.floor(x0 / cx); c <= Math.floor((x1 - 1) / cx); c++)
          jobs.push(this.chunk(l, a, b, c).then((data) => ({ a, b, c, data })));
    for (const { a, b, c, data } of await Promise.all(jobs)) {
      const zs = Math.max(z0, a * cz), ze = Math.min(z1, (a + 1) * cz), ys = Math.max(y0, b * cy), ye = Math.min(y1, (b + 1) * cy),
        xs = Math.max(x0, c * cx), xe = Math.min(x1, (c + 1) * cx);
      for (let z = zs; z < ze; z++) for (let y = ys; y < ye; y++) {
        let src = ((z - a * cz) * cy + (y - b * cy)) * cx + (xs - c * cx), dst = ((z - z0) * (y1 - y0) + (y - y0)) * (x1 - x0) + (xs - x0);
        for (let x = xs; x < xe; x++) out[dst++] = Math.log(data[src++] + 1);
      }
    }
    return out;
  }
}

