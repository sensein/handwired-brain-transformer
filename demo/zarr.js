import { bloscDecode } from './blosc.js';
import * as fz from 'https://cdn.jsdelivr.net/npm/fzstd@0.1.1/esm/index.mjs';

export const S3 = 'https://dandiarchive.s3.amazonaws.com/zarr/';
export const clamp = (v, a, b) => Math.min(Math.max(v, a), b);

export class Limiter { constructor(n) { this.n = n; this.q = []; this.run = 0; }
  async go(fn) { if (this.run >= this.n) await new Promise((r) => this.q.push(r)); this.run++;
    try { return await fn(); } finally { this.run--; const r = this.q.shift(); if (r) r(); } } }

export class Zarr {
  constructor(id) { this.stat = { inflight: 0, chunks: 0, bytes: 0 }; this.base = S3 + id; this.cache = new Map(); this.bytes = 0; this.net = 0; this.lim = new Limiter(8); this.inflight = new Map(); }
  async init() {
    const attrs = await (await fetch(this.base + '/.zattrs')).json();
    const ds = attrs.multiscales[0].datasets;
    this.levels = await Promise.all(ds.map(async (d) => {
      const za = await (await fetch(`${this.base}/${d.path}/.zarray`)).json();
      const sc = d.coordinateTransformations.find((t) => t.type === 'scale').scale.slice(-3);
      const tr = (d.coordinateTransformations.find((t) => t.type === 'translation') || { translation: [0, 0, 0, 0, 0] }).translation.slice(-3);
      return { path: d.path, shape: za.shape.slice(-3), chunks: za.chunks.slice(-3), sp: sc, tr };
    }));
    this.top = this.levels.length - 1;
  }
  chunk(l, cz, cy, cx) {
    const key = `${l}/${cz}/${cy}/${cx}`, hit = this.cache.get(key);
    if (hit) { this.cache.delete(key); this.cache.set(key, hit); return Promise.resolve(hit); }
    if (this.inflight.has(key)) return this.inflight.get(key);
    const L = this.levels[l], n = L.chunks[0] * L.chunks[1] * L.chunks[2];
    this.stat.inflight++;
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
      this.inflight.delete(key); this.stat.inflight--; this.stat.chunks++; this.stat.bytes = this.net;
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



// A slab = several strips (chunks) stacked along y with ~0.8 mm overlap; each zarr stores its physical translation (um).
// Slab places every strip by that translation (what Neuroglancer does with a multi-source layer), blends overlaps with a
// feathered average along y, and exposes the same interface as Zarr (levels, top, crop, stat, net, bytes, base).
export class Slab {
  constructor(ids) { this.members = ids.map((id) => new Zarr(id)); this.base = `slab:${ids[0]}+${ids.length}`; this.ids = ids; }
  get stat() { const t = { inflight: 0, chunks: 0, bytes: 0 }; for (const m of this.members) { t.inflight += m.stat.inflight; t.chunks += m.stat.chunks; t.bytes += m.stat.bytes; } return t; }
  get net() { return this.members.reduce((a, m) => a + m.net, 0); }
  get bytes() { return this.members.reduce((a, m) => a + m.bytes, 0); }
  async init() {
    await Promise.all(this.members.map((m) => m.init()));
    // A strip that covers most of an earlier strip's y-range is a re-acquisition, not a neighbour (neighbours overlap ~10%):
    // averaging two scans with different illumination makes a band, so keep the first and list the rest as dropped.
    const top0 = this.members[0].levels[this.members[0].levels.length - 1], yr = (m) => { const L = m.levels[m.levels.length - 1]; return [L.tr[1], L.tr[1] + L.shape[1] * L.sp[1]]; };
    const keep = [], dropped = [];
    this.members.forEach((m, i) => {
      const [a0, a1] = yr(m), dup = keep.find((k) => { const [b0, b1] = yr(this.members[k]); return Math.min(a1, b1) - Math.max(a0, b0) > 0.5 * Math.min(a1 - a0, b1 - b0); });
      (dup === undefined ? keep : dropped).push(i);
    });
    this.dropped = dropped.map((i) => this.ids[i]); this.members = keep.map((i) => this.members[i]); this.ids = keep.map((i) => this.ids[i]); void top0;
    const nl = Math.min(...this.members.map((m) => m.levels.length)); this.top = nl - 1; this.levels = [];
    for (let l = 0; l < nl; l++) {
      const sp = this.members[0].levels[l].sp, org = [0, 1, 2].map((a) => Math.min(...this.members.map((m) => m.levels[l].tr[a])));
      const end = [0, 1, 2].map((a) => Math.max(...this.members.map((m) => m.levels[l].tr[a] + m.levels[l].shape[a] * sp[a])));
      const offs = this.members.map((m) => [0, 1, 2].map((a) => Math.round((m.levels[l].tr[a] - org[a]) / sp[a])));
      this.levels.push({ path: 'slab', shape: [0, 1, 2].map((a) => Math.ceil((end[a] - org[a]) / sp[a])), chunks: [1, 256, 256], sp, origin: org, offs });
    }
  }
  async crop(level, z0, z1, y0, y1, x0, x1) {
    const L = this.levels[level], n = (z1 - z0) * (y1 - y0) * (x1 - x0), out = new Float32Array(n), wsum = new Float32Array(n);
    const feather = Math.max(2, Math.round(780 / L.sp[1])), jobs = [];
    this.members.forEach((m, i) => {
      const o = L.offs[i], sh = m.levels[level].shape;
      const b = [Math.max(0, z0 - o[0]), Math.min(sh[0], z1 - o[0]), Math.max(0, y0 - o[1]), Math.min(sh[1], y1 - o[1]), Math.max(0, x0 - o[2]), Math.min(sh[2], x1 - o[2])];
      if (b[1] <= b[0] || b[3] <= b[2] || b[5] <= b[4]) return;
      jobs.push(m.crop(level, b[0], b[1], b[2], b[3], b[4], b[5]).then((sub) => ({ sub, b, o, sh })));
    });
    for (const { sub, b, o, sh } of await Promise.all(jobs)) {
      const dz = b[1] - b[0], dy = b[3] - b[2], dx = b[5] - b[4];
      for (let z = 0; z < dz; z++) for (let y = 0; y < dy; y++) {
        const ly = b[2] + y, w = Math.max(1e-3, Math.min(1, Math.min(ly + 1, sh[1] - ly) / feather));
        let src = (z * dy + y) * dx, dst = ((b[0] + o[0] + z - z0) * (y1 - y0) + (ly + o[1] - y0)) * (x1 - x0) + (b[4] + o[2] - x0);
        for (let x = 0; x < dx; x++, src++, dst++) { out[dst] += w * sub[src]; wsum[dst] += w; }
      }
    }
    for (let i = 0; i < n; i++) out[i] = wsum[i] > 0 ? out[i] / wsum[i] : 0;
    return out;
  }
}
