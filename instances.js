// Instance building shared by the hand-written programs and the Neurodesk reference models (demo/nd/nd-seg.js):
// per-plane 2-D components and their linking across z into 3-D objects.

// ---- 2-D connected components (4-neighbour) of a thresholded plane; pixel indices are local to the w x h region.
export function components(seg, dims, ox, oy, w, h) {
  const mask = new Uint8Array(w * h), seen = new Uint8Array(w * h), out = [];
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) mask[y * w + x] = seg[(oy + y) * dims[2] + ox + x] > 0.5 ? 1 : 0;
  for (let i = 0; i < w * h; i++) if (mask[i] && !seen[i]) {
    let n = 0, sx = 0, sy = 0; const st = [i], pix = []; seen[i] = 1;
    while (st.length) { const j = st.pop(), x = j % w, y = (j / w) | 0; n++; sx += x; sy += y; pix.push(j);
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) { const xx = x + dx, yy = y + dy; if (xx < 0 || yy < 0 || xx >= w || yy >= h) continue; const k = yy * w + xx; if (mask[k] && !seen[k]) { seen[k] = 1; st.push(k); } } }
    if (n >= 3) out.push({ x: sx / n, y: sy / n, r: Math.sqrt(n / Math.PI), pix });
  }
  return out;
}

// ---- instances of a 2-D label plane (ids > 0, e.g. Cellpose / StarDist masks) as the same {x, y, r, pix} records
export function labelComponents(lab, w, h, minPix = 3) {
  const by = new Map();
  for (let i = 0; i < w * h; i++) { const v = lab[i]; if (!v) continue; let c = by.get(v); if (!c) by.set(v, (c = { n: 0, sx: 0, sy: 0, pix: [] })); c.n++; c.sx += i % w; c.sy += (i / w) | 0; c.pix.push(i); }
  const out = []; for (const c of by.values()) if (c.n >= minPix) out.push({ x: c.sx / c.n, y: c.sy / c.n, r: Math.sqrt(c.n / Math.PI), pix: c.pix });
  return out;
}

// ---- 3-D objects = per-plane 2-D blobs linked across z.
// A 3-D connected-component pass merges touching cells (columns of neighbours become one tall ellipsoid), so each plane is split into
// 2-D components first and components in adjacent planes are linked when their centres overlap. A track ends where its area dips
// below 60% of its peak and rises again (two cells stacked in z). Object centre/z extent are area-weighted over the track;
// the in-plane radius comes from the largest cross-section, the z radius from the area-weighted z spread (r = sqrt(5 var)).
// planeComps[z] = [{x, y, r, pix}] with pix local to w x h. Returns per-voxel ids (z, y, x) on w x h and the object list.
export function linkTracks(planeComps, w, h, idBase = 0) {
  const nz = planeComps.length, tracks = []; let open = [];
  for (let z = 0; z < nz; z++) {
    const cs = planeComps[z], next = [], taken = new Set();
    const cand = []; for (let ci = 0; ci < cs.length; ci++) for (let ti = 0; ti < open.length; ti++) { const c = cs[ci], t = open[ti], l = t.pl[t.pl.length - 1], dd = Math.hypot(c.x - l.x, c.y - l.y); if (dd < 0.75 * Math.max(c.r, l.r)) cand.push([dd, ci, ti]); }
    cand.sort((p, q) => p[0] - q[0]); const cDone = new Set();
    for (const [, ci, ti] of cand) { if (cDone.has(ci) || taken.has(ti)) continue; const c = cs[ci], t = open[ti], l = t.pl[t.pl.length - 1], pk = Math.max(...t.pl.map((p) => p.r));
      if (l.r < 0.6 * pk && c.r > 1.3 * l.r) continue;                         // valley then rise: a different cell
      t.pl.push({ ...c, z }); cDone.add(ci); taken.add(ti); next.push(t); }
    for (let ci = 0; ci < cs.length; ci++) if (!cDone.has(ci)) { const t = { pl: [{ ...cs[ci], z }] }; tracks.push(t); next.push(t); }
    open = next;
  }
  const out = [], lab = new Uint32Array(nz * h * w);                           // per-voxel instance ids (the segmentation)
  tracks.forEach((t, ti) => { for (const p of t.pl) { const base = p.z * h * w; for (const q of p.pix) lab[base + q] = idBase + ti + 1; } });
  for (const t of tracks) {
    let A = 0, sx = 0, sy = 0, sz = 0, qz = 0, rmax = 0, nv = 0;
    for (const p of t.pl) { const a_ = p.r * p.r * Math.PI; A += a_; sx += a_ * p.x; sy += a_ * p.y; sz += a_ * p.z; qz += a_ * p.z * p.z; rmax = Math.max(rmax, p.r); nv += a_; }
    const mz = sz / A, vz = Math.max(qz / A - mz * mz, 0);
    out.push({ x: sx / A, y: sy / A, z: mz, rx: rmax, ry: rmax, rz: Math.sqrt(5 * vz + 0.25), vox: nv, edgeZ: t.pl[0].z === 0 || t.pl[t.pl.length - 1].z === nz - 1, planes: t.pl.length });
  }
  return { objs: out, n: tracks.length, lab };
}

// thresholded program output (padded crop, seg channel) -> linked 3-D instances
export function components3(seg, dims, ox, oy, w, h, nz, idBase = 0) {
  const plane = dims[1] * dims[2], pcs = [];
  for (let z = 0; z < nz; z++) pcs.push(components(seg.subarray(z * plane, (z + 1) * plane), dims, ox, oy, w, h));
  return linkTracks(pcs, w, h, idBase);
}
