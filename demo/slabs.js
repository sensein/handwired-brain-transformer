import { parseProgram, GPUEngine } from './engine.js';
import { Zarr, clamp } from './zarr.js';
import { segRegion, renderSegCanvas, regionStats, toCanvas, COLORS, CLASSES } from './seg.js';

const $ = (id) => document.getElementById(id);
const assets = (await (await fetch('assets.json')).json()).filter((a) => a.sub === 'MITU01');
const modelIndex = await (await fetch('models/index.json')).json();
const programs = {};
const getProgram = async (n) => programs[n] ||= parseProgram(await (await fetch(`models/${modelIndex[n].file}`)).text());
let engine = null; try { engine = await GPUEngine.create(); } catch (e) { /* CPU reference fallback */ }
const ROWS = ['YO', 'NN', 'LEC'];
const samples = [...new Set(assets.filter((a) => ROWS.includes(a.stain)).map((a) => a.sample))].sort((a, b) => +a - +b);
const LEVEL_LABEL = ['2.6 um/voxel (cells / vessels)', '5.1 um (cells / vessels)', '10 um (layers)', '20 um (layers)', '41 um (tissue)', '82 um (tissue)', '164 um (tissue)'];
LEVEL_LABEL.forEach((t, i) => $('level').add(new Option(`level ${i}: ${t}`, i)));
$('level').value = 3;
$('model').add(new Option('Auto (follows zoom)', 'auto'));
for (const k of Object.keys(modelIndex)) $('model').add(new Option(k, k));
$('model').add(new Option('None (raw only)', 'none'));

// routing by zoom (um per pixel = voxel size at the shown level): same rule as the viewer
function route(p, stain, sel) {
  if (sel !== 'auto') return { model: sel, level: null };
  if (p >= 30) return { model: 'tissue' };
  if (p >= 8.5) return { model: 'layers' };
  return { model: ['LEC', 'CD31'].includes(stain) ? 'vessels_26dir' : 'cells' };
}
function segLevel(model, zarr, p) {
  const kind = modelIndex[model].kind, sp0 = zarr.levels[0].sp[2];
  if (kind === 'layers') return Math.min(5, zarr.top);
  if (kind === 'tissue') return clamp(Math.ceil(Math.log2(Math.max(p, sp0) / sp0)), Math.min(4, zarr.top), Math.min(5, zarr.top));
  return 0;
}

// ---- grid
const results = new Map();     // cell key -> stats row (for CSV)
const table = $('grid');
table.innerHTML = `<tr><th></th>${samples.map((s) => `<th>${s}</th>`).join('')}</tr>` +
  ROWS.map((st) => `<tr><th class="row">${st}</th>${samples.map((s) => `<td data-stain="${st}" data-sample="${s}"></td>`).join('')}</tr>`).join('');
const cells = [...table.querySelectorAll('td')];
let generation = 0;

function pickChunk(stain, sample) {
  const list = assets.filter((a) => a.stain === stain && a.sample === sample).sort((a, b) => a.chunk - b.chunk);
  if (!list.length) return null;
  const m = $('chunk').value; return m === 'first' ? list[0] : m === 'last' ? list[list.length - 1] : list[Math.floor((list.length - 1) / 2)];
}

const queue = []; let running = false;
function enqueue(td) { queue.push([td, generation]); if (!running) pump(); }
async function pump() {
  running = true;
  while (queue.length) { const [td, g] = queue.shift(); if (g !== generation) continue; try { await loadCell(td, g); } catch (e) { td.innerHTML = `<small>failed: ${String(e.message).slice(0, 60)}</small>`; } status(); }
  running = false; status();
}
function status() { $('status').textContent = `${results.size} slab windows done · ${queue.length} queued · backend ${engine ? 'WebGPU' : 'CPU reference (slow)'}`; }

async function loadCell(td, g) {
  const stain = td.dataset.stain, sample = td.dataset.sample, a = pickChunk(stain, sample);
  if (!a) { td.innerHTML = '<small>no data</small>'; return; }
  const z = new Zarr(a.zarr); await z.init();
  const lv = Math.min(+$('level').value, z.top), sp = z.levels[lv].sp, p = sp[2];
  // brightest-tissue location from the coarsest level (few MB)
  const top = z.top, T = z.levels[top], arr = await z.crop(top, 0, T.shape[0], 0, T.shape[1], 0, T.shape[2]);
  const [Z, Y, X] = T.shape; let best = -1, bi = [Z >> 1, Y >> 1, X >> 1];
  for (let zz = 1; zz < Z - 1; zz += 2) for (let yy = 1; yy < Y - 1; yy += 2) for (let xx = 1; xx < X - 1; xx += 2) {
    let s = 0; for (let dz = -1; dz <= 1; dz++) for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) s += arr[((zz + dz) * Y + yy + dy) * X + xx + dx];
    if (s > best) { best = s; bi = [zz, yy, xx]; } }
  const f = 2 ** (top - lv), L = z.levels[lv], c = [bi[0] * f, bi[1] * f, bi[2] * f];
  const W = 128, x0 = clamp(Math.round(c[2] - W / 2), 0, Math.max(0, L.shape[2] - W)), x1 = Math.min(L.shape[2], x0 + W);
  const y0 = clamp(Math.round(c[1] - W / 2), 0, Math.max(0, L.shape[1] - W)), y1 = Math.min(L.shape[1], y0 + W);
  const zl = clamp(Math.round(c[0]), 0, L.shape[0] - 1);
  const raw = await z.crop(lv, zl, zl + 1, y0, y1, x0, x1), w = x1 - x0, h = y1 - y0;
  const sorted = Float32Array.from(raw).sort(), lo = sorted[Math.floor(sorted.length * 0.02)], hi = Math.max(sorted[Math.floor(sorted.length * 0.997)], lo + 1);
  // physical frame of the cell (um), scaled so the longer side is 132 px
  const wum = w * sp[2], hum = h * sp[1], k = 132 / Math.max(wum, hum);
  td.innerHTML = '<div class="cell"><canvas width="132" height="132"></canvas><small></small></div>';
  const cv = td.querySelector('canvas'), ctx = cv.getContext('2d'); ctx.imageSmoothingEnabled = false;
  const base = toCanvas(null, w, h, (px) => { for (let i = 0; i < w * h; i++) { const v = clamp((raw[i] - lo) / (hi - lo), 0, 1) * 255; px[4 * i] = px[4 * i + 1] = px[4 * i + 2] = v; px[4 * i + 3] = 255; } });
  ctx.drawImage(base, 0, 0, wum * k, hum * k);
  let label = 'raw';
  const r = route(p, stain, $('model').value), model = r.model;
  if (model !== 'none' && g === generation) {
    const kind = modelIndex[model].kind, prog = await getProgram(model), rl = r.level ?? segLevel(model, z, p), SL = z.levels[rl], ts = modelIndex[model].tile || 128;
    const cxum = (x0 + w / 2) * sp[2], cyum = (y0 + h / 2) * sp[1];     // window center in um
    const gx = clamp(Math.floor(cxum / SL.sp[2] / ts), 0, Math.ceil(SL.shape[2] / ts) - 1), gy = clamp(Math.floor(cyum / SL.sp[1] / ts), 0, Math.ceil(SL.shape[1] / ts) - 1);
    const sy0 = gy * ts, sx0 = gx * ts, sy1 = Math.min(sy0 + ts, SL.shape[1]), sx1 = Math.min(sx0 + ts, SL.shape[2]);
    const szl = clamp(Math.floor(zl * 2 ** lv / 2 ** rl), 0, SL.shape[0] - 1);
    const seg = await segRegion(z, engine, prog, rl, szl, sy0, sy1, sx0, sx1);
    if (g !== generation) return;
    ctx.globalAlpha = +$('opacity').value / 100;
    ctx.drawImage(renderSegCanvas(kind, seg), (sx0 * SL.sp[2] - x0 * sp[2]) * k, (sy0 * SL.sp[1] - y0 * sp[1]) * k, (sx1 - sx0) * SL.sp[2] * k, (sy1 - sy0) * SL.sp[1] * k);
    ctx.globalAlpha = 1;
    const st = regionStats(kind, seg, SL.sp);
    results.set(`${stain}|${sample}`, { stain, sample, chunk: a.chunk, zarr: a.zarr, level: lv, model, seg_level: rl, ...st });
    const fmt = Object.entries(st).sort((u, v) => v[1] - u[1]).slice(0, 3).map(([n, v]) => n === 'cells_per_mm2' ? `${Math.round(v)} cells/mm2` : `${n} ${(v * 100).toFixed(0)}%`).join(' ');
    label = `${model} L${rl} · ${fmt}`;
  }
  td.querySelector('small').textContent = `${label}`;
  td.querySelector('.cell').onclick = () => open(`index.html#id=${a.zarr}&p=${p.toFixed(2)}&cx=${cxum0(x0, w, sp)}&cy=${cyum0(y0, h, sp)}&z=${Math.round(zl * 2 ** lv)}`, '_blank');
}
const cxum0 = (x0, w, sp) => Math.round((x0 + w / 2) * sp[2]), cyum0 = (y0, h, sp) => Math.round((y0 + h / 2) * sp[1]);

// ---- lazy loading + controls
const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting && !e.target.dataset.queued) { e.target.dataset.queued = '1'; enqueue(e.target); } }), { root: $('scroll'), rootMargin: '200px' });
function reset() { generation++; queue.length = 0; results.clear(); cells.forEach((td) => { td.innerHTML = ''; delete td.dataset.queued; io.unobserve(td); io.observe(td); }); legend(); status(); }
function legend() {
  const m = $('model').value, lv = +$('level').value, r = route(lv >= 0 ? [2.6, 5.1, 10.3, 20.5, 41, 82, 164][lv] : 0, 'YO', m).model;
  const kind = r === 'none' ? null : modelIndex[r].kind;
  $('legend').innerHTML = CLASSES[kind] ? CLASSES[kind].map((n) => `<span style="background:rgb(${COLORS[n]})"></span>${n}`).join('') : kind === 'vessels' ? '<span style="background:rgb(40,220,255)"></span>vessel' : kind === 'cells' ? '<span style="background:rgb(255,60,200)"></span>cell' : '';
}
['level', 'model', 'chunk'].forEach((id) => { $(id).onchange = reset; });
$('opacity').oninput = () => reset();
$('csv').onclick = () => {
  const rows = [...results.values()], cols = [...new Set(rows.flatMap((r) => Object.keys(r)))];
  const csv = [cols.join(','), ...rows.map((r) => cols.map((c) => r[c] ?? '').join(','))].join('\n');
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' })); a.download = 'slab_stats.csv'; a.click();
};
window.__slabs = { get done() { return results.size; }, get stats() { return [...results.values()]; }, reset };
reset();
