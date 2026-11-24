// Python lab: author transformer weights in Python (numpy-only hwseg modules, run by Pyodide in the page), then execute
// them on the WebGPU engine against a window streamed from DANDI 000108. No server, no training.
import { parseProgram, nParams, GPUEngine } from './engine.js';
import { Zarr } from './zarr.js';
import { segRegion, percentileWindow, adjustLogVol, IDENT } from './seg.js';
import { banner } from './banner.js';

const $ = (id) => document.getElementById(id);
const PYO = 'https://cdn.jsdelivr.net/pyodide/v314.0.8/full/';
const DEFAULT = `# Build a program from the numpy-only hwseg modules, or write the weights yourself. Name the result \`prog\`.
# Try: change kappa / tau0 (detection thresholds), sigma_um (blob size), scale (x2 = larger blobs), or swap in vessels().
from hwseg.programs import cells, vessels

prog = cells(sigma_um=3.5, kappa=2.0, tau0=0.03, scale=1)
# prog = vessels(n_dirs=13, kappa=1.0, tau0=0.12)
`;
try { $('code').value = localStorage.getItem('hw-lab-code') || DEFAULT; } catch (e) { $('code').value = DEFAULT; }
const assets = await (await fetch('assets.json')).json();
const pick = { YO: assets.find((a) => a.stain === 'YO' && a.sample === '90' && a.chunk === 5), NN: assets.find((a) => a.stain === 'NN' && a.sample === '86' && a.chunk === 3) || assets.find((a) => a.stain === 'NN'), LEC: assets.find((a) => a.stain === 'LEC' && a.sample === '150' && a.chunk === 3) || assets.find((a) => a.stain === 'LEC') };
let py = null, engine = null, last = null; const spots = {};

async function getPy() {
  if (py) return py;
  const t = banner.start('other', 'Loading Pyodide (Python in WebAssembly) + numpy');
  await new Promise((res, rej) => { const s = document.createElement('script'); s.src = PYO + 'pyodide.js'; s.onload = res; s.onerror = () => rej(new Error('could not load Pyodide from the CDN')); document.head.append(s); });
  py = await loadPyodide({ indexURL: PYO });   // eslint-disable-line no-undef
  await py.loadPackage('numpy');
  py.FS.mkdirTree('/pkg/hwseg'); py.FS.writeFile('/pkg/hwseg/__init__.py', '');
  for (const f of ['program', 'wiring', 'programs', 'exposure']) py.FS.writeFile(`/pkg/hwseg/${f}.py`, await (await fetch(`py/hwseg/${f}.py`)).text());
  py.runPython("import sys; sys.path.insert(0, '/pkg')");
  banner.finish(t, true, 'Python ' + py.runPython('import sys; sys.version.split()[0]') + ', numpy ready');
  return py;
}

async function getSpot(stain) {   // brightest tissue of the first acquisition for this stain, as a 128 x 128 level-0 window
  if (spots[stain]) return spots[stain];
  const t = banner.start('scan', `Opening ${stain} and finding tissue`);
  const z = new Zarr(pick[stain].zarr); await z.init();
  const top = z.top, L = z.levels[top], a = await z.crop(top, 0, L.shape[0], 0, L.shape[1], 0, L.shape[2]), [Z, Y, X] = L.shape; let best = -1, bi = [Z >> 1, Y >> 1, X >> 1];
  for (let zz = 1; zz < Z - 1; zz += 2) for (let yy = 1; yy < Y - 1; yy += 2) for (let xx = 1; xx < X - 1; xx += 2) { let s = 0; for (let dz = -1; dz <= 1; dz++) for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) s += a[((zz + dz) * Y + yy + dy) * X + xx + dx]; if (s > best) { best = s; bi = [zz, yy, xx]; } }
  const f = 2 ** top, L0 = z.levels[0], N = 128, cl = (v, n) => Math.min(Math.max(v, 0), Math.max(0, n - N));
  const y0 = cl(Math.round(bi[1] * f - N / 2), L0.shape[1]), x0 = cl(Math.round(bi[2] * f - N / 2), L0.shape[2]), zl = Math.min(Math.round(bi[0] * f), L0.shape[0] - 1);
  banner.finish(t, true, `level-0 window at z ${zl}, y ${y0}, x ${x0}`);
  return (spots[stain] = { z, y0, x0, zl, N });
}

function show() {
  if (!last) return;
  const { r, prog } = last, name = $('outsel').value, cv = $('cv'), g = cv.getContext('2d'), img = g.createImageData(r.w, r.h);
  cv.width = r.w; cv.height = r.h; let on = 0;
  for (let y = 0; y < r.h; y++) for (let x = 0; x < r.w; x++) {
    const i = (r.oy + y) * r.dims[2] + r.ox + x, o = 4 * (y * r.w + x), u = r.raw[i] * 255, hit = $('showov').checked && r.out[name][i] > 0.5; if (r.out[name][i] > 0.5) on++;
    img.data[o] = hit ? 0.35 * u + 0.65 * 255 : u; img.data[o + 1] = hit ? 0.35 * u + 0.65 * 60 : u; img.data[o + 2] = hit ? 0.35 * u + 0.65 * 200 : u; img.data[o + 3] = 255;
  }
  g.putImageData(img, 0, 0);
  $('stat').textContent = `${name} > 0.5 on ${(100 * on / (r.w * r.h)).toFixed(2)}% of the window`;
}

async function run() {
  const out = $('out'); out.className = ''; out.textContent = '';
  try { localStorage.setItem('hw-lab-code', $('code').value); } catch (e) { /* storage unavailable */ }
  const t = banner.start('infer', 'Python lab: build weights, run on WebGPU');
  try {
    const p = await getPy();
    banner.update(t, { detail: 'running your Python' });
    const t0 = performance.now(), g = p.globals.get('dict')();
    p.runPython($('code').value, { globals: g });
    if (!g.has('prog')) throw new Error('your code must define `prog` (a Program)');
    const json = g.get('prog').to_json(), prog = parseProgram(json), np = nParams(prog), tPy = performance.now() - t0;
    out.textContent = `Python built "${prog.name}" in ${tPy.toFixed(0)} ms: ${prog.layers.length} layers, ${prog.layers.reduce((a, l) => a + l.heads.length, 0)} heads, d=${prog.d}, ${np.nonzero} nonzero params (${np.total} dense), ${(json.length / 1024).toFixed(1)} kB JSON\n`;
    $('outsel').innerHTML = ''; for (const k of Object.keys(prog.out)) $('outsel').add(new Option(k, k)); if (prog.out.seg !== undefined) $('outsel').value = 'seg';
    if (!engine) { engine = await GPUEngine.create(); banner.log('WebGPU ready'); }
    banner.update(t, { detail: 'streaming a window from DANDI' });
    const spot = await getSpot($('stainsel').value);
    banner.update(t, { detail: 'running the model on WebGPU' });
    const t1 = performance.now();
    const r = await segRegion(spot.z, engine, prog, 0, spot.zl, spot.y0, spot.y0 + spot.N, spot.x0, spot.x0 + spot.N, (v) => { const w = percentileWindow(v); return adjustLogVol(v, w[0], w[1], IDENT); }, { halo: true });
    out.textContent += `WebGPU ran it on a ${r.w}x${r.h} window (+halo, ${r.dims.join('x')} voxels) in ${(performance.now() - t1).toFixed(0)} ms`;
    last = { r, prog }; show(); banner.finish(t, true, `${np.nonzero} params`);
  } catch (e) { out.className = 'err'; out.textContent = String(e.message || e); banner.finish(t, false, String(e.message || e).split('\n').pop()); }
}
$('run').onclick = run; $('outsel').onchange = show; $('showov').onchange = show;
$('code').addEventListener('keydown', (e) => { if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') { e.preventDefault(); run(); } });
banner.init(); banner.setStream(() => null);
window.__lab = { run, get last() { return last; } };
