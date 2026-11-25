// Label-free comparison of our hand-written programs with the Neurodesk reference models on the same real windows.
// Runs everything in headless Chromium exactly as the Neuroglancer hook does (WebGPU engine for our programs, onnxruntime-web
// for the references), on windows streamed from DANDI 000108, and dumps the window + every label volume to OUT_DIR for
// scripts/nd_compare.py (metrics, overlays). Env: DEMO_ROOT (demo/), OUT_DIR, PORT, EP (wasm|webgpu), CELLS / VESSELS (asset
// specs "STAIN:SAMPLE,..."), CORE (window core, px), Z (planes incl. margins).
import { chromium } from 'playwright';
import http from 'http'; import fs from 'fs'; import path from 'path';
const ROOT = process.env.DEMO_ROOT || path.resolve('demo'), OUT = process.env.OUT_DIR || 'nd_compare_out', PORT = +(process.env.PORT || 8391);
fs.mkdirSync(OUT, { recursive: true });
const mime = { '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json', '.html': 'text/html' };
// cross-origin isolated (COOP + COEP credentialless) so onnxruntime-web may use WASM threads
const srv = http.createServer((q, r) => { const f = path.join(ROOT, q.url.split('?')[0]); if (!fs.existsSync(f) || fs.statSync(f).isDirectory()) { r.writeHead(404); return r.end(); }
  r.writeHead(200, { 'content-type': mime[path.extname(f)] || 'application/octet-stream', 'cross-origin-opener-policy': 'same-origin', 'cross-origin-embedder-policy': 'credentialless' }); r.end(fs.readFileSync(f)); }).listen(PORT);
const browser = await chromium.launch({ headless: true, args: ['--no-sandbox', '--enable-unsafe-webgpu', '--enable-features=Vulkan,UnsafeWebGPU', '--disable-vulkan-surface', '--use-vulkan=swiftshader', '--use-webgpu-adapter=swiftshader', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist', '--headless=new'] });
const page = await browser.newPage();
page.on('console', (m) => { if (!/Download the React|tf_half_pixel/.test(m.text())) console.log('[page]', m.text().slice(0, 300)); });
page.on('pageerror', (e) => console.log('[pageerror]', e.message.slice(0, 300)));
await page.exposeFunction('saveBin', (name, b64) => { fs.writeFileSync(path.join(OUT, name), Buffer.from(b64, 'base64')); });
await page.goto(`http://localhost:${PORT}/test/blank.html`);
const assets = JSON.parse(fs.readFileSync(path.join(ROOT, 'assets.json')));
const pick = (spec) => spec.split(',').filter(Boolean).map((s) => { const [stain, sample] = s.split(':'), rows = assets.filter((a) => a.stain === stain && a.sample === sample && a.sub === 'MITU01').sort((a, b) => a.chunk - b.chunk); return { stain, sample, zarr: rows[rows.length >> 1].zarr, chunk: rows[rows.length >> 1].chunk }; });
const jobs = [...pick(process.env.CELLS ?? 'YO:90,YO:117,NN:40,NN:91').map((a) => ({ ...a, task: 'cells' })), ...pick(process.env.VESSELS ?? 'LEC:165,LEC:97').map((a) => ({ ...a, task: 'vessels' }))];
const t00 = Date.now();
const meta = await page.evaluate(async ({ jobs, EP, CORE, ZN }) => {
  globalThis.__hwNdEP = EP;
  const E = await import('/engine.js'), Zm = await import('/zarr.js'), S = await import('/seg.js'), I = await import('/instances.js'), N = await import('/nd/nd-seg.js');
  const engine = await E.GPUEngine.create(), index = await (await fetch('/models/index.json')).json();
  const prog = async (k) => E.parseProgram(await (await fetch(`/models/${index[k].file}`)).text());
  const b64 = (arr) => { const u = new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength); let s = ''; for (let i = 0; i < u.length; i += 0x8000) s += String.fromCharCode.apply(null, u.subarray(i, i + 0x8000)); return btoa(s); };
  const out = [], M = 24;                                                         // M = context margin around the scored core
  for (const job of jobs) {
    const z = new Zm.Zarr(job.zarr); await z.init(); const L0 = z.levels[0], L4 = z.levels[4], sp = L0.sp;
    // tissue-rich windows: 12 x 12 blocks of the level-4 centre plane ranked by mean log intensity; take the 80th and 95th percentile blocks
    const zc4 = L4.shape[0] >> 1, pl = await z.crop(4, zc4, zc4 + 1, 0, L4.shape[1], 0, L4.shape[2]), B = Math.max(2, Math.round(CORE / 16)), blocks = [];
    for (let by = 1; by + 1 < Math.floor(L4.shape[1] / B); by++) for (let bx = 1; bx + 1 < Math.floor(L4.shape[2] / B); bx++) { let s = 0; for (let y = 0; y < B; y++) for (let x = 0; x < B; x++) s += pl[(by * B + y) * L4.shape[2] + bx * B + x]; blocks.push({ by, bx, m: s / (B * B) }); }
    blocks.sort((a, b) => a.m - b.m);
    for (const q of [0.8, 0.95]) {
      const bk = blocks[Math.floor(q * (blocks.length - 1))], cy = Math.round((bk.by + 0.5) * B * 16), cx = Math.round((bk.bx + 0.5) * B * 16), zc = L0.shape[0] >> 1;
      const y0 = cy - (CORE >> 1) - M, y1 = y0 + CORE + 2 * M, x0 = cx - (CORE >> 1) - M, x1 = x0 + CORE + 2 * M, z0 = zc - (ZN >> 1), z1 = z0 + ZN, H = y1 - y0, W = x1 - x0;
      const name = `${job.stain}${job.sample}_q${Math.round(q * 100)}`, rec0 = { name, ...job, q, box: [z0, z1, y0, y1, x0, x1], core: CORE, margin: M, sp, models: {} };
      const vol = await z.crop(0, z0, z1, y0, y1, x0, x1); await saveBin(`${name}__raw.f32`, b64(vol));
      const t1 = performance.now(), rec = await S.recommendExposure(z, engine, await prog('exposure'), 0, zc, y0, y1, x0, x1), adjust = (v) => S.adjustLogVol(v, rec.lo, rec.hi, { b: 0, c: 1, g: rec.gamma });
      rec0.exposure = rec;
      const hand = job.task === 'cells' ? 'fable_blobs_v2' : 'fable_vessels_v2';
      const r = await S.segVolume(z, engine, await prog(hand), 0, z0, z1, y0, y1, x0, x1, adjust, { halo: false });
      let lab;
      if (job.task === 'cells') { const t3 = I.components3(r.out.seg, r.dims, r.ox, r.oy, r.w, r.h, r.nz); lab = t3.lab; rec0.models[hand] = { n: t3.n, s: (performance.now() - t1) / 1000 }; }
      else { lab = Uint32Array.from(S.labelVolume('vessels_ms', r)); rec0.models[hand] = { s: (performance.now() - t1) / 1000 }; }
      await saveBin(`${name}__${hand}.u32`, b64(lab)); console.log(name, hand, JSON.stringify(rec0.models[hand]));
      const refs = job.task === 'cells' ? ['nd_cellpose', 'nd_stardist', 'nd_cellfinder'] : ['nd_vesselboost'];
      const diamUm = job.task === 'cells' ? (job.stain === 'NN' ? 16 : 10) : 8; rec0.diamUm = diamUm;
      for (const k of refs) {
        const t2 = performance.now();
        if (k === 'nd_vesselboost') {           // 64-plane context slab centred on the window, scored on the window's planes
          const Z64 = 64, za = zc - (Z64 >> 1), v64 = await z.crop(0, za, za + Z64, y0, y1, x0, x1), rr = await N.refTile(k, v64, [Z64, H, W], sp, { diamUm });
          const o = new Uint32Array(ZN * H * W); o.set(rr.cls.subarray((z0 - za) * H * W, (z1 - za) * H * W)); rec0.models[k] = { s: (performance.now() - t2) / 1000, patches: rr.patches }; await saveBin(`${name}__${k}.u32`, b64(o));
        } else if (k === 'nd_cellfinder') {     // extra z context for the ball filter, scored on the window's planes
          const zx = N.REF[k].zCtx, vx = await z.crop(0, z0 - zx, z1 + zx, y0, y1, x0, x1), rr = await N.refTile(k, vx, [ZN + 2 * zx, H, W], sp, { diamUm });
          const o = new Uint32Array(ZN * H * W); o.set(rr.lab.subarray(zx * H * W, (zx + ZN) * H * W)); rec0.models[k] = { n: rr.n, s: (performance.now() - t2) / 1000 }; await saveBin(`${name}__${k}.u32`, b64(o));
        } else {
          const norm = N.fovNorm(k, vol.subarray((ZN >> 1) * H * W, ((ZN >> 1) + 1) * H * W)), rr = await N.refTile(k, vol, [ZN, H, W], sp, { diamUm, norm });
          rec0.models[k] = { n: rr.n, s: (performance.now() - t2) / 1000, norm, resampled: rr.resampled }; await saveBin(`${name}__${k}.u32`, b64(rr.lab));
        }
        console.log(name, k, JSON.stringify(rec0.models[k]));
      }
      rec0.nd = N.ndInfo(); out.push(rec0);
    }
  }
  return out;
}, { jobs, EP: process.env.EP || 'wasm', CORE: +(process.env.CORE || 160), ZN: +(process.env.Z || 9) });
fs.writeFileSync(path.join(OUT, 'meta.json'), JSON.stringify(meta, null, 1));
console.log(`done: ${meta.length} windows in ${((Date.now() - t00) / 60000).toFixed(1)} min -> ${OUT}`);
await browser.close(); srv.close();
