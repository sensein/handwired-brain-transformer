// Integration check of the viewer path for hier1 programs: seg.js segRegion / segVolume (raw log input, no exposure) on the
// WebGPU engine with a mock zarr serving the golden crops; compares the centre plane with the Python golden.
// Run from a directory with playwright installed: DEMO_ROOT=<repo>/demo node hier_seg_test.mjs
import { chromium } from 'playwright';
import http from 'http'; import fs from 'fs'; import path from 'path';
const ROOT = process.env.DEMO_ROOT || path.resolve('demo');
const mime = { '.js': 'text/javascript', '.json': 'application/json', '.html': 'text/html', '.wasm': 'application/wasm' };
const srv = http.createServer((q, r) => { const f = path.join(ROOT, q.url.split('?')[0]);
  if (!fs.existsSync(f)) { r.writeHead(404); return r.end(); }
  r.writeHead(200, { 'content-type': mime[path.extname(f)] || 'application/octet-stream' }); r.end(fs.readFileSync(f)); }).listen(8125);
const args = ['--no-sandbox', '--enable-unsafe-webgpu', '--enable-features=Vulkan,UnsafeWebGPU', '--disable-vulkan-surface',
  '--use-vulkan=swiftshader', '--use-webgpu-adapter=swiftshader', '--use-angle=swiftshader', '--ignore-gpu-blocklist', '--headless=new'];
const browser = await chromium.launch({ args, headless: true });
const page = await browser.newPage();
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
await page.goto('http://localhost:8125/test/blank.html');
const res = await page.evaluate(async () => {
  const E = await import('/engine.js'), H = await import('/hier.js'), Sg = await import('/seg.js'), R = await import('/route.js');
  const engine = await E.GPUEngine.create(), gold = await (await fetch('/test/golden_hier.json')).json(), out = {};
  for (const name of ['hier_regions', 'hier_cells']) {
    const g = gold[name], prog = H.parseHier(await (await fetch(`/models/${name}.json`)).text()), [Z, Y, X] = g.dims, vol = Float32Array.from(g.logI);
    const zarr = { top: 0, levels: [{ sp: g.spacing, shape: g.dims }], crop: async (l, z0, z1, y0, y1, x0, x1) => {
      const o = new Float32Array((z1 - z0) * (y1 - y0) * (x1 - x0)); let k = 0;
      for (let z = z0; z < z1; z++) for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) o[k++] = vol[(z * Y + y) * X + x]; return o; } };
    const zc = Z >> 1, key = name === 'hier_regions' ? 'tissue' : 'seg';
    const halo = Sg.haloOf(prog, g.spacing);
    // the whole golden crop is the FOV (halo off, all planes): identical to the Python run
    const r = await Sg.segVolume(zarr, engine, prog, 0, 0, Z, 0, Y, 0, X, () => { throw new Error('exposure must not be applied'); }, { halo: false });
    const ref = g.out[key]; let flips = 0, md = 0;
    for (let z = 0; z < Z; z++) for (let i = 0; i < Y * X; i++) { const a = r.out[key][z * Y * X + i], b = ref[z * Y * X + i]; if ((a > 0.5) !== (b > 0.5)) flips++; md = Math.max(md, Math.abs(a - b)); }
    out[name] = { halo, flipFrac: flips / (Z * Y * X), maxdiff: md, family: R.FAMILIES[name], route: R.scaledModel(name, 3, { top: 0, levels: [{ sp: g.spacing }] }) };
  }
  out.familyForZoom = { yoFine: R.familyForZoom(2, 'YO', { hier: true }), yoCoarse: R.familyForZoom(80, 'YO', { hier: true }), lec: R.familyForZoom(3, 'LEC', { hier: true }) };
  return out;
});
console.log(JSON.stringify(res, null, 1));
await browser.close(); srv.close();
const ok = res.hier_regions && res.hier_regions.flipFrac < 5e-3 && res.hier_cells.flipFrac < 5e-3;
process.exit(ok ? 0 : 1);
