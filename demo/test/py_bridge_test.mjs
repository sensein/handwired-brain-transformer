// Our own Neuroglancer build on the assembled Pages tree: the hook reads the view, runs the model on it, writes annotations back.
import { chromium } from 'playwright';
import { ngState } from '../ng-state.js';
import http from 'http'; import fs from 'fs'; import path from 'path';
const ROOT = process.env.SITE, OUT = process.env.SHOT_DIR || '.', PORT = +(process.env.PORT || 8261), ASSET = process.env.ASSET;
const mime = { '.js': 'text/javascript', '.json': 'application/json', '.html': 'text/html', '.css': 'text/css', '.wasm': 'application/wasm', '.py': 'text/plain' };
const srv = http.createServer((q, r) => { let f = path.join(ROOT, q.url.split('?')[0].split('#')[0]); if (f.endsWith('/')) f += 'index.html';
  if (!fs.existsSync(f) || fs.statSync(f).isDirectory()) { r.writeHead(404); return r.end(); }
  r.writeHead(200, { 'content-type': mime[path.extname(f)] || 'application/octet-stream' }); r.end(fs.readFileSync(f)); }).listen(PORT);
const browser = await chromium.launch({ headless: true, args: ['--no-sandbox', '--enable-unsafe-webgpu', '--enable-features=Vulkan,UnsafeWebGPU', '--disable-vulkan-surface', '--use-vulkan=swiftshader', '--use-webgpu-adapter=swiftshader', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist', '--headless=new'] });
const page = await browser.newPage({ viewport: { width: 1300, height: 800 } });
page.on('pageerror', (e) => console.log('[pageerror]', e.message.slice(0, 200)));
const sp = [2.5632620818012344, 3.625, 2.5632620818012344];
const state = ngState([ASSET], { sp, off: [180247 , 26568, -237] }, { cx: 24818 * sp[2], cy: 1023 * sp[1], z: 1019, p: +(process.env.P || 3.1) });
await page.goto(`http://localhost:${PORT}/ng/index.html#!${encodeURIComponent(JSON.stringify(state))}`);
await page.waitForFunction(() => window.__hwNg && window.__hwNg.run, null, { timeout: 120000 });
console.log('hook ready:', await page.evaluate(() => window.__hwNg.status));
await page.waitForTimeout(8000);
await page.screenshot({ path: `${OUT}/ng_before.png` });
const CODE = `import numpy as np
from scipy import ndimage as ndi
v = await ng.read(depth=5)
print(v)
sig = 3.0 / np.mean(v.spacing_um)
g = ndi.gaussian_filter(v.data, sig)
bg = ndi.gaussian_filter(v.data, 4 * sig)
mask = (g - bg) > 0.25
labels, n = ndi.label(mask)
print(n, "blobs at level", v.level)
await ng.segmentation(labels.astype(np.uint32), view=v, name="python blobs")
cm = np.array(ndi.center_of_mass(mask, labels, range(1, n + 1)))
if n: await ng.annotations(cm, view=v, radii_um=5, name="python centroids")
`;
let t0 = Date.now(); const r = await page.evaluate((c) => window.__hwNg.runPython(c), CODE);
console.log('python run:', JSON.stringify(r).slice(0, 3000), `(${((Date.now() - t0) / 1000).toFixed(0)} s)`);
const layers = await page.evaluate(() => window.viewer.state.toJSON().layers.map((l) => l.name + ':' + l.type + (l.annotations ? '(' + l.annotations.length + ')' : '')));
console.log('layers:', JSON.stringify(layers));
const stateJson = await page.evaluate(() => JSON.stringify(window.viewer.state.toJSON()));
const count = (p) => p.evaluate(async () => { const j = window.viewer.state.toJSON(), seg = j.layers.find((l) => l.type === 'segmentation'); if (!seg) return -1;
  const u = String(seg.source.url || seg.source).replace(/^precomputed:\/\//, ''), info = await (await fetch(u + '/info')).json(), sc = info.scales[0], px = j.position, ratio = sc.resolution[0] / (j.dimensions.x[0] * 1e9);
  const x0 = Math.floor(px[3] / ratio / 64) * 64, y0 = Math.floor(px[2] / ratio / 64) * 64, z0 = Math.floor(px[1] / ratio / 16) * 16; let n = 0, ids = new Set();   // sum over 9 x 7 xy chunks and 2 z chunks around the centre
  for (let cx = -4; cx <= 4; cx++) for (let cy = -3; cy <= 3; cy++) for (let cz = -1; cz <= 0; cz++) { const x = x0 + 64 * cx, y = y0 + 64 * cy, z = z0 + 16 * cz;
    const b = new Uint32Array(await (await fetch(`${u}/0/${x}-${x + 64}_${y}-${y + 64}_${z}-${z + 16}`)).arrayBuffer()); for (const q of b) if (q) { n++; ids.add(q); } }
  return { voxels: n, objects: ids.size }; });
console.log('A labels:', JSON.stringify(await count(page)));
await page.screenshot({ path: `${OUT}/py_A.png` });
// a fresh page opened from the state alone: the Python code must NOT run until approved
const aEvents = (await page.evaluate(() => window.__hwNg.events.map((e) => e.msg))).filter((m) => /python/.test(m)); console.log('A events:', JSON.stringify(aEvents.slice(-4)));
await page.close();   // a link is opened in a fresh page; the first page (which holds the original job) is gone
const page2 = await browser.newPage({ viewport: { width: 1300, height: 800 } }); page2.on('pageerror', (e) => console.log('[B pageerror]', e.message.slice(0, 200)));
await page2.bringToFront();   // the service worker answers via the most recently focused Neuroglancer page
await page2.goto(`http://localhost:${PORT}/ng/index.html#!${encodeURIComponent(stateJson)}`);
await page2.waitForFunction(() => window.__hwNg && window.__hwNg.ready, null, { timeout: 120000 }); await page2.waitForTimeout(8000);
console.log('B pending approval:', JSON.stringify(await page2.evaluate(() => window.__hwNg.pendingPy.map((p) => ({ name: p.name, codeLen: p.code.length })))), '| B labels before approval:', JSON.stringify(await count(page2)));
t0 = Date.now(); await page2.evaluate(() => window.__hwNg.approvePy(window.__hwNg.pendingPy[0].id)); await page2.waitForFunction(() => window.__hwNg.events.some((e) => /regenerated segmentation/.test(e.msg)), null, { timeout: 120000 }).catch(() => console.log('B: regeneration did not finish in 120 s'));
console.log('B labels after approval:', JSON.stringify(await count(page2)), `(${((Date.now() - t0) / 1000).toFixed(0)} s)`);
console.log('B events:', JSON.stringify((await page2.evaluate(() => window.__hwNg.events.map((e) => e.msg))).filter((m) => /python|failed|error/.test(m))));
await page2.screenshot({ path: `${OUT}/py_B.png` });
await browser.close(); srv.close();
