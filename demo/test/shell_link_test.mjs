// The Neuroglancer shell page: sidebar selection, embedded Neuroglancer + hook, run a model on the whole stitched slab.
import { chromium } from 'playwright';
import http from 'http'; import fs from 'fs'; import path from 'path';
const ROOT = process.env.SITE, OUT = process.env.SHOT_DIR || '.', PORT = +(process.env.PORT || 8311);
const mime = { '.js': 'text/javascript', '.json': 'application/json', '.html': 'text/html', '.css': 'text/css', '.wasm': 'application/wasm', '.py': 'text/plain' };
const srv = http.createServer((q, r) => { let f = path.join(ROOT, q.url.split('?')[0].split('#')[0]); if (f.endsWith('/')) f += 'index.html';
  if (!fs.existsSync(f) || fs.statSync(f).isDirectory()) { r.writeHead(404); return r.end(); }
  r.writeHead(200, { 'content-type': mime[path.extname(f)] || 'application/octet-stream' }); r.end(fs.readFileSync(f)); }).listen(PORT);
const browser = await chromium.launch({ headless: true, args: ['--no-sandbox', '--enable-unsafe-webgpu', '--enable-features=Vulkan,UnsafeWebGPU', '--disable-vulkan-surface', '--use-vulkan=swiftshader', '--use-webgpu-adapter=swiftshader', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist', '--headless=new'] });
const ctx = await browser.newContext({ viewport: { width: 1500, height: 900 } });
const A = await ctx.newPage(); A.on('pageerror', (e) => console.log('[A pageerror]', e.message.slice(0, 200)));
await A.goto(`http://localhost:${PORT}/index.html`);
const ready = (p) => p.waitForFunction(() => { const h = window.__shell && window.__shell.hook(); return h && h.ready; }, null, { timeout: 150000 });
await ready(A); await A.waitForTimeout(12000);
await A.selectOption('#model', 'regions'); await A.click('#run');
let t0 = Date.now(), st = ''; while (Date.now() - t0 < 400000) { await A.waitForTimeout(3000); st = await A.evaluate(() => window.__shell.hook().status); if (/segments|error|zoom in/.test(st)) break; }
console.log('A run:', st, `(${((Date.now() - t0) / 1000).toFixed(0)} s)`); await A.waitForTimeout(4000);
await A.evaluate(() => { const r = document.getElementById('segalpha'); r.value = 20; r.dispatchEvent(new Event('input')); }); await A.waitForTimeout(1500);
console.log('A opacity in state:', await A.evaluate(() => document.getElementById('ng').contentWindow.viewer.state.toJSON().layers.filter((l) => l.type === 'segmentation').map((l) => l.selectedAlpha)));
const url = await A.evaluate(() => window.__shell.shareUrl()); console.log('link length', url.length, '| has detections annotations:', /hwseg%20detections/.test(url) && /annotations/.test(decodeURIComponent(url).split('annotations')[1] || ''));
await A.screenshot({ path: `${OUT}/link_A.png` });
// a second browser context: nothing in memory, only the link
const ctx2 = await browser.newContext({ viewport: { width: 1500, height: 900 } }); const B = await ctx2.newPage(); B.on('pageerror', (e) => console.log('[B pageerror]', e.message.slice(0, 200)));
await B.goto(url.replace(/^https?:\/\/[^/]+/, `http://localhost:${PORT}`));
await ready(B);
const probe = (p) => p.evaluate(async () => {
  const v = document.getElementById('ng').contentWindow.viewer, j = v.state.toJSON(), seg = j.layers.find((l) => l.type === 'segmentation'), img = j.layers.find((l) => l.type === 'image'); if (!seg) return { layers: j.layers.map((l) => l.name) };
  const u = String(seg.source.url || seg.source).replace(/^precomputed:\/\//, ''), info = await (await fetch(u + '/info')).json(), sc = info.scales[0], px = j.position;
  const ratio = sc.resolution[0] / (j.dimensions.x[0] * 1e9), [x, y] = [Math.floor(px[0] / ratio / 64) * 64, Math.floor(px[1] / ratio / 64) * 64];
  let n = 0; for (const dz of [0]) { const zc = Math.floor(px[2] / ratio / 16) * 16, b = new Uint32Array(await (await fetch(`${u}/0/${x}-${x + 64}_${y}-${y + 64}_${zc}-${zc + 16}`)).arrayBuffer()); for (const v2 of b) if (v2) n++; }
  return { alpha: seg.selectedAlpha, layers: j.layers.map((l) => l.name), shader: img && img.shaderControls, selected: j.selectedLayer, nonzero: n, size: sc.size, model: document.getElementById('model').value, stain: document.getElementById('stain').value };
});
t0 = Date.now(); let r = await probe(B); console.log('B (from link, before regeneration finishes):', JSON.stringify(r), `(${((Date.now() - t0) / 1000).toFixed(0)} s)`);
await B.waitForTimeout(5000); await B.screenshot({ path: `${OUT}/link_B.png` });
await B.click('#regen'); await B.waitForTimeout(60000); r = await probe(B); console.log('B after Regenerate:', JSON.stringify(r)); await B.screenshot({ path: `${OUT}/link_B_regen.png` });
await browser.close(); srv.close();
