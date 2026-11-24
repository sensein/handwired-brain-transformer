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
await page.selectOption('#hw-m', process.env.MODEL || 'cells'); if (process.env.DEPTH) await page.evaluate((d) => window.__hwNg.setDepth(d), process.env.DEPTH); await page.click('#hw-run');
const t0 = Date.now(); let st = '';
while (Date.now() - t0 < 720000) { await page.waitForTimeout(2000); st = await page.evaluate(() => window.__hwNg.status); if (/detections|segments|detections|error/.test(st)) break; }
console.log('status:', st, `(${((Date.now() - t0) / 1000).toFixed(0)} s)`);
const res = await page.evaluate(() => { const j = window.viewer.state.toJSON(), l = (j.layers || []).find((x) => x.name === 'hwseg detections'); const img = j.layers.find((x) => x.type === 'image'); return { shaderControls: img && img.shaderControls, position: j.position, scale: j.crossSectionScale, last: window.__hwNg.last, layers: j.layers.map((x) => x.name + ':' + x.type + (x.annotations ? '(' + x.annotations.length + ')' : '')), displayDimensions: j.displayDimensions, first: l && l.annotations && l.annotations[0] }; });
console.log(JSON.stringify(res));
await page.waitForTimeout(2500); await page.screenshot({ path: `${OUT}/ng_after.png` });
await browser.close(); srv.close();
