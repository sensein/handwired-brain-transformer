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
const errs = []; page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') errs.push(m.type() + ': ' + m.text().slice(0, 200)); });
const stat = () => page.evaluate(() => [...document.querySelectorAll('.neuroglancer-status-message')].map((e) => e.textContent.trim().slice(0, 160)));
await page.evaluate(() => window.__hwNg.setAuto(true, 'regions'));
const waitRun = async (tag) => { const t0 = Date.now(); let st = ''; while (Date.now() - t0 < 240000) { await page.waitForTimeout(2000); st = await page.evaluate(() => window.__hwNg.status); if (/segments|error|zoom in/.test(st)) break; } console.log(tag, st, '| status msgs:', JSON.stringify(await stat())); };
await waitRun('run1:');
for (let k = 2; k <= 4; k++) {   // pan: the continuous mode re-runs after the view settles
  await page.mouse.move(400, 400); await page.mouse.down(); await page.mouse.move(400 - 120 * k, 400, { steps: 8 }); await page.mouse.up();   // pan with the mouse like a user
  await page.waitForTimeout(3000); await waitRun('run' + k + ':');
}
console.log('errors:', JSON.stringify(errs.slice(0, 8)));
await page.screenshot({ path: `${OUT}/auto_after.png` });
await browser.close(); srv.close();
