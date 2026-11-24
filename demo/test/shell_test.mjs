// The Neuroglancer shell page: sidebar selection, embedded Neuroglancer + hook, run a model on the whole stitched slab.
import { chromium } from 'playwright';
import http from 'http'; import fs from 'fs'; import path from 'path';
const ROOT = process.env.SITE, OUT = process.env.SHOT_DIR || '.', PORT = +(process.env.PORT || 8311);
const mime = { '.js': 'text/javascript', '.json': 'application/json', '.html': 'text/html', '.css': 'text/css', '.wasm': 'application/wasm', '.py': 'text/plain' };
const srv = http.createServer((q, r) => { let f = path.join(ROOT, q.url.split('?')[0].split('#')[0]); if (f.endsWith('/')) f += 'index.html';
  if (!fs.existsSync(f) || fs.statSync(f).isDirectory()) { r.writeHead(404); return r.end(); }
  r.writeHead(200, { 'content-type': mime[path.extname(f)] || 'application/octet-stream' }); r.end(fs.readFileSync(f)); }).listen(PORT);
const browser = await chromium.launch({ headless: true, args: ['--no-sandbox', '--enable-unsafe-webgpu', '--enable-features=Vulkan,UnsafeWebGPU', '--disable-vulkan-surface', '--use-vulkan=swiftshader', '--use-webgpu-adapter=swiftshader', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist', '--headless=new'] });
const page = await browser.newPage({ viewport: { width: 1500, height: 900 } });
page.on('pageerror', (e) => console.log('[pageerror]', e.message.slice(0, 200)));
await page.goto(`http://localhost:${PORT}/index.html`);
const ready = () => page.waitForFunction(() => { const h = window.__shell && window.__shell.hook(); return h && h.ready; }, null, { timeout: 150000 });
await ready(); await page.waitForTimeout(15000);
console.log('sidebar:', JSON.stringify(await page.evaluate(() => ({ stain: stain.value, sample: sample.value, session: session.selectedOptions[0].text, model: model.value, strips: document.querySelectorAll('#strip option').length }))).slice(0, 300));
await page.screenshot({ path: `${OUT}/shell_initial.png` });
await page.selectOption('#model', 'regions'); await page.click('#run');
const t0 = Date.now(); let st = '';
while (Date.now() - t0 < 280000) { await page.waitForTimeout(3000); st = await page.evaluate(() => window.__shell.hook().status); if (/segments|error|zoom in/.test(st)) break; }
console.log('regions:', st, `(${((Date.now() - t0) / 1000).toFixed(0)} s)`); await page.waitForTimeout(3000);
await page.screenshot({ path: `${OUT}/shell_regions.png` });
console.log('banner:', await page.textContent('#bn-line'));
await page.selectOption('#stain', 'LEC'); await page.waitForTimeout(3000);
console.log('LEC model options:', JSON.stringify(await page.evaluate(() => [...document.querySelectorAll('#model option')].map((o) => (o.disabled ? '[x] ' : '') + o.textContent))), '| selected:', await page.inputValue('#model'));
await browser.close(); srv.close();
