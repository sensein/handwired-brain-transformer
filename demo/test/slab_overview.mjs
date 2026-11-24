// Whole stitched slab at overview zoom (all strips), with the status banner expanded.
import { chromium } from 'playwright';
import http from 'http'; import fs from 'fs'; import path from 'path';
const ROOT = process.env.DEMO_ROOT || path.resolve('demo'), OUT = process.env.SHOT_DIR || '.', ASSET = process.env.ASSET, PORT = +(process.env.PORT || 8191);
const mime = { '.js': 'text/javascript', '.json': 'application/json', '.html': 'text/html' };
const srv = http.createServer((q, r) => { const f = path.join(ROOT, q.url.split('?')[0].split('#')[0]);
  if (!fs.existsSync(f) || fs.statSync(f).isDirectory()) { r.writeHead(404); return r.end(); }
  r.writeHead(200, { 'content-type': mime[path.extname(f)] || 'application/octet-stream' }); r.end(fs.readFileSync(f)); }).listen(PORT);
const browser = await chromium.launch({ headless: true, args: ['--no-sandbox', '--enable-unsafe-webgpu', '--enable-features=Vulkan,UnsafeWebGPU', '--disable-vulkan-surface', '--use-vulkan=swiftshader', '--use-webgpu-adapter=swiftshader', '--use-angle=swiftshader', '--ignore-gpu-blocklist', '--headless=new'] });
const page = await browser.newPage({ viewport: { width: 1500, height: 1000 } });
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
page.on('console', (m) => { if (m.type() === 'error') console.log('[console.error]', m.text().slice(0, 200)); });
await page.goto(`http://localhost:${PORT}/index.html#id=${ASSET}`);
await page.waitForFunction(() => window.__hw && window.__hw.view, null, { timeout: 180000 });
await page.click('#bn-toggle');
const t0 = Date.now(); let st = '';
while (Date.now() - t0 < 150000) { await page.waitForTimeout(2000); st = await page.evaluate(() => window.__hw.status); if (/FOV forward/.test(st)) break; }
if (process.env.SEAMS) { await page.check('#seams'); await page.waitForTimeout(45000); console.log('seams banner log:', (await page.textContent('#bn-log')).split('\n').filter((l) => /seam/i.test(l)).slice(-2).join(' | ')); }
await page.waitForTimeout(3000);
const info = await page.evaluate(() => ({ levels: window.__hw.zarr ? window.__hw.zarr.levels.length : null, view: { ...window.__hw.view }, banner: document.getElementById('bn-line').textContent, meta: document.getElementById('bn-meta').textContent, sel: [...document.querySelectorAll('#chunk option')][0].textContent }));
console.log(JSON.stringify(info)); console.log('status:', st);
await page.screenshot({ path: `${OUT}/slab_overview.png` });
await browser.close(); srv.close();
