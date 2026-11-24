import { chromium } from 'playwright';
import http from 'http'; import fs from 'fs'; import path from 'path';
const ROOT = process.env.DEMO_ROOT || path.resolve('demo'), OUT = process.env.SHOT_DIR || '.';
const mime = { '.js': 'text/javascript', '.json': 'application/json', '.html': 'text/html' };
const srv = http.createServer((q, r) => { const f = path.join(ROOT, q.url.split('?')[0].split('#')[0]);
  if (!fs.existsSync(f) || fs.statSync(f).isDirectory()) { r.writeHead(404); return r.end(); }
  r.writeHead(200, { 'content-type': mime[path.extname(f)] || 'application/octet-stream' }); r.end(fs.readFileSync(f)); }).listen(8125);
const browser = await chromium.launch({ headless: true, args: ['--no-sandbox', '--enable-unsafe-webgpu', '--enable-features=Vulkan,UnsafeWebGPU', '--disable-vulkan-surface', '--use-vulkan=swiftshader', '--use-webgpu-adapter=swiftshader', '--use-angle=swiftshader', '--ignore-gpu-blocklist', '--headless=new'] });
const page = await browser.newPage({ viewport: { width: 1300, height: 700 } });
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
await page.goto('http://localhost:8125/slabs.html');
const want = +(process.env.WANT || 6);
for (const lv of (process.env.LEVELS || '5,4,0').split(',')) {
  await page.selectOption('#level', lv);
  const t0 = Date.now(); let n = 0;
  while (Date.now() - t0 < 170000) { n = await page.evaluate(() => window.__slabs.done); if (n >= want) break; await page.waitForTimeout(1000); }
  const stats = await page.evaluate(() => window.__slabs.stats.slice(0, 4));
  console.log(`level ${lv}: ${n} cells in ${((Date.now() - t0) / 1000).toFixed(0)}s`, JSON.stringify(stats.map((s) => ({ ...s, zarr: undefined }))).slice(0, 700));
  await page.screenshot({ path: `${OUT}/slabs_level${lv}.png` });
}
await browser.close(); srv.close();
