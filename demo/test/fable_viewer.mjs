// Hand-written (Fable) scale-invariant programs in the viewer: forced model, a few zooms, screenshots + class stats.
import { chromium } from 'playwright';
import http from 'http'; import fs from 'fs'; import path from 'path';
const ROOT = process.env.DEMO_ROOT || path.resolve('demo'), OUT = process.env.SHOT_DIR || '.', ASSET = process.env.ASSET, MODEL = process.env.MODEL || 'fable_blobs', PORT = +(process.env.PORT || 8221);
const mime = { '.js': 'text/javascript', '.json': 'application/json', '.html': 'text/html' };
const srv = http.createServer((q, r) => { const f = path.join(ROOT, q.url.split('?')[0].split('#')[0]);
  if (!fs.existsSync(f) || fs.statSync(f).isDirectory()) { r.writeHead(404); return r.end(); }
  r.writeHead(200, { 'content-type': mime[path.extname(f)] || 'application/octet-stream' }); r.end(fs.readFileSync(f)); }).listen(PORT);
const browser = await chromium.launch({ headless: true, args: ['--no-sandbox', '--enable-unsafe-webgpu', '--enable-features=Vulkan,UnsafeWebGPU', '--disable-vulkan-surface', '--use-vulkan=swiftshader', '--use-webgpu-adapter=swiftshader', '--use-angle=swiftshader', '--ignore-gpu-blocklist', '--headless=new'] });
const page = await browser.newPage({ viewport: { width: 1100, height: 900 } });
page.on('pageerror', (e) => console.log('[pageerror]', e.message.slice(0, 200)));
await page.goto(`http://localhost:${PORT}/viewer.html#slab=0&id=${ASSET}`);
await page.waitForFunction(() => window.__hw && window.__hw.view, null, { timeout: 120000 });
await page.selectOption('#model', MODEL);
for (const p of (process.env.ZOOMS || '6,3,1.5').split(',').map(Number)) {
  await page.evaluate((p) => { window.__hw.view.p = p; window.__hw.resetSeg(); window.__hw.schedule(0); }, p);
  const t0 = Date.now(); let st = '';
  while (Date.now() - t0 < 200000) { await page.waitForTimeout(1500); st = await page.evaluate(() => window.__hw.status); if (/FOV forward|failed|outside/.test(st) && !/settle/.test(st)) break; }
  const s = await page.evaluate(() => window.__hw.segSummary());
  console.log(`p ${p}: ${st.slice(0, 110)} | ${JSON.stringify(s && Object.fromEntries(Object.entries(s).map(([k, v]) => [k, typeof v === 'number' ? +v.toFixed(3) : v])))}`);
  await page.locator('#cv').screenshot({ path: `${OUT}/fable_${MODEL}_p${p}.png` });
}
await browser.close(); srv.close();
