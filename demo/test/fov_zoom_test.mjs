// Zooming in must shrink the field of view (finer level, fewer voxels); zooming out must grow it (coarser level).
// The model always sees exactly the fetched FOV (one forward pass) when it fits in device memory.
import { chromium } from 'playwright';
import http from 'http'; import fs from 'fs'; import path from 'path';
const ROOT = process.env.DEMO_ROOT || path.resolve('demo'), ASSET = process.env.ASSET, PORT = +(process.env.PORT || 8181);
const VW = +(process.env.VW || 700), VH = +(process.env.VH || 760);
const mime = { '.js': 'text/javascript', '.json': 'application/json', '.html': 'text/html' };
const srv = http.createServer((q, r) => { const f = path.join(ROOT, q.url.split('?')[0].split('#')[0]);
  if (!fs.existsSync(f) || fs.statSync(f).isDirectory()) { r.writeHead(404); return r.end(); }
  r.writeHead(200, { 'content-type': mime[path.extname(f)] || 'application/octet-stream' }); r.end(fs.readFileSync(f)); }).listen(PORT);
const browser = await chromium.launch({ headless: true, args: ['--no-sandbox', '--enable-unsafe-webgpu', '--enable-features=Vulkan,UnsafeWebGPU', '--disable-vulkan-surface', '--use-vulkan=swiftshader', '--use-webgpu-adapter=swiftshader', '--use-angle=swiftshader', '--ignore-gpu-blocklist', '--headless=new'] });
const page = await browser.newPage({ viewport: { width: VW, height: VH } });
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
await page.goto(`http://localhost:${PORT}/viewer.html#id=${ASSET}`);
await page.waitForFunction(() => window.__hw && window.__hw.view, null, { timeout: 120000 });
await page.evaluate(() => window.__hw.setMaxTiles(2));
const canvasPx = await page.evaluate(() => [document.getElementById('cv').width, document.getElementById('cv').height]);
console.log('canvas', canvasPx.join('x'), 'px');
for (const p of (process.env.ZOOMS || '160,60,25,12,6,3,1.5,0.75').split(',').map(Number)) {
  await page.evaluate((p) => { window.__hw.view.p = p; window.__hw.resetSeg(); window.__hw.schedule(0); }, p);
  const t0 = Date.now(); let st = '';
  while (Date.now() - t0 < 120000) { await page.waitForTimeout(1000); st = await page.evaluate(() => window.__hw.status); if (/FOV forward|tiles|outside/.test(st) && !/settle/.test(st)) break; }
  const r = await page.evaluate(() => { const x = window.__hw.route; return x && { model: x.model, level: x.level }; });
  const m = st.match(/\((\d+)x(\d+) voxels\)/);
  console.log(`p ${String(p).padStart(6)} um/px  FOV ${(canvasPx[0] * p / 1000).toFixed(2)} x ${(canvasPx[1] * p / 1000).toFixed(2)} mm  ${String(r && r.model).padEnd(14)} L${r && r.level}  ${m ? `fetched ${m[1]}x${m[2]} voxels (one pass)` : 'tiles: ' + st.slice(0, 90)}`);
}
await browser.close(); srv.close();
