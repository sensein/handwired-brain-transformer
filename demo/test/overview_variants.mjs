// Whole-slab overview (zoomed out): exposure variants, tissue forward pass over the whole visible slab, screenshots of the canvas.
import { chromium } from 'playwright';
import http from 'http'; import fs from 'fs'; import path from 'path';
const ROOT = process.env.DEMO_ROOT || path.resolve('demo'), OUT = process.env.SHOT_DIR || '.', ASSET = process.env.ASSET, PORT = +(process.env.PORT || 8171);
const mime = { '.js': 'text/javascript', '.json': 'application/json', '.html': 'text/html' };
const srv = http.createServer((q, r) => { const f = path.join(ROOT, q.url.split('?')[0].split('#')[0]);
  if (!fs.existsSync(f) || fs.statSync(f).isDirectory()) { r.writeHead(404); return r.end(); }
  r.writeHead(200, { 'content-type': mime[path.extname(f)] || 'application/octet-stream' }); r.end(fs.readFileSync(f)); }).listen(PORT);
const browser = await chromium.launch({ headless: true, args: ['--no-sandbox', '--enable-unsafe-webgpu', '--enable-features=Vulkan,UnsafeWebGPU', '--disable-vulkan-surface', '--use-vulkan=swiftshader', '--use-webgpu-adapter=swiftshader', '--use-angle=swiftshader', '--ignore-gpu-blocklist', '--headless=new'] });
const page = await browser.newPage({ viewport: { width: 1500, height: 900 } });
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
await page.goto(`http://localhost:${PORT}/viewer.html#id=${ASSET}`);
await page.waitForFunction(() => window.__hw && window.__hw.view, null, { timeout: 120000 });
// frame the whole slab: centre on the volume, p large enough that the long axis fits
const info = await page.evaluate(() => { const v = window.__hw.view; return { z: v.z }; });
await page.evaluate(() => { const h = window.__hw; h.view.p = 160; h.view.cx = 0; });
const variants = [['default', 0, 1, 1], ['gamma0.5', 0, 1, .5], ['gamma2', 0, 1, 2], ['contrast0.5', 0, .5, 1], ['contrast2', 0, 2, 1], ['bright+0.3', .3, 1, 1], ['bright-0.3', -.3, 1, 1], ['flat c0.1', 0, .1, 1], ['dark b-0.5', -.5, 1, 1]];
for (const [name, b, c, g] of variants) {
  await page.evaluate(([b, c, g]) => { for (const [id, v] of [['bri', b], ['con', c], ['gam', g]]) { const e = document.getElementById(id); e.value = v; e.dispatchEvent(new Event('input')); } window.__hw.resetSeg(); window.__hw.schedule(0); }, [b, c, g]);
  const t0 = Date.now(); let s = null;
  while (Date.now() - t0 < 120000) { await page.waitForTimeout(1200); const st = await page.evaluate(() => window.__hw.status); s = await page.evaluate(() => window.__hw.segSummary()); if (/FOV forward/.test(st) && s) break; }
  const rec = await page.evaluate(() => { const d = [...document.querySelectorAll('#info dt')].find((x) => /Exposure/.test(x.textContent)); return d ? d.nextSibling.textContent : ''; });
  console.log(name.padEnd(12), s ? `${s.model} L${s.level}  bg ${(s.bg * 100).toFixed(0)}% gray ${(s.gray * 100).toFixed(0)}% white ${(s.white * 100).toFixed(0)}%` : 'no result', '|', rec);
  await page.locator('#cv').screenshot({ path: `${OUT}/overview_${name.replace(/[ +.]/g, '_')}.png` });
}
await browser.close(); srv.close();
