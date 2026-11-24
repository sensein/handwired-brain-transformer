// Drive the real viewer through zoom levels and brightness/contrast/gamma settings; record what each program picks up.
import { chromium } from 'playwright';
import http from 'http'; import fs from 'fs'; import path from 'path';
const ROOT = process.env.DEMO_ROOT || path.resolve('demo'), OUT = process.env.SHOT_DIR || '.';
const ASSET = process.env.ASSET, STAIN = process.env.STAIN || 'YO', TILES = +(process.env.TILES || 6);
const mime = { '.js': 'text/javascript', '.json': 'application/json', '.html': 'text/html' };
const srv = http.createServer((q, r) => { const f = path.join(ROOT, q.url.split('?')[0].split('#')[0]);
  if (!fs.existsSync(f) || fs.statSync(f).isDirectory()) { r.writeHead(404); return r.end(); }
  r.writeHead(200, { 'content-type': mime[path.extname(f)] || 'application/octet-stream' }); r.end(fs.readFileSync(f)); }).listen(+(process.env.PORT || 8126));
const browser = await chromium.launch({ headless: true, args: ['--no-sandbox', '--enable-unsafe-webgpu', '--enable-features=Vulkan,UnsafeWebGPU', '--disable-vulkan-surface', '--use-vulkan=swiftshader', '--use-webgpu-adapter=swiftshader', '--use-angle=swiftshader', '--ignore-gpu-blocklist', '--headless=new'] });
const page = await browser.newPage({ viewport: { width: 1100, height: 700 } });
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
await page.goto(`http://localhost:${process.env.PORT || 8126}/viewer.html#id=${ASSET}`);
await page.waitForFunction(() => window.__hw && window.__hw.view, null, { timeout: 120000 });
await page.evaluate((n) => window.__hw.setMaxTiles(n), TILES);
const rec = [];
async function measure(name, setup) {
  await page.evaluate(setup);
  await page.evaluate(() => { window.__hw.resetSeg(); window.__hw.draw(); window.__hw.schedule(); });
  const t0 = Date.now(); let s = null;
  while (Date.now() - t0 < 150000) { await page.waitForTimeout(1500); s = await page.evaluate(() => window.__hw.segSummary()); const st = await page.evaluate(() => window.__hw.status); if (s && s.tiles >= 1 && /(\d+)\/\1 tiles|budget|zoom in|FOV forward|outside the volume/.test(st)) break; }
  const row = { name, ...(s || { model: 'none' }), adj: await page.evaluate(() => ({ ...window.__hw.adj })), status: await page.evaluate(() => window.__hw.status) };
  rec.push(row); console.log(JSON.stringify(row).slice(0, 400));
  await page.screenshot({ path: `${OUT}/eval_${STAIN}_${name}.png` });
}
const setP = (p) => `window.__hw.view.p = ${p}`;
const setAdj = (b, c, g) => `(() => { for (const [id, v] of [['bri', ${b}], ['con', ${c}], ['gam', ${g}]]) { const e = document.getElementById(id); e.value = v; e.dispatchEvent(new Event('input')); } })()`;
for (const p of [160, 60, 25, 12, 6, 3]) await measure(`p${p}`, setP(p));
for (const [nm, b, c, g] of [['gamma0.5', 0, 1, .5], ['gamma2', 0, 1, 2], ['contrast2', 0, 2, 1], ['bright+.15', .15, 1, 1], ['bright-.15', -.15, 1, 1]]) {
  for (const p of [60, 3]) await measure(`${nm}_p${p}`, `${setP(p)}; ${setAdj(b, c, g)}`);
}
fs.writeFileSync(`${OUT}/eval_${STAIN}.json`, JSON.stringify(rec, null, 1));
await browser.close(); srv.close();
