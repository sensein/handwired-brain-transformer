// Depth navigation (keys, Shift+wheel), slab projection, exposure controls, 3D renderer with and without a segmentation volume.
import { chromium } from 'playwright';
import http from 'http'; import fs from 'fs'; import path from 'path';
const ROOT = process.env.DEMO_ROOT || path.resolve('demo'), OUT = process.env.SHOT_DIR || '.', ASSET = process.env.ASSET;
const mime = { '.js': 'text/javascript', '.json': 'application/json', '.html': 'text/html' };
const srv = http.createServer((q, r) => { const f = path.join(ROOT, q.url.split('?')[0].split('#')[0]);
  if (!fs.existsSync(f) || fs.statSync(f).isDirectory()) { r.writeHead(404); return r.end(); }
  r.writeHead(200, { 'content-type': mime[path.extname(f)] || 'application/octet-stream' }); r.end(fs.readFileSync(f)); }).listen(+(process.env.PORT || 8151));
const browser = await chromium.launch({ headless: true, args: ['--no-sandbox', '--enable-unsafe-webgpu', '--enable-features=Vulkan,UnsafeWebGPU', '--disable-vulkan-surface', '--use-vulkan=swiftshader', '--use-webgpu-adapter=swiftshader', '--use-angle=swiftshader', '--ignore-gpu-blocklist', '--headless=new'] });
const page = await browser.newPage({ viewport: { width: 1200, height: 1500 } });
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
await page.goto(`http://localhost:${process.env.PORT || 8151}/viewer.html#id=${ASSET}`);
await page.waitForFunction(() => window.__hw && window.__hw.view, null, { timeout: 120000 });
const z = () => page.evaluate(() => window.__hw.view.z);
const z0 = await z(); await page.evaluate(() => { window.__hw.view.p = 3; window.__hw.draw(); window.__hw.schedule(); }); await page.waitForTimeout(2500);
await page.keyboard.press('PageUp'); const z1 = await z(); await page.keyboard.press('Shift+PageDown'); const z2 = await z();
await page.mouse.move(400, 300); await page.keyboard.down('Shift'); await page.mouse.wheel(0, -100); await page.keyboard.up('Shift'); const z3 = await z();
console.log('depth: start', z0, 'PageUp', z1, 'Shift+PageDown', z2, 'Shift+wheel up', z3);
await page.selectOption('#depth', '7'); await page.selectOption('#dmode', 'max'); await page.waitForTimeout(4000);
console.log('hud:', await page.evaluate(() => document.getElementById('hud').textContent));
await page.screenshot({ path: `${OUT}/smoke_depth.png` });
console.log('exposure info:', await page.evaluate(() => [...document.querySelectorAll('#info dt')].map((d) => d.textContent + ': ' + d.nextSibling.textContent).filter((t) => /Exposure|Scale|Program/.test(t)).join(' | ')));
await page.selectOption('#v3size', '96'); await page.selectOption('#v3depth', '16');
for (const seg of [false, true]) {
  await page.evaluate((s) => { document.getElementById('v3seg').checked = s; }, seg);
  await page.click('#v3go');
  const t0 = Date.now(); let st = '';
  while (Date.now() - t0 < 240000) { st = await page.evaluate(() => document.getElementById('v3status').textContent); if (/^3D:|failed|unavailable/.test(st)) break; await page.waitForTimeout(1000); }
  console.log(`3D seg=${seg}:`, st.slice(0, 260));
  for (const mode of ['1', '0', '2']) { await page.selectOption('#v3mode', mode); await page.waitForTimeout(600); await page.locator('#v3cv').screenshot({ path: `${OUT}/smoke3d_seg${seg}_mode${mode}.png` }); }
}
await browser.close(); srv.close();
