// Segmentation debouncing: many view events must produce few runs, none while dragging, one after settling.
import { chromium } from 'playwright';
import http from 'http'; import fs from 'fs'; import path from 'path';
const ROOT = process.env.DEMO_ROOT || path.resolve('demo'), ASSET = process.env.ASSET, PORT = +(process.env.PORT || 8161);
const mime = { '.js': 'text/javascript', '.json': 'application/json', '.html': 'text/html' };
const srv = http.createServer((q, r) => { const f = path.join(ROOT, q.url.split('?')[0].split('#')[0]);
  if (!fs.existsSync(f) || fs.statSync(f).isDirectory()) { r.writeHead(404); return r.end(); }
  r.writeHead(200, { 'content-type': mime[path.extname(f)] || 'application/octet-stream' }); r.end(fs.readFileSync(f)); }).listen(PORT);
const browser = await chromium.launch({ headless: true, args: ['--no-sandbox', '--enable-unsafe-webgpu', '--enable-features=Vulkan,UnsafeWebGPU', '--disable-vulkan-surface', '--use-vulkan=swiftshader', '--use-webgpu-adapter=swiftshader', '--use-angle=swiftshader', '--ignore-gpu-blocklist', '--headless=new'] });
const page = await browser.newPage({ viewport: { width: 1200, height: 800 } });
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
await page.goto(`http://localhost:${PORT}/viewer.html#id=${ASSET}`);
await page.waitForFunction(() => window.__hw && window.__hw.view, null, { timeout: 120000 });
await page.evaluate(() => { window.__hw.setMaxTiles(2); window.__hw.view.p = 3; window.__hw.schedule(0); });
await page.waitForFunction(() => window.__hw.counts.started >= 1, null, { timeout: 120000 });
const idle = async (ms = 60000) => { const t0 = Date.now(); await page.waitForTimeout(800); while (Date.now() - t0 < ms) { const st = await page.evaluate(() => window.__hw.status); if (/\d+\/\d+ tiles|FOV forward|manual|settle/.test(st) && !/settle/.test(st)) break; await page.waitForTimeout(500); } };
await idle(); const C = () => page.evaluate(() => ({ ...window.__hw.counts }));
const reset = () => page.evaluate(() => { Object.assign(window.__hw.counts, { scheduled: 0, started: 0, tiles: 0, fov: 0 }); });
const row = (name, c, extra = '') => console.log(`${name.padEnd(34)} scheduled ${String(c.scheduled).padStart(3)}  runs started ${c.started}  tiles ${c.tiles} ${extra}`);

await reset(); await page.mouse.move(600, 300); await page.mouse.down();
for (let i = 0; i < 40; i++) { await page.mouse.move(600 + i * 4, 300 + i * 2); await page.waitForTimeout(40); }
let c = await C(); row('during a 1.6 s drag (40 events)', c, c.started === 0 ? 'OK: none while dragging' : 'FAIL');
await page.mouse.up(); await page.waitForTimeout(250); c = await C(); row('250 ms after release (still settling)', c);
await idle(); c = await C(); row('after settling', c, c.started === 1 ? 'OK: exactly one run' : 'check');

await reset(); await page.mouse.move(600, 300);
for (let i = 0; i < 20; i++) { await page.mouse.wheel(0, i % 2 ? 40 : -40); await page.waitForTimeout(30); }
await idle(); c = await C(); row('wheel burst (20 events)', c, c.started <= 1 ? 'OK: coalesced' : 'FAIL');

await reset();
await page.evaluate(async () => { const e = document.getElementById('bri'); for (let i = 0; i < 15; i++) { e.value = -0.2 + i * 0.02; e.dispatchEvent(new Event('input')); await new Promise((r) => setTimeout(r, 35)); } });
await page.waitForTimeout(450); c = await C(); row('slider burst, 450 ms after last input', c, c.started === 0 ? 'OK: still waiting (longer settle for exposure)' : 'check');
await idle(); c = await C(); row('slider burst, after settling', c, c.started === 1 ? 'OK: one run' : 'check');

await reset(); await page.evaluate(() => { document.getElementById('segauto').checked = false; });
await page.mouse.move(600, 300); await page.mouse.down(); for (let i = 0; i < 10; i++) { await page.mouse.move(600 - i * 5, 300); await page.waitForTimeout(30); } await page.mouse.up(); await page.waitForTimeout(1200);
c = await C(); row('manual mode, after panning', c, c.started === 0 ? 'OK: nothing runs' : 'FAIL');
await page.click('#segrun'); await page.waitForFunction(() => window.__hw.counts.started >= 1, null, { timeout: 60000 }); await idle(); c = await C(); row('manual mode, Run segmentation now', c, c.started === 1 ? 'OK' : 'check');
await browser.close(); srv.close();
