import { chromium } from 'playwright';
import http from 'http'; import fs from 'fs'; import path from 'path';
const ROOT = process.env.DEMO_ROOT || path.resolve('demo'), OUT = process.env.SHOT_DIR || '.';
const mime = { '.js': 'text/javascript', '.json': 'application/json', '.html': 'text/html' };
const srv = http.createServer((q, r) => { const f = path.join(ROOT, q.url.split('?')[0] === '/' ? '/index.html' : q.url.split('?')[0]);
  if (!fs.existsSync(f)) { r.writeHead(404); return r.end(); }
  r.writeHead(200, { 'content-type': mime[path.extname(f)] || 'application/octet-stream' }); r.end(fs.readFileSync(f)); }).listen(8124);
const browser = await chromium.launch({ headless: true, args: ['--no-sandbox', '--enable-unsafe-webgpu', '--enable-features=Vulkan,UnsafeWebGPU', '--disable-vulkan-surface', '--use-vulkan=swiftshader', '--use-webgpu-adapter=swiftshader', '--use-angle=swiftshader', '--ignore-gpu-blocklist', '--headless=new'] });
const page = await browser.newPage({ viewport: { width: 1200, height: 800 } });
page.on('console', (m) => { const t = m.text(); if (!/GPU stall|swiftshader/i.test(t)) console.log('[page]', t.slice(0, 200)); });
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
await page.goto('http://localhost:8124/');
const waitStatus = async (re, ms = 150000) => { const t0 = Date.now(); for (;;) { const s = await page.evaluate(() => window.__hw && window.__hw.status); if (s && re.test(s)) return s; if (Date.now() - t0 > ms) return 'TIMEOUT: ' + s; await page.waitForTimeout(500); } };
const step = async (name, setup, re, want) => { await page.evaluate(setup); await page.evaluate(() => window.__hw.schedule()); await page.waitForTimeout(1500); const t0 = Date.now(); while (Date.now() - t0 < 60000 && (await page.evaluate(() => window.__hw.route && window.__hw.route.model)) !== want) await page.waitForTimeout(300); const s = await waitStatus(re, 90000); const r = await page.evaluate(() => window.__hw.route); console.log(name, '=>', JSON.stringify({ model: r && r.model, level: r && r.level }), '|', s); await page.screenshot({ path: `${OUT}/${name}.png` }); };
console.log('initial:', await waitStatus(/tiles|Raw|zoom/));
await step('1_cells_p3', () => { window.__hw.view.p = 3; }, /(\d+)\/\1 tiles|failed/, 'cells');
await step('2_tissue_p60', () => { window.__hw.view.p = 60; }, /^tissue: (\d+)\/\1 tiles|failed/, 'tissue');
await browser.close(); srv.close();
