// Python lab: Pyodide builds weights, WebGPU runs them on a streamed DANDI window; editing the Python changes the result.
import { chromium } from 'playwright';
import http from 'http'; import fs from 'fs'; import path from 'path';
const ROOT = process.env.DEMO_ROOT || path.resolve('demo'), OUT = process.env.SHOT_DIR || '.', PORT = +(process.env.PORT || 8231);
const mime = { '.js': 'text/javascript', '.json': 'application/json', '.html': 'text/html', '.py': 'text/plain' };
const srv = http.createServer((q, r) => { const f = path.join(ROOT, q.url.split('?')[0].split('#')[0]);
  if (!fs.existsSync(f) || fs.statSync(f).isDirectory()) { r.writeHead(404); return r.end(); }
  r.writeHead(200, { 'content-type': mime[path.extname(f)] || 'application/octet-stream' }); r.end(fs.readFileSync(f)); }).listen(PORT);
const browser = await chromium.launch({ headless: true, args: ['--no-sandbox', '--enable-unsafe-webgpu', '--enable-features=Vulkan,UnsafeWebGPU', '--disable-vulkan-surface', '--use-vulkan=swiftshader', '--use-webgpu-adapter=swiftshader', '--use-angle=swiftshader', '--ignore-gpu-blocklist', '--headless=new'] });
const page = await browser.newPage({ viewport: { width: 1200, height: 800 } });
page.on('pageerror', (e) => console.log('[pageerror]', e.message.slice(0, 200)));
await page.goto(`http://localhost:${PORT}/lab.html`);
const expected = JSON.parse(fs.readFileSync(path.join(ROOT, 'models/index.json'))).cells.nonzero;
async function go(label, code) {
  if (code) await page.fill('#code', code);
  const t0 = Date.now(); await page.click('#run');
  await page.waitForFunction(() => /WebGPU ran it|must define|Error|error/.test(document.getElementById('out').textContent), null, { timeout: 280000 });
  const out = await page.textContent('#out'), stat = await page.textContent('#stat');
  console.log(`[${label}] ${((Date.now() - t0) / 1000).toFixed(1)} s\n   ${out.replace(/\n/g, '\n   ')}\n   ${stat}`);
  await page.screenshot({ path: `${OUT}/lab_${label}.png` });
}
await go('default');
console.log('   (builder-exported cells.json has', expected, 'nonzero params)');
await go('scale2', `from hwseg.programs import cells\nprog = cells(sigma_um=3.5, kappa=2.0, tau0=0.03, scale=2)\n`);
await go('error', `prog_ = 1\n`);
await browser.close(); srv.close();
