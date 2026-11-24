// "Open this view in Neuroglancer": the stitched slab becomes a multi-source layer with our shader, position and zoom.
import { chromium } from 'playwright';
import http from 'http'; import fs from 'fs'; import path from 'path';
const ROOT = process.env.SITE, OUT = process.env.SHOT_DIR || '.', PORT = +(process.env.PORT || 8301), ASSET = process.env.ASSET;
const mime = { '.js': 'text/javascript', '.json': 'application/json', '.html': 'text/html', '.css': 'text/css', '.wasm': 'application/wasm', '.py': 'text/plain' };
const srv = http.createServer((q, r) => { let f = path.join(ROOT, q.url.split('?')[0].split('#')[0]); if (f.endsWith('/')) f += 'index.html';
  if (!fs.existsSync(f) || fs.statSync(f).isDirectory()) { r.writeHead(404); return r.end(); }
  r.writeHead(200, { 'content-type': mime[path.extname(f)] || 'application/octet-stream' }); r.end(fs.readFileSync(f)); }).listen(PORT);
const browser = await chromium.launch({ headless: true, args: ['--no-sandbox', '--enable-unsafe-webgpu', '--enable-features=Vulkan,UnsafeWebGPU', '--disable-vulkan-surface', '--use-vulkan=swiftshader', '--use-webgpu-adapter=swiftshader', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist', '--headless=new'] });
const ctx = await browser.newContext({ viewport: { width: 1300, height: 850 } });
const page = await ctx.newPage(); page.on('pageerror', (e) => console.log('[pageerror]', e.message.slice(0, 200)));
await page.goto(`http://localhost:${PORT}/index.html#id=${ASSET}`);
await page.waitForFunction(() => window.__hw && window.__hw.view && window.__hw.zarr, null, { timeout: 180000 });
await page.waitForTimeout(25000);                      // overview + exposure
await page.screenshot({ path: `${OUT}/link_viewer.png` });
await page.evaluate(() => { window.__opened = null; window.open = (u) => { window.__opened = u; return null; }; });   // headless browsers block popups: capture the URL
await page.click('#ngopen'); await page.waitForTimeout(1500);
const opened = await page.evaluate(() => window.__opened); if (!opened) throw new Error('the button did not call window.open');
const st0 = JSON.parse(decodeURIComponent(opened.split('#!')[1]));
console.log('state: sources', Array.isArray(st0.layers[0].source) ? st0.layers[0].source.length : 1, '| position', st0.position.map((x) => +x.toFixed(0)), '| scale', +st0.crossSectionScale.toFixed(2), '| shaderControls', JSON.stringify(st0.layers[0].shaderControls));
const pop = await ctx.newPage(); pop.on('pageerror', (e) => console.log('[popup pageerror]', e.message.slice(0, 160))); await pop.goto(`http://localhost:${PORT}/` + opened);
await pop.waitForFunction(() => window.viewer, null, { timeout: 90000 }); await pop.waitForTimeout(30000);
console.log('popup viewer layers:', await pop.evaluate(() => window.viewer.state.toJSON().layers.map((l) => l.name + ':' + l.type)));
await pop.screenshot({ path: `${OUT}/link_ng.png` });
if (process.env.REGIONS) {
  await pop.selectOption('#hw-m', 'regions'); await pop.click('#hw-run');
  const t0 = Date.now(); let st = '';
  while (Date.now() - t0 < 240000) { await pop.waitForTimeout(2000); st = await pop.evaluate(() => window.__hwNg.status); if (/segments|error/.test(st)) break; }
  console.log('regions status:', st, `(${((Date.now() - t0) / 1000).toFixed(0)} s)`); console.log(JSON.stringify(await pop.evaluate(() => window.__hwNg.last)));
  await pop.waitForTimeout(3000); await pop.screenshot({ path: `${OUT}/link_ng_regions.png` });
}
await browser.close(); srv.close();
