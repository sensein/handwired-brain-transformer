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
const [pop] = await Promise.all([ctx.waitForEvent('popup'), page.click('#ngopen')]);
const url = pop.url(), st = JSON.parse(decodeURIComponent(url.split('#!')[1]));
console.log('state: sources', Array.isArray(st.layers[0].source) ? st.layers[0].source.length : 1, '| layer', st.layers[0].name, '| position', st.position.map((x) => +x.toFixed(0)), '| scale', +st.crossSectionScale.toFixed(2), '| shaderControls', JSON.stringify(st.layers[0].shaderControls), '| displayDimensions', st.displayDimensions);
await pop.waitForFunction(() => window.viewer, null, { timeout: 90000 }); await pop.waitForTimeout(30000);
console.log('popup viewer layers:', await pop.evaluate(() => window.viewer.state.toJSON().layers.map((l) => l.name + ':' + l.type)));
await pop.screenshot({ path: `${OUT}/link_ng.png` });
await browser.close(); srv.close();
