import { chromium } from 'playwright';
import http from 'http'; import fs from 'fs'; import path from 'path';
const ROOT = process.env.DEMO_ROOT || path.resolve('demo');
const mime = { '.js': 'text/javascript', '.json': 'application/json', '.html': 'text/html' };
const srv = http.createServer((q, r) => { const f = path.join(ROOT, q.url.split('?')[0]);
  if (!fs.existsSync(f)) { r.writeHead(404); return r.end(); }
  r.writeHead(200, { 'content-type': mime[path.extname(f)] || 'application/octet-stream' }); r.end(fs.readFileSync(f)); }).listen(8123);
const flags = (process.argv[2] || 'swiftshader').split(',');
const args = ['--no-sandbox', '--enable-unsafe-webgpu', '--enable-features=Vulkan,UnsafeWebGPU', '--disable-vulkan-surface',
  ...(flags.includes('swiftshader') ? ['--use-vulkan=swiftshader', '--use-webgpu-adapter=swiftshader', '--use-angle=swiftshader'] : []),
  '--ignore-gpu-blocklist', '--headless=new'];
const browser = await chromium.launch({ args, headless: true });
const page = await browser.newPage();
page.on('console', (m) => console.log('[page]', m.text()));
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
await page.goto('http://localhost:8123/test/blank.html');
const res = await page.evaluate(async () => {
  const E = await import('/engine.js');
  let eng; try { eng = await E.GPUEngine.create(); } catch (e) { return { error: String(e) }; }
  const gold = await (await fetch('/test/golden.json')).json();
  const out = {};
  for (const [name, g] of Object.entries(gold)) {
    const prog = E.parseProgram(await (await fetch(`/models/${name}.json`)).text());
    const [Z, Y, X] = g.dims, zc = name === 'tissue' ? 6 : name === 'layers' ? 4 : 7;
    const r = await eng.run(prog, Float32Array.from(g.logI), g.dims, g.spacing, zc);
    for (const [k, ref] of Object.entries(g.out)) {
      let md = 0, mx = 0;
      for (let i = 0; i < Y * X; i++) { const rr = ref[zc * Y * X + i]; md = Math.max(md, Math.abs(r[k][i] - rr)); mx = Math.max(mx, Math.abs(rr)); }
      out[`${name}.${k}`] = { maxdiff: md, refmax: mx };
    }
  }
  return out;
});
console.log(JSON.stringify(res, null, 1));
await browser.close(); srv.close();
