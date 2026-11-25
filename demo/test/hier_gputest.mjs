// WebGPU (SwiftShader by default) vs Python golden for hier1 programs, in headless Chromium via Playwright.
// Run from a directory with playwright installed: DEMO_ROOT=<repo>/demo node hier_gputest.mjs [swiftshader|hw]
import { chromium } from 'playwright';
import http from 'http'; import fs from 'fs'; import path from 'path';
const ROOT = process.env.DEMO_ROOT || path.resolve('demo');
const mime = { '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json', '.html': 'text/html' };
const srv = http.createServer((q, r) => { const f = path.join(ROOT, q.url.split('?')[0]);
  if (!fs.existsSync(f)) { r.writeHead(404); return r.end(); }
  r.writeHead(200, { 'content-type': mime[path.extname(f)] || 'application/octet-stream' }); r.end(fs.readFileSync(f)); }).listen(8124);
const flags = (process.argv[2] || 'swiftshader').split(',');
const args = ['--no-sandbox', '--enable-unsafe-webgpu', '--enable-features=Vulkan,UnsafeWebGPU', '--disable-vulkan-surface',
  ...(flags.includes('swiftshader') ? ['--use-vulkan=swiftshader', '--use-webgpu-adapter=swiftshader', '--use-angle=swiftshader'] : []),
  '--ignore-gpu-blocklist', '--headless=new'];
const browser = await chromium.launch({ args, headless: true });
const page = await browser.newPage();
page.on('console', (m) => console.log('[page]', m.text()));
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
await page.goto('http://localhost:8124/test/blank.html');
const res = await page.evaluate(async () => {
  const H = await import('/hier.js');
  let eng; try { eng = await H.HierGPU.create(); } catch (e) { return { error: String(e) }; }
  const gold = await (await fetch('/test/golden_hier.json')).json();
  const out = {}; let worst = 0, worstFlip = 0;
  for (const [name, g] of Object.entries(gold)) {
    const prog = H.parseHier(await (await fetch(`/models/${name}.json`)).text());
    const t0 = performance.now(); const r = await eng.run(prog, Float32Array.from(g.logI), g.dims, g.spacing); const dt = performance.now() - t0;
    for (const [k, ref] of Object.entries(g.out)) {
      const a = r[k];
      if (typeof ref === 'number') { const d = Math.abs(a - ref); out[`${name}.${k}`] = { diff: d, ref }; worst = Math.max(worst, d / Math.max(1, Math.abs(ref))); continue; }
      let md = 0, mx = 0, flips = 0, binary = true;
      for (let i = 0; i < ref.length; i++) { md = Math.max(md, Math.abs(a[i] - ref[i])); mx = Math.max(mx, Math.abs(ref[i])); if ((a[i] > 0.5) !== (ref[i] > 0.5)) flips++; if (ref[i] > 1e-6 && ref[i] < 1 - 1e-6) binary = false; }
      out[`${name}.${k}`] = { maxdiff: md, refmax: mx, flipFrac: flips / ref.length, binary };
      if (binary) worstFlip = Math.max(worstFlip, flips / ref.length); else worst = Math.max(worst, md / Math.max(mx, 1e-6));
    }
    out[`${name}.ms`] = Math.round(dt);
  }
  out.summary = { worst, worstFlip, pass: worst < 1e-2 && worstFlip < 5e-3 };
  return out;
});
console.log(JSON.stringify(res, null, 1));
await browser.close(); srv.close();
process.exit(res.summary && res.summary.pass ? 0 : 1);
