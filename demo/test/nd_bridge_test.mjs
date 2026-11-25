// Neurodesk reference model inside our Neuroglancer build: select it in the hook, run it on the view, and check that the served
// segmentation volume has labels across the field of view; then drop the cached volume and check that it regenerates from the
// parameters in its source URL (the shared-link path). Env: SITE (assembled Pages tree), ASSET, MODEL (nd_stardist), P (um per
// screen px), DEPTH (planes), EP (wasm|webgpu), SHOT_DIR.
import { chromium } from 'playwright';
import { ngState } from '../ng-state.js';
import http from 'http'; import fs from 'fs'; import path from 'path';
const ROOT = process.env.SITE, OUT = process.env.SHOT_DIR || '.', PORT = +(process.env.PORT || 8395), ASSET = process.env.ASSET || '383772ae-d8b1-456e-b2fe-eca883a13a66', MODEL = process.env.MODEL || 'nd_stardist';
const mime = { '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json', '.html': 'text/html', '.css': 'text/css', '.wasm': 'application/wasm', '.py': 'text/plain' };
const srv = http.createServer((q, r) => { let f = path.join(ROOT, q.url.split('?')[0].split('#')[0]); if (f.endsWith('/')) f += 'index.html';
  if (!fs.existsSync(f) || fs.statSync(f).isDirectory()) { r.writeHead(404); return r.end(); }
  r.writeHead(200, { 'content-type': mime[path.extname(f)] || 'application/octet-stream' }); r.end(fs.readFileSync(f)); }).listen(PORT);
const browser = await chromium.launch({ headless: true, args: ['--no-sandbox', '--enable-unsafe-webgpu', '--enable-features=Vulkan,UnsafeWebGPU', '--disable-vulkan-surface', '--use-vulkan=swiftshader', '--use-webgpu-adapter=swiftshader', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist', '--headless=new'] });
const page = await browser.newPage({ viewport: { width: 1300, height: 800 } });
page.on('pageerror', (e) => console.log('[pageerror]', e.message.slice(0, 200)));
if (process.env.EP) await page.addInitScript((ep) => { globalThis.__hwNdEP = ep; }, process.env.EP);
const sp = [2.5632620818012344, 3.625, 2.5632620818012344];
const state = ngState([ASSET], { sp, off: [180247, 26568, -237] }, { cx: 24818 * sp[2], cy: 1023 * sp[1], z: 1019, p: +(process.env.P || 1.0) });
await page.goto(`http://localhost:${PORT}/ng/index.html#!${encodeURIComponent(JSON.stringify(state))}`);
await page.waitForFunction(() => window.__hwNg && window.__hwNg.ready, null, { timeout: 180000 });
await page.waitForTimeout(6000);
const opts = await page.evaluate(() => [...document.querySelectorAll('#hw-m option')].map((o) => o.value));
if (!opts.includes(MODEL)) { console.log('FAIL: model not in the hook dropdown', opts); process.exit(1); }
await page.selectOption('#hw-m', MODEL); await page.evaluate((d) => window.__hwNg.setDepth(d), process.env.DEPTH || '1');
const wait = async (t0) => { let st = ''; while (Date.now() - t0 < 840000) { await page.waitForTimeout(2000); st = await page.evaluate(() => window.__hwNg.events.map((e) => e.msg).filter((m) => /^segmentation:|^error/.test(m)).pop() || ''); if (st) break; } return st; };
let t0 = Date.now(); await page.click('#hw-run'); const st = await wait(t0);
console.log('status:', st, `(${((Date.now() - t0) / 1000).toFixed(0)} s)`);
const last = await page.evaluate(() => window.__hwNg.last); console.log('last:', JSON.stringify(last));
// labels on a 4 x 4 grid of 48 x 48 blocks spread over the run box, on the displayed plane, straight from the served volume
const grid = async () => page.evaluate(async () => {
  const j = window.viewer.state.toJSON(), l = j.layers.find((x) => x.type === 'segmentation'); if (!l) return null;
  const base = String(l.source.url || l.source).replace(/^precomputed:\/\//, ''), info = await (await fetch(base + '/info')).json(), s0 = info.scales[0], vo = s0.voxel_offset;
  const b = window.__hwNg.last.boxG, z = Math.floor(j.position[1]), cells = [];
  for (let iy = 0; iy < 4; iy++) for (let ix = 0; ix < 4; ix++) {
    const x0 = Math.round(b[0][0] + (ix + 0.5) * (b[0][1] - b[0][0]) / 4) - 24, y0 = Math.round(b[1][0] + (iy + 0.5) * (b[1][1] - b[1][0]) / 4) - 24;
    const a = new Uint32Array(await (await fetch(`${base}/0/${x0}-${x0 + 48}_${y0}-${y0 + 48}_${z}-${z + 1}`)).arrayBuffer());
    cells.push(new Set([...a].filter(Boolean)).size);
  }
  return { layer: l.name, vo, size: s0.size, z, ids: cells };
});
const g1 = await grid(); console.log('GRID', JSON.stringify(g1));
await page.waitForTimeout(2500); await page.screenshot({ path: `${OUT}/nd_${MODEL}_after.png` });
// regeneration from the source-URL parameters (what a shared link does)
t0 = Date.now(); await page.evaluate(() => window.__hwNg.regenSeg());
await page.waitForFunction(() => { const l = (window.viewer.state.toJSON().layers || []).find((x) => x.type === 'segmentation'); return l && /virtual\/seg\/[A-Za-z0-9_-]+\.[a-z0-9]+/.test(JSON.stringify(l.source)); }, null, { timeout: 60000 });
const g2 = await grid(); console.log('REGEN', JSON.stringify(g2), `(${((Date.now() - t0) / 1000).toFixed(0)} s)`);
const occupied = (g) => (g && g.ids ? g.ids.filter((n) => n > 0).length : 0);
const ok = /segmentation:/.test(st) && last && last.n > 0 && last.covers && occupied(g1) >= 12 && occupied(g2) >= 12;
console.log(ok ? `PASS: ${MODEL} labelled ${occupied(g1)}/16 grid blocks across the FOV (${last.n} objects), regenerated ${occupied(g2)}/16` : 'FAIL');
await browser.close(); srv.close(); process.exit(ok ? 0 : 1);
