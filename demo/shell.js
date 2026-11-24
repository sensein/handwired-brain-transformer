// Neuroglancer shell: sidebar (sample / model selection) + an embedded Neuroglancer (our build, /ng/) whose hook runs the
// weight-programmed models on exactly the visible field of view with WebGPU and writes the results back as annotation layers.
import { Zarr, Slab } from './zarr.js';
import { ngState, ngStateVirtual } from './ng-state.js';
import { banner } from './banner.js';

const $ = (id) => document.getElementById(id);
const assets = await (await fetch('assets.json')).json();
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
banner.init(); banner.setStream(() => null);
// the stitched slab is served to Neuroglancer as a virtual volume by sw.js; register it before any iframe is created
let swOk = false;
try { if ('serviceWorker' in navigator) { await navigator.serviceWorker.register('sw.js'); await navigator.serviceWorker.ready; swOk = true; } } catch (e) { banner.log('service worker unavailable: whole slabs fall back to Neuroglancer\'s native multi-source layer (overlaps show as bright bands)', 'error'); }

// ---- sample selection
const PRIMARY = ['YO', 'NN', 'LEC'];
const stains = [...new Set(assets.map((a) => a.stain))].sort((a, b) => (PRIMARY.indexOf(b) + 1) - (PRIMARY.indexOf(a) + 1) || a.localeCompare(b));
stains.forEach((s) => $('stain').add(new Option(s + ({ YO: ' (nuclei)', NN: ' (NeuN)', LEC: ' (vessels)' }[s] || ''), s)));
const sel = () => assets.filter((a) => a.stain === $('stain').value && a.sample === $('sample').value && (!$('session').value || a.session === $('session').value)).sort((a, b) => a.chunk - b.chunk);
function fillSamples() { const ss = [...new Set(assets.filter((a) => a.stain === $('stain').value).map((a) => a.sample))].sort((a, b) => +a - +b); $('sample').innerHTML = ''; ss.forEach((v) => $('sample').add(new Option('sample ' + v, v))); fillSessions(); }
function fillSessions() {
  const rows = assets.filter((a) => a.stain === $('stain').value && a.sample === $('sample').value), ss = [...new Set(rows.map((a) => a.session))].sort();
  $('session').innerHTML = ''; ss.forEach((v) => $('session').add(new Option(`${v.slice(0, 4)}-${v.slice(4, 6)}-${v.slice(6, 8)} (${rows.filter((a) => a.session === v).length} strips)`, v))); fillStrips();
}
function fillStrips() { $('strip').innerHTML = ''; sel().forEach((a) => $('strip').add(new Option('strip ' + a.chunk, a.zarr))); }

// ---- models, by what each is meant for
const MODELS = [
  { key: 'fable_blobs_v2', label: 'Cells: nuclei / somata (hand-written v2)', stains: ['YO', 'NN'], note: 'ellipsoids on detected nuclei / somata (levels 0-1; zoom in to a few mm)' },
  { key: 'fable_vessels_v2', label: 'Vessels (hand-written v2)', stains: ['LEC', 'CD31'], note: 'ellipsoids on vessel segments (levels 0-1; LEC only)' },
  { key: 'regions', label: 'Tissue outline + cortex / white-matter boundary', stains: ['YO', 'NN'], note: 'yellow tissue outline, cyan cortex / white-matter boundary (whole slab or >= 10 mm views)' },
  { key: '', label: 'None (image only)', stains: null, note: '' },
];
function fillModels() {
  const st = $('stain').value, ok = (m) => !m.stains || m.stains.includes(st); $('model').innerHTML = '';
  MODELS.forEach((m) => { const o = new Option(m.label + (ok(m) ? '' : ' (not for this stain)'), m.key); o.disabled = !ok(m); $('model').add(o); });
  $('model').value = (MODELS.find((m) => m.stains && m.stains.includes(st)) || MODELS[0]).key;
  if (!MODELS.find((m) => m.key === $('model').value && ok(m))) $('model').value = '';
}
const hook = () => { const f = $('ng'); try { return f && f.contentWindow && f.contentWindow.__hwNg; } catch (e) { return null; } };
let current = null, polled = 0, wasBusy = false, runTask = null;

async function open() {
  const mode = $('mode').value, rows = sel(); if (!rows.length) return;
  const bt = banner.start('open', `Opening ${mode === 'slab' ? `the stitched slab (${rows.length} strips)` : 'one strip'} · ${$('stain').value} sample ${$('sample').value}`);
  try {
    const src = mode === 'slab' ? new Slab(rows.map((a) => a.zarr)) : new Zarr($('strip').value || rows[0].zarr); await src.init();
    const L0 = src.levels[0], sp = L0.sp, off = src.ids ? L0.origin : L0.tr, ids = src.ids || [$('strip').value || rows[0].zarr];
    const frame = $('ng').getBoundingClientRect(), wum = L0.shape[2] * sp[2], hum = L0.shape[1] * sp[1];
    const view = { cx: wum / 2, cy: hum / 2, z: L0.shape[0] / 2, p: Math.max(wum / Math.max(frame.width - 330, 300), hum / Math.max(frame.height, 300)) * 1.05 };   // leave room for Neuroglancer's layer panel
    const state = mode === 'slab' && swOk ? ngStateVirtual(ids, { sp }, view, $('seams').checked) : ngState(ids, { sp, off }, view, null);   // virtual volume: our blended stitch, not Neuroglancer's native overlap compositing
    const f = document.createElement('iframe'); f.id = 'ng'; f.title = 'Neuroglancer'; f.allow = 'fullscreen';
    f.src = 'ng/index.html?embedded=1#!' + encodeURIComponent(JSON.stringify(state)); $('ng').replaceWith(f);
    current = { ids, sp, off, dropped: src.dropped || [], shape: L0.shape }; polled = 0; wasBusy = false;
    banner.finish(bt, true, `${ids.length} strip${ids.length > 1 ? 's' : ''}${current.dropped.length ? `, ${current.dropped.length} re-acquired strip(s) left out` : ''}, level-0 grid ${L0.shape.join(' x ')}`);
    updateInfo(); legend();
  } catch (e) { banner.finish(bt, false, e.message); }
}

function legend() {
  const m = MODELS.find((x) => x.key === $('model').value);
  $('legend').innerHTML = !m || !m.key ? '' : m.key === 'regions' ? '<span style="background:#ffd23c"></span>tissue outline<span style="background:#3cd2ff"></span>cortex / white-matter boundary' : '<span style="background:#ff3cc8"></span>detections';
}
function updateInfo() {
  const h = hook(), rows = [];
  if (current) { rows.push(['Strips', `${current.ids.length}${current.dropped.length ? ` (+${current.dropped.length} re-acquired, left out)` : ''}`]); rows.push(['Voxel (level 0)', current.sp.map((x) => x.toFixed(2)).join(' x ') + ' um']); }
  try { const j = $('ng').contentWindow.viewer && $('ng').contentWindow.viewer.state.toJSON(), im = j && j.layers.find((l) => l.type === 'image'), sc = im && im.shaderControls;
    if (sc && sc.lo !== undefined) rows.push(['Window (model)', `log ${sc.lo}..${sc.hi}, gamma ${sc.gamma ?? 1}`]); } catch (e) { /* not ready */ }
  if (h && h.last) rows.push(['Last run', h.last.regions ? `tissue ${h.last.tissuePct.toFixed(0)}%, white matter ${h.last.wmPct.toFixed(0)}% (level ${h.last.level})` : `${h.last.n} detections (${h.last.model}, level ${h.last.level})`]);
  $('info').innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('');
}

// ---- bridge to the hook inside the iframe: status events -> banner, busy flag -> a banner task
setInterval(() => {
  const h = hook(); $('run').disabled = !(h && h.ready) || !$('model').value;
  if (!h) return;
  for (; polled < h.events.length; polled++) banner.log('model: ' + h.events[polled].msg, /^error/.test(h.events[polled].msg) ? 'error' : 'info');
  if (h.busy && !wasBusy) runTask = banner.start('infer', `${MODELS.find((m) => m.key === $('model').value)?.label || 'model'} on the current view`);
  if (!h.busy && wasBusy && runTask) { const last = h.events[h.events.length - 1]; banner.finish(runTask, !/^error/.test(last ? last.msg : ''), last ? last.msg : ''); runTask = null; updateInfo(); }
  if (runTask && h.events.length) banner.update(runTask, { detail: h.events[h.events.length - 1].msg });
  wasBusy = h.busy;
}, 400);

// ---- controls
$('stain').onchange = () => { fillSamples(); fillModels(); open(); };
$('sample').onchange = () => { fillSessions(); open(); };
$('session').onchange = () => { fillStrips(); open(); };
$('mode').onchange = () => { $('strip').hidden = $('striplab').hidden = $('mode').value !== 'strip'; open(); };
$('strip').onchange = () => open();
$('model').onchange = () => { legend(); const h = hook(); if ($('auto').checked && h) h.setAuto(true, $('model').value); };
$('depth').onchange = () => { const h = hook(); if (h) h.setDepth($('depth').value); };
$('run').onclick = () => { const h = hook(); if (h) h.setDepth($('depth').value); if (h && $('model').value) h.runModel($('model').value); };
$('auto').onchange = () => { const h = hook(); if (h) h.setAuto($('auto').checked && !!$('model').value, $('model').value); };
$('fit').onclick = () => open();
$('seams').onchange = () => open();
$('copy').onclick = async () => { try { const u = $('ng').contentWindow.location.href.replace('?embedded=1', ''); await navigator.clipboard.writeText(u); banner.log('copied the Neuroglancer link (state in the URL hash)'); } catch (e) { banner.log('could not copy: ' + e.message, 'error'); } };
$('nglink').onchange = () => {
  let txt = $('nglink').value.trim(); try { txt = decodeURIComponent(txt); } catch (e) { /* raw */ }
  const ids = [...new Set([...txt.matchAll(/dandiarchive\.s3\.amazonaws\.com\/zarr\/([0-9a-f-]{36})/g)].map((m) => m[1]))], hit = ids.map((id) => assets.find((a) => a.zarr === id)).filter(Boolean);
  if (!hit.length) { banner.log('no known DANDI 000108 zarr sources in that link', 'error'); return; }
  const a = hit[0]; $('stain').value = a.stain; fillSamples(); $('sample').value = a.sample; fillSessions(); $('session').value = a.session; fillStrips(); fillModels(); $('mode').value = ids.length > 1 ? 'slab' : 'strip'; $('strip').hidden = $('striplab').hidden = $('mode').value !== 'strip'; if (ids.length === 1) $('strip').value = a.zarr;
  banner.log(`link: ${ids.length} zarr source(s) -> ${a.stain} sample ${a.sample}`); open();
};

// deep link: ?stain=YO&sample=90  (or #id=<zarr id>)
const q = new URLSearchParams(location.search), H = new URLSearchParams(location.hash.slice(1));
const start = assets.find((a) => a.zarr === H.get('id')) || assets.find((a) => a.stain === (q.get('stain') || 'YO') && a.sample === (q.get('sample') || '90')) || assets[0];
$('stain').value = start.stain; fillSamples(); $('sample').value = start.sample; fillSessions(); $('session').value = start.session; fillStrips(); fillModels();
window.__shell = { open, hook, get current() { return current; } };
await open();
