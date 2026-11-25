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
const subjects = [...new Set(assets.map((a) => a.sub))].sort((a, b) => (a === 'MITU01' ? -1 : b === 'MITU01' ? 1 : a.localeCompare(b)));   // the BICCN dashboard's subject first
subjects.forEach((s_) => $('sub').add(new Option(s_, s_)));
const inSub = (a) => a.stain === $('stain').value && a.sub === $('sub').value;                      // slab numbers repeat across subjects: always select by (subject, stain, sample)
const sel = () => assets.filter((a) => inSub(a) && a.sample === $('sample').value && (!$('session').value || a.session === $('session').value)).sort((a, b) => a.chunk - b.chunk);
function fillSubjects() { const have = new Set(assets.filter((a) => a.stain === $('stain').value).map((a) => a.sub)), cur = $('sub').value; [...$('sub').options].forEach((o) => { o.disabled = !have.has(o.value); }); if (!have.has(cur)) $('sub').value = subjects.find((x) => have.has(x)); }
function fillSamples() { fillSubjects(); const ss = [...new Set(assets.filter(inSub).map((a) => a.sample))].sort((a, b) => +a - +b); $('sample').innerHTML = ''; ss.forEach((v) => $('sample').add(new Option('sample ' + v, v))); fillSessions(); }
function fillSessions() {
  const rows = assets.filter((a) => inSub(a) && a.sample === $('sample').value), ss = [...new Set(rows.map((a) => a.session))].sort();
  $('session').innerHTML = ''; ss.forEach((v) => $('session').add(new Option(`${v.slice(0, 4)}-${v.slice(4, 6)}-${v.slice(6, 8)} (${rows.filter((a) => a.session === v).length} strips)`, v))); fillStrips();
}
function fillStrips() { $('strip').innerHTML = ''; sel().forEach((a) => $('strip').add(new Option('strip ' + a.chunk, a.zarr))); }

// ---- models, by what each is meant for
const MODELS = [
  { key: 'fable_blobs_v2', label: 'Cells: nuclei / somata (hand-written v2)', stains: ['YO', 'NN'], note: 'ellipsoids on detected nuclei / somata (levels 0-1; zoom in to a few mm)' },
  { key: 'fable_vessels_v2', label: 'Vessels (hand-written v2)', stains: ['LEC', 'CD31'], note: 'ellipsoids on vessel segments (levels 0-1; LEC only)' },
  { key: 'regions', label: 'Tissue outline + cortex / white-matter boundary', stains: ['YO', 'NN'], note: 'yellow tissue outline, cyan cortex / white-matter boundary (whole slab or >= 10 mm views)' },
  // Neurodesk reference models (github.com/neurodesk/webapps): trained networks / a classical detector, run with onnxruntime-web
  // (no WebGPU needed); weights are fetched at run time from the Neurodesk asset store. Level 0, about 1-3 mm fields of view.
  { key: 'nd_cellpose', label: 'Reference: Cellpose 3 nuclei (Neurodesk)', stains: ['YO', 'NN'], ref: true, diam: true, note: 'Cellpose nuclei, 2-D per plane, planes resampled so the declared diameter spans 17 px, linked across z; 26 MB weights (BSD-3; training data partly CC-BY-NC)' },
  { key: 'nd_stardist', label: 'Reference: StarDist 2D fluorescence (Neurodesk)', stains: ['YO', 'NN'], ref: true, diam: true, note: 'StarDist 2D_versatile_fluo, 2-D per plane, resampled so the declared diameter spans 16 px, polygon NMS, linked across z; 6 MB weights (BSD-3)' },
  { key: 'nd_cellfinder', label: 'Reference: cellfinder candidates (Neurodesk port)', stains: ['YO', 'NN'], ref: true, diam: true, note: 'cellfinder 1.10.1 classical 3-D detection (no weights; BSD-3): candidate centres drawn as spheres of the declared diameter' },
  { key: 'nd_vesselboost', label: 'Reference: VesselBoost (Neurodesk, MRA-trained, experimental)', stains: ['LEC', 'CD31'], ref: true, diam: false, note: 'VesselBoost 3-D U-Net trained on MR angiography, 64^3 patches on a 64-plane slab; 26 MB weights, licence not stated (code MIT)' },
  { key: '', label: 'None (image only)', stains: null, note: '' },
];
const DIAM = { YO: 10, NN: 16 };                                                  // declared nucleus / soma diameter (um) per stain for the reference models
function fillModels() {
  const st = $('stain').value, ok = (m) => !m.stains || m.stains.includes(st); $('model').innerHTML = '';
  MODELS.forEach((m) => { const o = new Option(m.label + (ok(m) ? '' : ' (not for this stain)'), m.key); o.disabled = !ok(m); $('model').add(o); });
  $('model').value = (MODELS.find((m) => m.stains && m.stains.includes(st)) || MODELS[0]).key;
  if (!MODELS.find((m) => m.key === $('model').value && ok(m))) $('model').value = '';
  $('diam').value = DIAM[st] || 10; showDiam();
}
const showDiam = () => { const m = MODELS.find((x) => x.key === $('model').value); $('diam').hidden = $('diamlab').hidden = !(m && m.diam); $('modelnote').textContent = m && m.ref ? m.note : ''; };
const hook = () => { const f = $('ng'); try { return f && f.contentWindow && f.contentWindow.__hwNg; } catch (e) { return null; } };
let current = null, polled = 0, wasBusy = false, runTask = null;

async function open(ngOverride = null) {
  const mode = $('mode').value, rows = sel(); if (!rows.length) return;
  const bt = banner.start('open', `Opening ${mode === 'slab' ? `the stitched slab (${rows.length} strips)` : 'one strip'} · ${$('sub').value} ${$('stain').value} sample ${$('sample').value}`);
  try {
    const src = mode === 'slab' ? new Slab(rows.map((a) => a.zarr)) : new Zarr($('strip').value || rows[0].zarr); await src.init();
    const L0 = src.levels[0], sp = L0.sp, off = src.ids ? L0.origin : L0.tr, ids = src.ids || [$('strip').value || rows[0].zarr];
    const frame = $('ng').getBoundingClientRect(), wum = L0.shape[2] * sp[2], hum = L0.shape[1] * sp[1];
    const view = { cx: wum / 2, cy: hum / 2, z: L0.shape[0] / 2, p: Math.max(wum / Math.max(frame.width - 330, 300), hum / Math.max(frame.height, 300)) * 1.05 };   // leave room for Neuroglancer's layer panel
    const state = mode === 'slab' && swOk ? ngStateVirtual(ids, { sp }, view, $('seams').checked) : ngState(ids, { sp, off }, view, null);   // virtual volume: our blended stitch, not Neuroglancer's native overlap compositing
    if (ngOverride) Object.assign(state, ngOverride);                            // a shared link: take its Neuroglancer state as is (layers, view, shader, panels)
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
  if (current) { rows.push(['Subject', $('sub').value]); rows.push(['Strips', `${current.ids.length}${current.dropped.length ? ` (+${current.dropped.length} re-acquired, left out)` : ''}`]); rows.push(['Voxel (level 0)', current.sp.map((x) => x.toFixed(2)).join(' x ') + ' um']); }
  try { const j = $('ng').contentWindow.viewer && $('ng').contentWindow.viewer.state.toJSON(), im = j && j.layers.find((l) => l.type === 'image'), sc = im && im.shaderControls;
    if (sc && sc.lo !== undefined) rows.push(['Window (model)', `log ${sc.lo}..${sc.hi}, gamma ${sc.gamma ?? 1}`]); } catch (e) { /* not ready */ }
  if (h && h.last) rows.push(['Last run', h.last.regions ? `tissue ${h.last.tissuePct.toFixed(0)}%, white matter ${h.last.wmPct.toFixed(0)}% (level ${h.last.level})` : `${h.last.n} detections (${h.last.model}, level ${h.last.level}${h.last.ref ? `, ${h.last.ref.seconds.toFixed(0)} s, ${Object.values(h.last.ref.ep).join('/')}` : ''})`]);
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
$('sub').onchange = () => { fillSamples(); open(); };
$('sample').onchange = () => { fillSessions(); open(); };
$('session').onchange = () => { fillStrips(); open(); };
$('mode').onchange = () => { $('strip').hidden = $('striplab').hidden = $('mode').value !== 'strip'; open(); };
$('strip').onchange = () => open();
$('diam').onchange = () => { const h = hook(); if (h && h.setDiam) h.setDiam($('diam').value); };
$('model').onchange = () => { legend(); showDiam(); const h = hook(); if ($('auto').checked && h) h.setAuto(true, $('model').value); };
$('segalpha').oninput = () => { $('segalpha-v').textContent = $('segalpha').value + '%'; const h = hook(); if (h && h.setSegAlpha) h.setSegAlpha($('segalpha').value / 100); };
$('depth').onchange = () => { const h = hook(); if (h) h.setDepth($('depth').value); };
$('run').onclick = () => { const h = hook(); if (h) { h.setDepth($('depth').value); if (h.setDiam) h.setDiam($('diam').value); if (h.setSegAlpha) h.setSegAlpha($('segalpha').value / 100); } if (h && $('model').value) h.runModel($('model').value); };
$('auto').onchange = () => { const h = hook(); if (h) h.setAuto($('auto').checked && !!$('model').value, $('model').value); };
$('fit').onclick = () => open();
$('seams').onchange = () => open();
// A link = this page + sidebar choices + the full Neuroglancer state (layers, view, shader, panel). Segmentation layers carry their
// run parameters in the source URL, so opening the link regenerates them on demand; the (large) cell-centre annotation layer is left out.
const uiState = () => ({ sub: $('sub').value, stain: $('stain').value, sample: $('sample').value, session: $('session').value, mode: $('mode').value, strip: $('strip').value, seams: $('seams').checked, alpha: $('segalpha').value, model: $('model').value, depth: $('depth').value, diam: $('diam').value, auto: $('auto').checked });
function shareUrl() {
  const v = $('ng').contentWindow.viewer; if (!v) throw new Error('viewer not ready');
  const ng = v.state.toJSON(); ng.layers = (ng.layers || []).filter((l) => !(l.type === 'annotation' && /^hwseg /.test(l.name)));
  const u = new URL('index.html', location.href); u.search = ''; u.hash = 's=' + encodeURIComponent(JSON.stringify({ v: 1, ui: uiState(), ng }));
  return u.href;
}
$('copy').onclick = async () => { try { const u = shareUrl(); await navigator.clipboard.writeText(u); banner.log(`copied a link with the full view (${(u.length / 1024).toFixed(1)} KB); segmentations are regenerated when it is opened`); } catch (e) { banner.log('could not copy: ' + e.message, 'error'); } };
$('regen').onclick = async () => { const h = hook(); if (!h) return; const n = await h.regenSeg(); banner.log(n ? `regenerating ${n} segmentation layer(s) from their parameters` : 'no segmentation layers to regenerate', n ? 'info' : 'error'); };
$('nglink').onchange = () => {
  let txt = $('nglink').value.trim(); try { txt = decodeURIComponent(txt); } catch (e) { /* raw */ }
  const ids = [...new Set([...txt.matchAll(/dandiarchive\.s3\.amazonaws\.com\/zarr\/([0-9a-f-]{36})/g)].map((m) => m[1]))], hit = ids.map((id) => assets.find((a) => a.zarr === id)).filter(Boolean);
  if (!hit.length) { banner.log('no known DANDI 000108 zarr sources in that link', 'error'); return; }
  const a = hit[0]; $('stain').value = a.stain; $('sub').value = a.sub; fillSamples(); $('sample').value = a.sample; fillSessions(); $('session').value = a.session; fillStrips(); fillModels(); $('mode').value = ids.length > 1 ? 'slab' : 'strip'; $('strip').hidden = $('striplab').hidden = $('mode').value !== 'strip'; if (ids.length === 1) $('strip').value = a.zarr;
  banner.log(`link: ${ids.length} zarr source(s) -> ${a.stain} sample ${a.sample}`); open();
};

// deep link: #s=<shared view>, ?stain=YO&sample=90  (or #id=<zarr id>)
const q = new URLSearchParams(location.search), H = new URLSearchParams(location.hash.slice(1));
let shared = null; try { if (H.get('s')) shared = JSON.parse(H.get('s')); } catch (e) { banner.log('could not read the shared view in the link', 'error'); }
if (shared && shared.ng) {                                                       // the link may come from another host: point our virtual volumes at this site
  const base = new URL('virtual/', location.href).href;
  shared.ng = JSON.parse(JSON.stringify(shared.ng).replace(/(precomputed:\/\/)https?:\/\/[^"]*?\/virtual\//g, '$1' + base));
}
const ui = shared && shared.ui || {};
const start = assets.find((a) => a.zarr === H.get('id')) || assets.find((a) => a.stain === (ui.stain || q.get('stain') || 'YO') && a.sub === (ui.sub || q.get('sub') || 'MITU01') && a.sample === (ui.sample || q.get('sample') || '90')) || assets.find((a) => a.stain === 'YO' && a.sub === 'MITU01') || assets[0];
$('stain').value = start.stain; $('sub').value = start.sub; fillSamples(); $('sample').value = start.sample; fillSessions(); $('session').value = ui.session && [...$('session').options].some((o) => o.value === ui.session) ? ui.session : start.session; fillStrips(); fillModels();
if (ui.mode) { $('mode').value = ui.mode; $('strip').hidden = $('striplab').hidden = ui.mode !== 'strip'; }
if (ui.strip && [...$('strip').options].some((o) => o.value === ui.strip)) $('strip').value = ui.strip;
if (ui.seams !== undefined) $('seams').checked = !!ui.seams; if (ui.model !== undefined && [...$('model').options].some((o) => o.value === ui.model && !o.disabled)) $('model').value = ui.model;
if (ui.alpha) { $('segalpha').value = ui.alpha; $('segalpha-v').textContent = ui.alpha + '%'; } if (ui.depth) $('depth').value = ui.depth; if (ui.diam) $('diam').value = ui.diam; if (ui.auto) $('auto').checked = true; legend(); showDiam();
window.__shell = { open, hook, shareUrl, get current() { return current; } };
await open(shared && shared.ng ? shared.ng : null);
