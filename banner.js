// Expandable status banner: what the viewer is doing right now (open, scan, stream, exposure, inference, 3D) with progress,
// elapsed time, streaming stats, and a short log. Pure UI: tasks are started/updated/finished by the app.
const el = (id) => document.getElementById(id);
const fmtT = (ms) => (ms < 1000 ? `${ms.toFixed(0)} ms` : `${(ms / 1000).toFixed(1)} s`);
const clock = () => new Date().toLocaleTimeString([], { hour12: false }) + '.' + String(Date.now() % 1000).padStart(3, '0');
export const banner = (() => {
  const tasks = new Map(), history = []; let seq = 0, pending = null, stream = () => null, timer = null, lastLog = '';
  const PRI = { infer: 5, exposure: 4, '3d': 4, scan: 3, open: 3, other: 1 };
  function log(msg, level = 'info') { if (msg === lastLog) return; lastLog = msg; history.push({ t: clock(), msg, level }); if (history.length > 80) history.shift(); render(); }
  function start(kind, label, total = null) { const id = ++seq; tasks.set(id, { id, kind, label, total, done: 0, detail: '', t0: performance.now() }); log(`start: ${label}`); spin(); return id; }
  function update(id, p) { const t = tasks.get(id); if (!t) return; Object.assign(t, p); render(); }
  function finish(id, ok = true, note = '') { const t = tasks.get(id); if (!t) return; tasks.delete(id); const dt = performance.now() - t.t0; log(`${ok ? 'done' : 'FAILED'}: ${t.label} (${fmtT(dt)})${note ? ' · ' + note : ''}`, ok ? 'info' : 'error'); render(); }
  function setPending(ms) { pending = ms ? { until: performance.now() + ms, ms } : null; spin(); render(); }
  function setStream(fn) { stream = fn; }
  function spin() { if (!timer) timer = setInterval(() => { render(); if (!tasks.size && !pending && !(stream() || {}).inflight) { clearInterval(timer); timer = null; render(); } }, 250); }
  function render() {
    const root = el('banner'); if (!root) return;
    const s = stream() || {}, act = [...tasks.values()].sort((a, b) => (PRI[b.kind] || 1) - (PRI[a.kind] || 1)), top = act[0], busy = act.length > 0 || pending || s.inflight > 0;
    const failed = history.length && history[history.length - 1].level === 'error';
    root.dataset.state = failed && !busy ? 'error' : busy ? 'busy' : 'idle';
    let line = 'Ready';
    if (top) { const dt = performance.now() - top.t0; line = `${top.label}${top.detail ? ' · ' + top.detail : ''}${top.total ? ` · ${top.done}/${top.total}` : ''} · ${fmtT(dt)}`; }
    else if (pending) line = `View changed: segmentation starts when it settles (${Math.max(0, pending.until - performance.now()).toFixed(0)} ms)`;
    else if (s.inflight > 0) line = `Streaming ${s.inflight} chunk${s.inflight > 1 ? 's' : ''} from DANDI`;
    else if (failed) line = history[history.length - 1].msg;
    else if (history.length) line = history[history.length - 1].msg.replace(/^done: /, 'Done: ');
    el('bn-line').textContent = line;
    el('bn-meta').textContent = s.chunks != null ? `${s.chunks} chunks · ${(s.bytes / 1e6).toFixed(1)} MB streamed${s.inflight ? ` · ${s.inflight} in flight` : ''}` : '';
    const bar = el('bn-bar').firstElementChild; const frac = top && top.total ? Math.min(1, top.done / top.total) : null;
    bar.style.width = frac != null ? `${(frac * 100).toFixed(0)}%` : busy ? '40%' : '0%'; bar.className = frac == null && busy ? 'ind' : '';
    if (!el('bn-detail').hidden) {
      el('bn-tasks').innerHTML = act.length || pending || s.inflight ? [...act.map((t) => `<li><b>${t.kind}</b> ${t.label}${t.detail ? ' · ' + t.detail : ''}${t.total ? ` · ${t.done}/${t.total}` : ''} <em>${fmtT(performance.now() - t.t0)}</em></li>`),
        pending ? `<li><b>debounce</b> waiting for the view to settle <em>${Math.max(0, pending.until - performance.now()).toFixed(0)} ms</em></li>` : '', s.inflight ? `<li><b>stream</b> ${s.inflight} chunk requests in flight</li>` : ''].join('') : '<li>idle</li>';
      el('bn-log').textContent = history.slice(-40).map((h) => `${h.t}  ${h.level === 'error' ? '! ' : '  '}${h.msg}`).join('\n');
    }
    if (pending && performance.now() > pending.until) pending = null;
  }
  function init() {
    el('bn-toggle').onclick = () => { const d = el('bn-detail'), open = d.hidden; d.hidden = !open; el('bn-toggle').setAttribute('aria-expanded', String(open)); el('bn-toggle').textContent = open ? '▾' : '▸'; render(); };
    el('banner').addEventListener('dblclick', () => el('bn-toggle').click()); render();
  }
  return { start, update, finish, log, setPending, setStream, init, render };
})();
