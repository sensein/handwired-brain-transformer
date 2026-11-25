// Service worker: serves `virtual/slab/<zarr ids joined by +>[~s]/...` requests (Neuroglancer `precomputed` info and chunks) by
// asking an open page (the Neuroglancer page, where ng-hook.js hosts our stitcher) to compute them. No network, no server.
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (e) => e.waitUntil(self.clients.claim()));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function host() {   // a window that has registered as the compute host (any page that loaded ng-hook.js)
  for (let i = 0; i < 300; i++) {
    const cs = await self.clients.matchAll({ type: 'window', includeUncontrolled: true });
    const h = cs.find((c) => /\/ng\//.test(c.url)); if (h) return h; await sleep(100);
  }
  return null;
}
function ask(client, msg) {
  return new Promise((resolve, reject) => { const ch = new MessageChannel(); const t = setTimeout(() => reject(new Error('virtual volume host timed out')), 120000);
    ch.port1.onmessage = (ev) => { clearTimeout(t); resolve(ev.data); }; client.postMessage(msg, [ch.port2]); });
}
self.addEventListener('fetch', (e) => {
  const m = new URL(e.request.url).pathname.match(/\/virtual\/(slab|seg)\/([^/]+)\/(.*)$/); if (!m) return;
  e.respondWith((async () => {
    try { const h = await host(); if (!h) return new Response('no virtual-volume host open', { status: 503 });
      const r = await ask(h, { kind: 'virtual', which: m[1], ids: decodeURIComponent(m[2]), path: decodeURIComponent(m[3]) });
      return new Response(r.body, { status: r.status || 200, headers: { 'content-type': r.type || 'application/octet-stream', 'cache-control': 'no-store' } });
    } catch (err) { return new Response(String(err), { status: 500 }); }
  })());
});
