// Canonical keys keep query-string reloads usable offline without returning
// HTML to JSON or asset requests. Local build status must always use network.
const CACHE = 'fpl-assistant-v7';
const ASSETS = ['./', './index.html', './sports-ui.css', './manifest.webmanifest',
                './icon.svg', './icon-180.png', './icon-512.png',
                './summary.json', './deadlines.ics'];
self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE)
    .then((c) => Promise.all(ASSETS.map((u) => c.add(new Request(u, {cache:'reload'})).catch(() => {}))))
    .then(() => self.skipWaiting()));
});
self.addEventListener('activate', (e) => {
  e.waitUntil(caches.keys().then((keys) => Promise.all(
    keys.filter((k) => k.startsWith('fpl-assistant-') && k !== CACHE).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});
self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin || /\/(health|refresh-status)$/.test(url.pathname)) return;
  const canonical = new URL(url.href);
  canonical.search = '';
  const key = canonical.href;
  e.respondWith((async () => {
    const cache = await caches.open(CACHE);
    try {
      const response = await fetch(req, {cache:'no-store'});
      if (response.ok) await cache.put(key, response.clone());
      return response;
    } catch (_) {
      const hit = await cache.match(key);
      if (hit) return hit;
      if (req.mode === 'navigate') {
        const page = await cache.match(new URL('./index.html', self.registration.scope).href);
        if (page) return page;
      }
      return new Response('Offline: resource not cached', {
        status: 503, headers: { 'Content-Type': 'text/plain; charset=utf-8' }
      });
    }
  })());
});
