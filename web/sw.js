// Cache the dashboard so it opens instantly from the home screen, even with no
// signal. Network-first, so a fresh build always wins when you are online.
const CACHE = 'fpl-assistant-v4';
// The stylesheet belongs here: without it a cold offline launch renders the
// dashboard unstyled, because nothing has warmed the runtime cache yet.
const ASSETS = ['./', './index.html', './sports-ui.css', './manifest.webmanifest',
                './icon.svg', './icon-180.png', './icon-512.png',
                './summary.json', './deadlines.ics'];

self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(CACHE)
      // One missing file must not abort the whole install, which is what
      // addAll() does — it rejects the entire batch on a single failure.
      .then((c) => Promise.all(ASSETS.map((u) => c.add(u).catch(() => {}))))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET') return;
  // Only same-origin responses are cacheable: putting an opaque cross-origin
  // response (the Google Fonts files) throws and rejects unhandled.
  const sameOrigin = new URL(req.url).origin === self.location.origin;
  e.respondWith(
    fetch(req)
      .then((res) => {
        if (sameOrigin && res.ok) {
          const copy = res.clone();
          caches.open(CACHE).then((c) => c.put(req, copy)).catch(() => {});
        }
        return res;
      })
      .catch(() => caches.match(req).then((hit) => hit || caches.match('./index.html')))
  );
});
