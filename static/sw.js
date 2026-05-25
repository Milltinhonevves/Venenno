// SW v11 — não intercepta POST/processar
const CACHE = 'venenno-v11';

self.addEventListener('install', e => {
  self.skipWaiting();
});

self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.map(k => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

// Nunca cacheia POST — passa direto pro servidor
self.addEventListener('fetch', e => {
  if (e.request.method !== 'GET') return;
  if (e.request.url.includes('/processar')) return;
  if (e.request.url.includes('/download')) return;
  // Só cacheia assets estáticos
  e.respondWith(
    caches.open(CACHE).then(cache =>
      fetch(e.request).then(resp => {
        cache.put(e.request, resp.clone());
        return resp;
      }).catch(() => caches.match(e.request))
    )
  );
});
