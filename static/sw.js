// Venenno SW v1 — sem cache agressivo
self.addEventListener('install',  () => self.skipWaiting());
self.addEventListener('activate', e  => {
  e.waitUntil(
    caches.keys()
      .then(ks => Promise.all(ks.map(k => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});
// Sem cache — sempre busca do servidor
self.addEventListener('fetch', e => e.respondWith(fetch(e.request)));
