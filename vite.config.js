import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    // Honour an assigned PORT so the dev server can coexist with others.
    port: Number(process.env.PORT) || 5173,
    host: '127.0.0.1',
    strictPort: false,
  },
  // Relative asset URLs, so the same build works at a domain root, in a
  // /repo-name/ subpath on GitHub Pages, or opened from a file server.
  base: './',
  build: {
    target: 'es2022',
    // Source maps are 6.6 MB of the 10 MB output and only help someone
    // debugging the minified bundle. Off for deploys; flip on locally if needed.
    sourcemap: false,
    chunkSizeWarningLimit: 4096,
  },
  // Rapier's compat build inlines its WASM as base64, so no special asset handling
  // is needed — but it must not be pre-bundled in a way that breaks the async init.
  optimizeDeps: {
    exclude: [],
  },
});
