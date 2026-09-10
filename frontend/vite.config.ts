import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src"),
    },
  },
  server: {
    port: 5173,
    host: true,
    allowedHosts: true,
    proxy: {
      // Frontend → FastAPI backend (dev only).  The api client prefixes all
      // HTTP calls with /backend in development; vite rewrites them to the
      // backend's own routes (e.g. /backend/predict -> http://127.0.0.1:8000/predict).
      "/backend": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        ws: true,
        rewrite: (path) => path.replace(/^\/backend/, ""),
      },
    },
  },
});
