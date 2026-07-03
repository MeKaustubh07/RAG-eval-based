import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server proxies API calls to the FastAPI backend on :8000, so the
// frontend can call /health, /compare, /agent as same-origin paths.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/health": "http://localhost:8000",
      "/ask": "http://localhost:8000",
      "/compare": "http://localhost:8000",
      "/agent": "http://localhost:8000",
    },
  },
});
