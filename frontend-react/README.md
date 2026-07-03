# RAG Frontend (React + TypeScript + Vite)

Typed port of the A/B comparison UI. The plain-HTML version lives at
`../frontend/index.html`; this is the full-stack variant with components,
typed API client, and a Vite dev server.

```bash
npm install
npm run dev      # http://localhost:5173, proxies /compare etc. to :8000
npm run build    # type-check + production bundle → dist/
```

Requires the FastAPI backend running on :8000 (`uvicorn app:app` from the repo root).

Structure:
- `src/types.ts` — response types mirroring the API
- `src/api.ts` — typed fetch client
- `src/App.tsx` — controls + state
- `src/components/ResultPanel.tsx` — one strategy's answer, metrics, sources
