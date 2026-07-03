# Deploying to Fly.io

The deployed instance is self-contained: the FAISS index and models are baked
into the image, and it uses **Gemini** for generation (there is no Ollama in the
cloud). You provide a Gemini API key as a secret.

## Cost reality check

The app loads torch + a sentence-transformer + a cross-encoder → **~1.5 GB RAM**.
The 256 MB free VM will OOM. `fly.toml` requests a **2 GB VM (paid)**. With
`auto_stop_machines = true` it scales to zero when idle, so you mostly pay for
active time — but this is not a free deployment. Render's free tier has the same
RAM problem. Budget accordingly, or use it as a "spin up for the demo" instance.

## One-time setup

```bash
# 1. install flyctl and sign in (opens a browser)
brew install flyctl
fly auth login

# 2. build the index locally so it can be baked into the image
python scripts/build_index.py          # creates data/processed/

# 3. create the app (pick a globally-unique name; update it in fly.toml too)
fly apps create rag-eval-based
```

## Set the Gemini key as a secret

```bash
fly secrets set GEMINI_API_KEY=your-key-here
```

Never put the key in `fly.toml` or the Dockerfile — secrets are injected at
runtime and stay out of the image and git.

## Deploy

```bash
fly deploy                # builds Dockerfile.deploy, pushes, boots the machine
fly open                  # opens the live URL
fly logs                  # watch startup (first boot loads models — ~30-60s)
```

## Verify

```bash
curl https://<your-app>.fly.dev/health
# {"status":"ok","chunks":...}
```

Then open the URL and use the A/B comparison UI. On the deployed instance,
leave the provider on **gemini** (ollama is local-only).

## Notes

- **Free-tier Gemini** is limited (~20 requests/day on some keys) — fine for a
  demo, not for load. Enable billing for real traffic.
- To redeploy after code or index changes: rebuild the index if the corpus
  changed (`python scripts/build_index.py`), then `fly deploy` again.
- **Render alternative:** point a Render Web Service at `Dockerfile.deploy`,
  set `GEMINI_API_KEY` in the dashboard, choose an instance with ≥ 2 GB RAM.
