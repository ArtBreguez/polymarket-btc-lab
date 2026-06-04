# 05 — Deployment

## Architecture Overview

```
GitHub (main) ──► CI (ci.yml) ──► tests + validate
                  │
                  └─► Deploy (deploy.yml) ──► validate + flyctl deploy ──► Fly.io (AMS)
```

---

## CI Pipeline (ci.yml)

Triggers: every push (all branches), PRs to main.

**Job 1: Unit tests**
- Python 3.12, `uv sync`, `pytest tests/ -v`

**Job 2: Validate champion model** (main branch or deploy/ changes only)
- Downloads champion.pkl from HuggingFace
- Runs `scripts/validate_model.py` — checks feature count, WF AUC >= 0.65, directional sanity probes
- Requires `HF_TOKEN` secret

---

## Deploy Pipeline (deploy.yml)

Triggers:
- Push to main when `deploy/**` or `.github/workflows/deploy.yml` changes
- **Manual trigger**: `gh workflow run deploy.yml` (with optional reason)

**Single job: validate-and-deploy**
1. Validate champion model (same as CI — gate before deploy)
2. Install flyctl
3. `flyctl deploy --app polymarket-maker-mm --ha=false --remote-only`
4. Verify: `flyctl status --app polymarket-maker-mm`

Environment: `production`

---

## Fly.io Configuration

| Setting | Value |
|---------|-------|
| App name | `polymarket-maker-mm` |
| Region | AMS (Amsterdam) |
| HA | Disabled (`--ha=false`) — single instance |
| Deploy | Remote build (`--remote-only`) |

The app runs `deploy/live_trader.py` as a long-running process.

---

## Manual Deploy

After promoting a new champion model:

```bash
gh workflow run deploy.yml --ref main -f reason="Promoted v18 champion"
```

Or via GitHub UI: Actions → Deploy to Fly.io → Run workflow.

---

## GitHub Secrets

| Secret | Purpose | Where used |
|--------|---------|------------|
| `HF_TOKEN` | HuggingFace API token — download champion.pkl | CI + Deploy |
| `FLY_API_TOKEN` | Fly.io deploy token | Deploy only |

Fly.io runtime also needs these secrets (set via `fly secrets set`):
- `POLY_PRIVATE_KEY` — EOA private key
- `POLY_SAFE_ADDRESS` — Proxy wallet
- `MM_BUILDER_KEY`, `MM_BUILDER_SECRET`, `MM_BUILDER_PASSPHRASE` — Builder API
- `HF_TOKEN` — model download at startup

---

## Common Failures

| Error | Cause | Fix |
|-------|-------|-----|
| `401 Unauthorized` (HF) | Expired or invalid HF_TOKEN | Update secret: Settings → Secrets → `HF_TOKEN` |
| `401 Unauthorized` (Fly) | Expired FLY_API_TOKEN | Regenerate: `fly tokens create deploy -a polymarket-maker-mm` |
| `429 Too Many Requests` (HF) | HuggingFace rate limit | Wait 60s and retry; CI will auto-retry on next push |
| `flyctl deploy` timeout | Fly.io build queue congestion | Re-run the workflow; check `fly status` |
| Validate fails in deploy | Champion model regressed or sanity probes fail | Check `scripts/validate_model.py` output; fix model or adjust probes |
| Machine not starting | Missing Fly secrets | `fly secrets list -a polymarket-maker-mm` — verify all 6 secrets present |
