# 08 — Troubleshooting

## Binance HTTP 451 (Geo-block)

**Symptom**: `HTTP 451 Unavailable For Legal Reasons` from Binance REST API.

**Cause**: Binance blocks requests from certain cloud regions (including Modal's US infra and some Fly.io regions).

**Fix**: Pre-fetch Binance spot data locally and upload to Modal Volume:
```bash
# Fetch locally (your machine has Binance access)
python scripts/fetch_binance_spot.py

# Upload to Modal Volume
modal volume put btc-local-data binance_spot_full.parquet /binance_spot_full.parquet
```

For live trading: the spot daemon uses Binance WebSocket (not REST), which typically works from AMS. If WS is also blocked, pre-fetch candles locally and upload periodically.

---

## HuggingFace 401 (Unauthorized)

**Symptom**: `401 Client Error` when downloading champion.pkl in CI or live_trader startup.

**Cause**: Expired or invalid HF_TOKEN.

**Fix**:
1. Generate new token at https://huggingface.co/settings/tokens
2. Update GitHub secret: Settings → Secrets → Actions → `HF_TOKEN`
3. Update Fly.io secret: `fly secrets set HF_TOKEN=hf_xxx -a polymarket-maker-mm`

---

## HuggingFace 429 (Rate Limited)

**Symptom**: `429 Too Many Requests` during model download.

**Cause**: Too many downloads in short period (common during rapid CI iterations).

**Fix**: Wait 60 seconds and retry. CI will auto-retry on next push. For urgent deploys, download manually and cache.

---

## Feature Parity (The 6 Bugs)

Feature parity between training and live inference is the #1 source of silent failures. Historical bugs discovered:

1. **OB timestamp field**: Training used `ts_ms`, live used `timestamp_ms` → wrong tick alignment. Fixed in v5.

2. **Sub-window boundaries**: Training computed 6x30s windows from t=0, live computed from first tick arrival → shifted windows. Fixed by aligning to slot start time.

3. **Zscore denominator**: Training used population std (ddof=0), live used sample std (ddof=1) → slightly different zscores. Fixed to use ddof=0 everywhere.

4. **Lag feature ring buffer**: Training had full history, live started with empty buffer → first few slots had lag_streak=0 always. Mitigated by graceful cold start.

5. **Spot return windows**: Training computed `pre_5m_ret` from exact 5-minute-ago candle, live used most-recent-complete candle → off-by-one minute. Fixed by aligning to candle close times.

6. **VWAP calculation**: Training included all ticks, live excluded ticks outside observation window → different VWAP values. Fixed by filtering consistently to 0-180s window.

**Prevention**: Any training script change that touches feature computation must be mirrored in `deploy/live_trader.py`. Run `tests/test_features.py` to catch mismatches.

---

## Sanity Check Failures

**Symptom**: `validate_model.py` fails with "UP signal not above neutral" or similar.

**Causes**:
- Model learned inverted signal (unlikely but check feature signs)
- Sanity probes are miscalibrated for new feature set (more likely — v18 showed this)
- Model has very different feature importance ordering than probes assume

**Fix**:
1. Check model's actual feature importances — are the probed features still top features?
2. Adjust probe values to match the new feature distribution
3. If model is genuinely correct (confirmed by WF metrics), update probe thresholds

Note: v18 showed "inverted" sanity results because `btc_inslot_ret` became #1 feature but probes didn't set it. The probes tested `btc_up_ratio` heavily, which dropped in importance. The model was correct — the probes needed updating.

---

## Negative P&L

**Symptom**: Consistent losses in live trading.

**Diagnosis checklist**:
1. **Feature parity**: Is `build_features()` producing the same features as training? Compare a sample slot's features between training replay and live computation.
2. **Stale asks**: Are you buying at inflated prices? Check `MIN_EDGE_MID` is active.
3. **Market regime change**: Has BTC volatility pattern changed? Check recent fold AUCs.
4. **Taker fee**: 2% fee requires sufficient edge. If model edge averages < 5%, fees eat profits.
5. **Cold start**: First few hours have degraded lag features. Don't judge P&L on first day.

---

## CI/CD Failures

| Error | Location | Fix |
|-------|----------|-----|
| `pytest` failures | CI test job | Fix the failing test; check `tests/test_features.py` |
| `validate_model.py` exit 1 | CI validate job | Check which sanity check failed (see logs) |
| `flyctl deploy` fails | Deploy job | Check Fly.io status page; verify `FLY_API_TOKEN` is valid |
| `uv sync` fails | CI test job | Check `pyproject.toml` for dependency conflicts |
| Deploy succeeds but app crashes | Fly.io runtime | `fly logs -a polymarket-maker-mm` — check for missing secrets or import errors |
