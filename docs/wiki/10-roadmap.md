# 10 — Roadmap

## Completed

### v19: L2 Orderbook Features — DONE
Real L2 orderbook features from pmdata poly_l2 data. AUC: 0.8979 → 0.9000. 40 features.

### v21: Ablation Study — DONE
Pruned 40 → 30 features. AUC maintained 0.9002. Champion at the time.

### v22–v27: OBS Window + Formula Alignment — DONE
A/B tests for OBS_SECS=60 vs 180. Formula fixes (tw_up_ratio, momentum, x_ob_drift). CLOB WS microstructure features added (v25). Pure real-time philosophy tested (v26–v27). None beat v21.

### v28: Full Train/Live Parity — DONE
Unified feature groups A–F. CLOB WS features (clob_spread_mean, clob_mid_volatility, clob_ask_pressure) integrated. Base for v29.

### v29: 20 Real-Time Features — DONE (CURRENT CHAMPION)
Pruned to 20 features. WF AUC=0.7918. Live on Fly.io (polymarket-maker-mm). Retornos brutos, CLOB window [0,60s). W4/L1, P&L=+$5.78.

### Position Sizing — DONE
Auto-sizing shares based on wallet balance. Linear scaling: 5 shares at $20 → 40 shares at $700. Risk cap: never spend >10% of balance per trade. Configurable via env vars (AUTO_SHARES, FIXED_SHARES, etc.).

### WebSocket Resilience — DONE
ws_manager.py with exponential backoff, active zombie detection, Binance REST fallback. 154 unit tests covering all components.

---

## In Progress

### v31: CLOB Window Unification [0,168s)

**Status**: Planned — próximo sprint

Mismatch atual: treino v29 usou ob_features_full.parquet ([108,168s)), live usa [0,60s).

Plano:
1. Fetch local pmdata [0,168s) para todos os mercados do dataset
2. Computar ob_features_v31.parquet com janela [0,168s)
3. Retreinar v31 com retornos brutos + CLOB [0,168s) — 200 trials Optuna sem timeout
4. Promover se AUC > 0.7918 (champion v29)
5. Atualizar live clob_features.py window_secs=168

Impacto esperado: maior janela CLOB captura mais eventos de microestrutura → melhor sinal.

---

## Planned

### Gate 4: Circuit Breaker

**Status**: Designed, not integrated in live loop

Stop trading when recent win rate drops below 40% over last 20 trades. Already in DataQualityGate code but not wired into the main trading loop.

### Automated Retraining

**Status**: Planned (depends on daily data expansion)

Pipeline:
1. Daily data expansion cron runs
2. If dataset has grown by >500 markets since last training:
   - Trigger `train_v21_modal.py` (or latest version)
   - Auto-promote if 2/3 metrics beat champion
   - Auto-deploy via Fly.io
3. Alert on regression (new model worse than champion)

---

## Research Ideas

### Regime Detection

BTC 5-minute markets behave differently in:
- High volatility (>2% daily moves) vs low volatility
- Trending vs ranging markets
- High liquidity (US hours) vs low liquidity (Asian hours)

Risk: With 22k samples, regime segmentation may fragment the data too much. Need 50k+ samples first.

### Kelly Criterion Sizing

Scale position size proportional to edge (model_prob - market_price) instead of linear balance scaling. Prerequisite: 200+ live trades to validate edge accuracy.

---

## Priority Order

| Priority | Item | Status | Expected Impact |
|----------|------|--------|-----------------|
| 1 | Renew pmdata API key | Blocked | Unlocks dataset expansion |
| 2 | Daily data expansion cron | Planned | +++ (more data = better model) |
| 3 | Gate 4 circuit breaker | Designed | Risk reduction |
| 4 | Automated retraining | Planned | Operational efficiency |
| 5 | Regime detection | Research | Unknown, needs 50k+ samples |
| 6 | Kelly criterion sizing | Research | Better capital efficiency |
