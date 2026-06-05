# v21 Live Deployment Session — 2026-06-05

## Changes Deployed

### 1. Active Zombie Detection (ws_manager.py)
Before this fix, CLOB WS disconnected every ~65s on one-sided markets (asks >= 0.97):
- Root cause: No CLOB *data* messages flow when market is one-sided, but server pings/pongs still flow at protocol level. `async for raw in ws` only sees data messages, not pings. So zombie detector (60s timeout) killed healthy connections repeatedly.
- Fix: Before killing, send protocol-level PING and wait for PONG (10s timeout). If PONG returns, connection is alive — just no data. Reset `last_message_at` and continue. Only kill if PING also fails.
- Also raised zombie_timeout from 60s to 120s for CLOB.
- Result: disconnects dropped from ~1/min (16 in 5min) to 0 in 6+ minutes.

### 2. v21 Model Promotion (30 features)
- Ablation study: tested 40/35/30 feature variants
- 30-feature variant: AUC=0.9002, Brier=0.1290, Acc=81.34% (3/3 gate)
- Uploaded to HuggingFace, bot restarted to pull new model

### 3. Live Feature Pruning (live_trader.py, -137 lines)
Removed computation of features not used by v21:
- Tick features: tw_up, vwap_trend, vwmom, tick_accel, vol_accel, momentum_vol_sync, n_ticks, vol_up, vol_dn, vol_ratio, avg_size
- History features: all zscores, realized_vol, lag_outcomes, lag_streak, prev_slot_n_ticks/vol
- Calendar: hour_sin/cos, dow_sin/cos, hour_x_tw_ur
- Cross: x_spread_x_vol, x_fill_imb_x_buy
- Kept intermediaries needed by remaining features (e.g., up_ratio_stability for signal_conviction)

### 4. Feature Count Logging Bug Fix
- Bug: `sum(1 for v in feat.values() if v != 0.0)` counted ALL computed features (54) including unused spot/OB extras
- Showed "50/54 non-zero" which was misleading — the 54 denominator didn't match the model's 30 features
- Fix: `sum(1 for f in features if feat.get(f, 0.0) != 0.0)` counts only MODEL features
- Now correctly shows "29/30 non-zero"
- Lesson: After pruning features from the model, also update any log/monitoring code that counts features. Always iterate over the model's feature list, not the full computed dict.

### 5. Copytrade Bot Diagnosis
- Bot functioning correctly — target (Respectful-Clan) hasn't traded since June 1 (4 days)
- 9 open positions correctly blocking re-entry
- 91 expired positions correctly filtered (curPrice <= 0.001)
- Balance up to $197.81 from $162.46
- Key insight: Respectful-Clan is a burst trader (2-3h sessions), days without trades are normal

### 6. Model Restart for HF Update
- `fly machines restart <id> --app <name>` forces model re-download from HuggingFace
- Needed because bot only downloads champion.pkl at startup
- Verified with `grep "Model loaded"` in logs: "Model loaded: 30 features, WF AUC=0.900"

### 7. Documentation Overhaul
Updated all docs (repo README, EXPERIMENTS.md, wiki, HF model card, feature_definitions.json, training_config.json) from v8/v18/v19 references to v21.

## Deployment Sequence
1. Patch ws_manager.py (active zombie detection) → tests pass → commit → deploy from deploy/ dir
2. Restart machine to pull v21 model from HF
3. Patch live_trader.py (prune features) → commit → deploy
4. Fix feature count logging → commit → deploy
5. Verify: 29/30 non-zero, 0 disconnects, 0 zombies
