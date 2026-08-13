"""Atualiza o model card HF com métricas reais do backtest OOS."""
import modal

image = modal.Image.debian_slim(python_version="3.12").pip_install("huggingface_hub>=0.26")
app = modal.App("update-model-card", image=image)

CARD = """---
language: en
tags:
  - polymarket
  - prediction-market
  - btc
  - lightgbm
  - time-series
license: mit
---

# polymarket-btc-model

Modelo LightGBM para prever se o mercado BTC de 5 minutos da Polymarket resolverá UP ou DOWN.

## Champion: v29_20f_rt

| Métrica | Valor |
|---|---|
| WF AUC (treino, walk-forward) | 0.7918 |
| OOS AUC (holdout 4d genuíno) | **0.4985** |
| OOS Brier | 0.2656 |
| OOS Acc | 48.2% |
| Trades simulados (holdout) | 56 / 1172 (4.8%) |
| Win rate simulado | 55.4% |
| ROI simulado | 10.8% |
| Sharpe simulado | 11.12 |

> ⚠️ **OOS AUC ≈ 0.50** — o modelo tem poder preditivo fraco no holdout genuíno.
> O gap WF→OOS indica overfitting ao período de treino.
> **Bot pausado** até v30 atingir OOS AUC >= 0.55 consistente.

## Features (20)

| # | Feature | Grupo |
|---|---|---|
| 1 | btc_inslot_ret | spot |
| 2 | ob_depth_ratio | order book |
| 3 | ob_imbalance | order book |
| 4 | btc_pre_5m_ret | spot |
| 5 | clob_spread_mean | CLOB |
| 6 | clob_spread_trend | CLOB |
| 7 | btc_inslot_range | spot |
| 8 | ob_total_depth | order book |
| 9 | x_imb_x_ur | cross |
| 10 | btc_up_w1 | ticks |
| 11 | x_depth_x_vol | cross |
| 12 | clob_mid_volatility | CLOB |
| 13 | lag_ur_zscore_20 | lag |
| 14 | prev_slot_up_ratio_3 | lag |
| 15 | prev_slot_up_ratio_5 | lag |
| 16 | btc_size_disparity | ticks |
| 17 | btc_dist_1k | spot |
| 18 | clob_ask_pressure | CLOB |
| 19 | btc_up_ratio_zscore_5s | ticks |
| 20 | btc_spot_vol_ratio | spot |

## Leakage audit

9 features removidas do v28 por leakage temporal (capturadas em t=108-168s, além do limite de 60s):
`ob_mid`, `ob_imbalance_end`, `ob_spread_end`, `ob_depth_change`, `ob_imb_momentum`,
`clob_mid_velocity`, `ob_weighted_imb`, `ob_bid_depth_5c`, `ob_ask_depth_5c`

## Holdout policy

- **Cutoff v29:** 2026-06-06 19:10 UTC
- **Holdout:** 2026-06-06 19:15 → 2026-06-10 22:25 (4 dias, 1191 mercados)
- Dados do holdout nunca entram no treino de versões futuras

## Plano v30

Ver `.hermes/plans/2026-06-11_010000-v30-model-improvement.md` no repo.
Foco: adversarial validation, purged CV com gap, regularização agressiva, mais dados históricos.
Meta: OOS AUC >= 0.55 em 3 períodos distintos antes de qualquer deploy.

## Dataset

- `artbreguez/polymarket-btc-updown` — 79K mercados, 616 resolvidos (dataset público)
- Treino: mercados até 2026-06-06 (~22K resolvidos)
- Ticks: Polymarket data-api (histórico até ~7 dias)
- Spot: Binance 1m candles (BTC/USDT)
- OB: snapshots capturados via pmdata em t=5-60s do slot
"""

@app.function(secrets=[modal.Secret.from_name("hf-token")])
def update():
    import os
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.upload_file(
        path_or_fileobj=CARD.encode(),
        path_in_repo="README.md",
        repo_id="artbreguez/polymarket-btc-model",
        repo_type="model",
        commit_message="Update model card: real OOS metrics (AUC=0.4985), v30 plan",
    )
    print("Model card atualizado OK")

@app.local_entrypoint()
def main():
    update.remote()
