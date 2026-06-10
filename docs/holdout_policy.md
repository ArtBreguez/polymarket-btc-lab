# Holdout Policy — BTC 5min Model

## Regra

O holdout é o único gate de validação real. Nunca entra no treino.

**Cutoff permanente:** `slot_ts > TRAIN_CUTOFF_TS` do modelo vigente.

## Implementação

### Antes de cada treino

1. Definir `HOLDOUT_CUTOFF` = último slot do dataset de treino anterior
2. Qualquer mercado com `slot_ts > HOLDOUT_CUTOFF` vai para `holdout_markets.csv`
3. O script de treino aplica o cutoff explicitamente — nunca usa mercados pós-cutoff
4. Após treino, rodar `backtest_v{N}.py` APENAS nos dados do holdout

### Arquivos HF (single source of truth)

| Arquivo | Conteúdo | Usado em |
|---|---|---|
| `data/all_markets.csv` | Todos os mercados de treino | treino |
| `data/new_markets.csv` | Mercados adicionais de treino | treino |
| `data/holdout_markets.csv` | Mercados pós-cutoff (nunca vistos) | backtest apenas |
| `data/holdout_ticks.parquet` | Ticks do holdout | backtest apenas |
| `data/holdout_ob_features.parquet` | OB features do holdout | backtest apenas |
| `data/ticks_btc_full_clean.parquet` | Ticks de treino | treino |
| `data/ob_features_full.parquet` | OB features de treino | treino |

### Cutoffs por versão

| Modelo | Train até | Holdout a partir de |
|---|---|---|
| v29 | 2026-06-06 19:10 UTC (slot=1780773000) | 2026-06-07 00:00 UTC |
| v30 | TBD | TBD |

## Regras anti-leakage

1. **Holdout nunca entra no treino** — mesmo se o modelo não convergir
2. **Holdout nunca é usado para tuning** — nem threshold, nem features
3. **Refresh do holdout**: antes de cada ciclo de treino, rodar `fetch_holdout_pipeline.py`
   para coletar mercados novos. Os novos ficam como holdout do próximo modelo
4. **Walk-forward real**: quando v30 for treinado, o holdout do v29 (7-10 jun) pode
   entrar como dados de treino do v30, e o holdout do v30 será de uma data posterior

## Script de validação

```bash
modal run scripts/backtest_v29.py   # usa holdout_markets.csv, não all_markets.csv
```

O backtest carrega o bundle do HF, lê os features do holdout e simula trades.
Nunca há overlap com dados de treino.
