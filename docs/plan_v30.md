# Plano: Melhorar modelo BTC Polymarket (v30+)

**Data:** 2026-06-11
**Status:** Rascunho — aguarda aprovação

---

## Contexto atual

O backtest honesto do v29 revelou:
- **OOS AUC = 0.4985** — praticamente aleatório no holdout (4 dias pós-cutoff)
- **WF AUC = 0.7918** — enorme gap → overfitting severo ao período de treino
- **56 trades, win rate 55.4%, ROI 10.8%** — os filtros de confiança/edge salvam o resultado de trading, mas a base preditiva é fraca
- **Ticks do holdout = zeros** — btc_up_ratio, btc_momentum etc. ficaram neutros, subestimando o sinal real
- **Bot pausado** até ter modelo com AUC OOS genuinamente > ~0.55

---

## Diagnóstico das causas raiz

### 1. Overfitting ao ruído
- 20 features com ~79K mercados de treino — aparente abundância mas muita sobreposição temporal
- LightGBM sem regularização agressiva o suficiente
- Walk-forward CV não separou suficientemente os folds

### 2. Features com baixo sinal real
- `btc_up_ratio` e derivados são zeros no holdout (sem ticks históricos) → o modelo foi treinado com esse sinal mas não o tem no OOS
- `clob_*` features são de WS real-time → no treino vieram de snapshots gravados, no live/holdout diferem
- Features de lag (`prev_slot_up_ratio_*`) têm menos historicidade nos 4 dias de holdout

### 3. Dataset pequeno e concentrado
- ~22K mercados mas muitos são do mesmo período, correlacionados
- Holdout é apenas 4 dias — pouca diversidade de regime de BTC

---

## Plano de melhoria

### Fase 1 — Corrigir o dataset de treino (1-2 dias)

**1.1 Buscar ticks históricos do holdout**
- Implementar `fetch_holdout_ticks_modal.py` usando a API de dados históricos da Polymarket
- Salvar `holdout_ticks.parquet` no HF
- Backtest reroda com features de tick reais → baseline honesto melhor

**1.2 Expandir dataset total**
- Buscar mercados BTC de períodos anteriores (jan-mar 2026 e 2025 se disponível)
- Meta: 50K+ mercados resolvidos com ticks
- Mais diversidade de regime BTC (bull, bear, lateral)

**1.3 Qualidade dos ticks de treino**
- Auditar `ticks_btc_full_clean.parquet`: verificar se há contaminação cross-market
- Confirmar que `btc_up_ratio` no treino usa apenas ticks dentro de `[slot_ts, slot_ts+60s]`

---

### Fase 2 — Feature engineering (2-3 dias)

**2.1 Remover features fracas**
Candidatas a remoção/substituição (baixa variância OOS):
- `lag_1..5_outcome` → sempre 0.5 (desconhecido no inference) — remover
- `prev_slot_n_ticks_*` e `prev_slot_vol_*` → sempre 0.0 — remover ou substituir

**2.2 Novas features com sinal mais robusto**
- `btc_dominance_trend`: retorno BTC relativo ao mercado (via Binance BTC.D ou proxy)
- `polymarket_total_vol_5m`: volume total da plataforma no slot (sinal de atividade)
- `btc_round_number_dist`: distância a múltiplos de $500 (não só $1k)
- `market_age_slots`: quantos slots o mercado tem de vida (jovens vs maduros)
- `ob_spread_zscore`: spread normalizado pelo histórico recente do mercado
- `time_to_expiry_slots`: slots restantes até resolução (urgência)

**2.3 Feature de qualidade do sinal**
- `n_ob_snapshots`: quantos snapshots OB foram capturados (proxy de confiança)
- Mercados com muito poucos snapshots → mascarar ou down-weightar

---

### Fase 3 — Pipeline de treino mais rigoroso (2-3 dias)

**3.1 Purged GroupKFold com gap**
- Splits por `slot_ts` com gap de 24h entre train e validation
- Previne leakage temporal de features de lag
- Substituir o walk-forward atual por `PurgedGroupTimeSeriesSplit`

**3.2 Regularização agressiva**
- Aumentar `min_child_samples`, `reg_alpha`, `reg_lambda`
- Reduzir `num_leaves` e `max_depth`
- Adicionar noise canary feature: se o modelo usa → overfitting
- Calibração isotônica com holdout separado (não o mesmo de validação)

**3.3 Adversarial validation**
- Treinar classificador: "este mercado é de treino ou de test?"
- Se AUC > 0.6 → distribuição divergiu → identificar feature problemática
- Executar antes de qualquer treino novo

**3.4 Variance-penalized tuning**
- Objetivo: `mean_auc - 2 * std_auc` nos folds (não só mean_auc)
- Evita modelos com AUC médio alto mas instável entre folds

---

### Fase 4 — Target alternativo (1 dia, experimental)

O target atual é binário: `UP ou DOWN após 5min`. Considerar:
- **Target contínuo**: magnitude do movimento (regressão)
- **Target de edge**: só mercados onde `|P_true - P_ask| > 10%` (high-confidence)
- **Multi-class**: UP forte / incerto / DOWN forte

---

### Fase 5 — Holdout rolling (ongoing)

- A cada treino novo, o holdout avança (nunca entra no treino)
- Manter pelo menos 7 dias de holdout genuíno
- Meta de AUC OOS mínima para deploy: **>= 0.55** (consistente em 3 períodos)

---

## Critérios de sucesso para novo deploy

| Métrica | Mínimo para deploy |
|---|---|
| OOS AUC (holdout 7d) | >= 0.55 |
| OOS Brier | <= 0.24 |
| Win rate simulado | >= 54% |
| AUC estável em 3 folds OOS distintos | std < 0.03 |
| Noise canary importance | 0 |

---

## Ordem de execução

1. `fetch_holdout_ticks_modal.py` → backtest honesto com ticks reais
2. Adversarial validation no dataset atual → identificar features problemáticas
3. Expandir dataset (mais mercados históricos)
4. Reimplementar CV com purge + gap
5. Treinar v30 com regularização agressiva
6. Backtest v30 no holdout expandido (7d+)
7. Se AUC >= 0.55 em 3 períodos → deploy gradual (min=1, max=5 shares)

---

## Arquivos que vão mudar

- `scripts/fetch_holdout_ticks_modal.py` — novo
- `scripts/adversarial_validation.py` — novo
- `scripts/train_v30.py` — novo (baseado no train_v28 com CV purged)
- `scripts/backtest_v29.py` — fix menor (ticks)
- `docs/features.md` — atualizar com features v30
- `docs/holdout_policy.md` — estender holdout para 7d
