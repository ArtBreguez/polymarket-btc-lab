#!/usr/bin/env python3
"""
audit_live_features.py
======================
Conecta nos 2 WebSockets (Binance kline_1m + Polymarket CLOB) e no
REST do CLOB Polymarket, aguarda ~3 minutos de dados e calcula as
20 features do v29 champion. Reporta:

  - Quantas features chegaram com valor não-zero / não-neutro
  - Quais ficaram zeradas (possível problema de feed)
  - Sanity check nos valores (ranges razoáveis)
  - Health dos 2 WS (msgs/min, última mensagem)

Uso:
    cd ~/polymarket-btc-lab
    python3 scripts/audit_live_features.py
"""
import asyncio
import json
import logging
import math
import sys
import time
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import requests
import websockets

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("audit")

# ── constantes ────────────────────────────────────────────────────────────────
BINANCE_WS   = "wss://stream.binance.com:9443/stream?streams=btcusdt@kline_1m"
CLOB_WS_URL  = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
GAMMA_HOST   = "https://gamma-api.polymarket.com"
CLOB_URL     = "https://clob.polymarket.com"
OBSERVE_SECS = 60
AUDIT_SECS   = 200   # duração total da auditoria

V29_FEATURES = [
    "btc_inslot_ret",
    "ob_depth_ratio",
    "btc_pre_5m_ret",
    "ob_imbalance",
    "btc_inslot_range",
    "clob_spread_trend",
    "clob_spread_mean",
    "lag_ur_zscore_20",
    "btc_up_w1",
    "x_imb_x_ur",
    "prev_slot_up_ratio_5",
    "btc_size_disparity",
    "btc_dist_1k",
    "clob_mid_volatility",
    "btc_up_ratio_zscore_5s",
    "x_depth_x_vol",
    "btc_pre_30m_ret",
    "clob_ask_pressure",
    "btc_pre_1h_ret",
    "btc_tw_up_ratio",
]

# ── estado global ─────────────────────────────────────────────────────────────
_spot_candles: deque = deque(maxlen=400)   # (ts_sec, open, high, low, close, vol)
_spot_lock    = threading.Lock()
_spot_msgs    = 0
_spot_last_ts = 0.0

_clob_books: dict  = {}   # token_id → {best_bid, best_ask, mid, spread}
_clob_pcs  : dict  = {}   # token_id → list of PriceChangeEvent dicts
_clob_lock  = threading.Lock()
_clob_msgs  = 0
_clob_last_ts = 0.0

_target_token: str = ""   # UP token do mercado atual
_slot_ts: int = 0

# ── Binance WS ────────────────────────────────────────────────────────────────
async def run_binance_ws():
    global _spot_msgs, _spot_last_ts
    log.info("[Binance] conectando...")
    async for ws in websockets.connect(BINANCE_WS, ping_interval=20):
        try:
            async for raw in ws:
                _spot_msgs += 1
                _spot_last_ts = time.time()
                msg = json.loads(raw)
                k = msg.get("data", {}).get("k", {})
                if not k:
                    continue
                ts  = int(k["t"]) // 1000   # ms → seg
                o, h, l, c, v = float(k["o"]), float(k["h"]), float(k["l"]), float(k["c"]), float(k["v"])
                with _spot_lock:
                    # Atualiza ou appenda
                    if _spot_candles and _spot_candles[-1][0] == ts:
                        _spot_candles[-1] = (ts, o, h, l, c, v)
                    else:
                        _spot_candles.append((ts, o, h, l, c, v))
        except Exception as e:
            log.warning("[Binance] WS erro: %s — reconectando", e)
            await asyncio.sleep(3)

# ── CLOB WS ───────────────────────────────────────────────────────────────────
async def run_clob_ws():
    global _clob_msgs, _clob_last_ts
    log.info("[CLOB] conectando...")
    async for ws in websockets.connect(CLOB_WS_URL, ping_interval=20):
        try:
            # Subscreve após conectar
            if _target_token:
                sub = json.dumps({"assets_ids": [_target_token], "type": "market"})
                await ws.send(sub)
                log.info("[CLOB] subscrito token %s…%s", _target_token[:8], _target_token[-6:])

            async for raw in ws:
                _clob_msgs += 1
                _clob_last_ts = time.time()
                try:
                    events = json.loads(raw)
                    if not isinstance(events, list):
                        events = [events]
                    for ev in events:
                        _process_clob_event(ev)
                except Exception:
                    pass
        except Exception as e:
            log.warning("[CLOB] WS erro: %s — reconectando", e)
            await asyncio.sleep(3)

def _process_clob_event(ev: dict):
    token_id = ev.get("asset_id", "")
    etype    = ev.get("event_type", "")
    now      = time.time()
    with _clob_lock:
        if etype == "book":
            asks = ev.get("asks", [])
            bids = ev.get("bids", [])
            best_ask = float(asks[0]["price"]) if asks else 0.5
            best_bid = float(bids[0]["price"]) if bids else 0.5
            mid    = (best_ask + best_bid) / 2
            spread = best_ask - best_bid
            _clob_books[token_id] = {
                "ts": now, "best_ask": best_ask, "best_bid": best_bid,
                "mid": mid, "spread": spread
            }
        elif etype == "price_change":
            pcs = ev.get("price_changes", [])
            if token_id not in _clob_pcs:
                _clob_pcs[token_id] = []
            for pc in pcs:
                side  = pc.get("side", "")
                price = float(pc.get("price", 0.5))
                size  = float(pc.get("size", 0.0))
                _clob_pcs[token_id].append({"ts": now, "side": side, "price": price, "size": size})

# ── Binance REST — buscar histórico para pré-slot features ────────────────────
def fetch_binance_history(minutes: int = 260) -> list:
    """Busca até `minutes` velas 1m do Binance via REST."""
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": "1m", "limit": minutes}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        candles = []
        for c in r.json():
            ts  = int(c[0]) // 1000
            o, h, l, cl, v = float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])
            candles.append((ts, o, h, l, cl, v))
        return candles
    except Exception as e:
        log.error("[Binance REST] erro: %s", e)
        return []

# ── Mercado atual (Gamma) ─────────────────────────────────────────────────────
def fetch_current_market():
    """Encontra o slot ativo de 5 min mais próximo."""
    now = int(time.time())
    # slots alinhados a 5min
    slot = (now // 300) * 300
    for candidate in [slot, slot - 300, slot + 300]:
        slug = f"btc-updown-5m-{candidate}"
        try:
            r = requests.get(f"{GAMMA_HOST}/markets/slug/{slug}", timeout=5)
            if not r.ok:
                continue
            m = r.json()
            if not m or m.get("closed") or m.get("archived"):
                continue
            token_ids = m.get("clobTokenIds", "[]")
            if isinstance(token_ids, str):
                token_ids = json.loads(token_ids)
            if len(token_ids) < 2:
                continue
            log.info("[Gamma] mercado encontrado: slot=%d slug=%s", candidate, slug)
            log.info("[Gamma] UP token: %s…%s", token_ids[0][:8], token_ids[0][-6:])
            return candidate, token_ids[0], token_ids[1]
        except Exception as e:
            log.warning("[Gamma] %s: %s", slug, e)
    return None, None, None

# ── OB REST snapshot ──────────────────────────────────────────────────────────
def fetch_ob_rest(token_id: str) -> dict:
    """Busca orderbook via REST e computa ob_imbalance e ob_depth_ratio."""
    try:
        r = requests.get(f"{CLOB_URL}/book?token_id={token_id}", timeout=5)
        r.raise_for_status()
        book = r.json()
        asks = book.get("asks", [])
        bids = book.get("bids", [])
        if not asks or not bids:
            return {}
        best_ask = float(asks[0]["price"])
        best_bid = float(bids[0]["price"])
        mid = (best_ask + best_bid) / 2

        # Depth dentro de 5 cents do mid
        ask_d5c = sum(float(a["size"]) for a in asks if float(a["price"]) <= mid + 0.05)
        bid_d5c = sum(float(b["size"]) for b in bids if float(b["price"]) >= mid - 0.05)
        total_d5c = ask_d5c + bid_d5c + 1e-8
        imbalance = float(bid_d5c / total_d5c)

        # Depth total (todos os levels)
        ask_total = sum(float(a["size"]) for a in asks)
        bid_total = sum(float(b["size"]) for b in bids)
        total_all = ask_total + bid_total + 1e-8
        depth_ratio = float(bid_total / total_all)

        return {
            "ob_imbalance":    imbalance,
            "ob_depth_ratio":  depth_ratio,
            "ob_ask_depth_5c": ask_d5c,
            "ob_bid_depth_5c": bid_d5c,
            "ob_mid":          mid,
            "ob_spread":       best_ask - best_bid,
        }
    except Exception as e:
        log.warning("[OB REST] erro: %s", e)
        return {}

# ── Feature computation ───────────────────────────────────────────────────────
def compute_spot_features(slot_ts: int) -> dict:
    with _spot_lock:
        candles = list(_spot_candles)
    if not candles:
        log.warning("[Spot] buffer vazio")
        return {}

    ts_arr = np.array([c[0] for c in candles], dtype=np.int64)
    px_arr = np.array([c[4] for c in candles], dtype=np.float64)  # close
    hi_arr = np.array([c[2] for c in candles], dtype=np.float64)
    lo_arr = np.array([c[3] for c in candles], dtype=np.float64)
    vl_arr = np.array([c[5] for c in candles], dtype=np.float64)
    feat = {}

    # inslot ret
    idx_s = int(np.searchsorted(ts_arr, slot_ts, side="left"))
    idx_e = int(np.searchsorted(ts_arr, slot_ts + OBSERVE_SECS, side="right")) - 1
    idx_s = max(0, min(idx_s, len(px_arr) - 1))
    idx_e = max(0, min(idx_e, len(px_arr) - 1))
    if idx_e > idx_s and px_arr[idx_s] > 0:
        feat["btc_inslot_ret"] = float(px_arr[idx_e] / px_arr[idx_s] - 1)
        seg = px_arr[idx_s:idx_e + 1]
        feat["btc_inslot_vol"] = float(np.std(seg) / (np.mean(seg) + 1e-8)) if len(seg) > 1 else 0.0
        inslot_hi = hi_arr[idx_s:idx_e + 1]
        inslot_lo = lo_arr[idx_s:idx_e + 1]
        feat["btc_inslot_range"] = float((inslot_hi.max() - inslot_lo.min()) / (px_arr[idx_e] + 1e-8))
    else:
        feat["btc_inslot_ret"]   = 0.0
        feat["btc_inslot_vol"]   = 0.0
        feat["btc_inslot_range"] = 0.0

    obs_end_ts = slot_ts + OBSERVE_SECS
    idx_obs = np.searchsorted(ts_arr, obs_end_ts, side="right") - 1
    px_obs  = float(px_arr[idx_obs]) if idx_obs >= 0 else 0.0

    for w_s, lbl in [(300, "5m"), (1800, "30m"), (3600, "1h")]:
        idx_h = np.searchsorted(ts_arr, slot_ts - w_s, side="right") - 1
        px_h  = float(px_arr[idx_h]) if idx_h >= 0 else 0.0
        if px_h > 0 and px_obs > 0:
            feat[f"btc_pre_{lbl}_ret"] = float(px_obs / px_h - 1)
        else:
            feat[f"btc_pre_{lbl}_ret"] = 0.0

    # vol_1h
    mask_1h = (ts_arr >= slot_ts - 3600) & (ts_arr < slot_ts)
    seg_1h  = px_arr[mask_1h]
    if len(seg_1h) >= 3:
        rets = np.diff(seg_1h) / seg_1h[:-1]
        feat["btc_vol_1h"] = float(np.std(rets))
    else:
        feat["btc_vol_1h"] = 0.0

    # Normalizar por vol_1h (igual ao live_trader)
    _vol_norm = max(feat["btc_vol_1h"], 1e-4)
    for fk in ("btc_inslot_ret", "btc_pre_5m_ret", "btc_pre_30m_ret", "btc_pre_1h_ret", "btc_inslot_range"):
        if fk in feat:
            feat[fk] = feat[fk] / _vol_norm

    # dist_1k
    if px_obs > 0:
        px_k = px_obs / 1000
        feat["btc_dist_1k"] = float(min(px_k - math.floor(px_k), math.ceil(px_k) - px_k))
    else:
        feat["btc_dist_1k"] = 0.5

    return feat


def compute_clob_features(token_id: str, slot_ts: int) -> dict:
    """Computa features do CLOB WS (spread, mid, ask_pressure)."""
    with _clob_lock:
        books = list(v for k, v in _clob_books.items() if k == token_id)
        pcs   = list(_clob_pcs.get(token_id, []))

    feat = {}
    cutoff = slot_ts + 168  # [slot_ts, slot_ts+168s)
    window_start = slot_ts + 108  # últimos 60s da janela

    # Book snapshots dentro da janela
    valid_books = [b for b in books if slot_ts <= b["ts"] < cutoff]

    if len(valid_books) >= 2:
        spreads = [b["spread"] for b in valid_books]
        mids    = [b["mid"] for b in valid_books]
        feat["clob_spread_mean"]   = float(np.mean(spreads))
        feat["clob_spread_trend"]  = float(spreads[-1] - spreads[0])
        feat["clob_mid_volatility"] = float(np.std(mids)) if len(mids) > 1 else 0.0
    elif len(valid_books) == 1:
        feat["clob_spread_mean"]    = valid_books[0]["spread"]
        feat["clob_spread_trend"]   = 0.0
        feat["clob_mid_volatility"] = 0.0
    else:
        # Usa último book disponível como fallback
        all_books = [v for k, v in _clob_books.items() if k == token_id]
        if all_books:
            b = all_books[-1]
            feat["clob_spread_mean"]    = b["spread"]
            feat["clob_spread_trend"]   = 0.0
            feat["clob_mid_volatility"] = 0.0
        else:
            feat["clob_spread_mean"]    = 0.0
            feat["clob_spread_trend"]   = 0.0
            feat["clob_mid_volatility"] = 0.0

    # clob_ask_pressure: fração de price_changes que foram ASK (últimos 60s)
    window_pcs = [p for p in pcs if window_start <= p["ts"] < cutoff]
    if window_pcs:
        n_ask = sum(1 for p in window_pcs if p["side"] in ("SELL", "ASK"))
        feat["clob_ask_pressure"] = float(n_ask / (len(window_pcs) + 1e-8))
    else:
        feat["clob_ask_pressure"] = 0.5

    return feat


def compute_all_features(slot_ts: int, token_id: str, ticks_sim: list = None) -> dict:
    """
    Computa todas as 20 features do v29 usando dados ao vivo.
    ticks_sim: lista simulada de ticks para testar (vazia = sem ticks disponíveis)
    """
    feat = {}
    ticks = ticks_sim or []

    # ── Order flow (CLOB data-api ticks) ──────────────────────────────────────
    if ticks:
        vol_up = sum(t["size_usdc"] for t in ticks if t.get("outcome") == "Up")
        vol_dn = sum(t["size_usdc"] for t in ticks if t.get("outcome") == "Down")
        total  = vol_up + vol_dn + 1e-8
        cur_up_ratio = float(vol_up / total)

        up_tks = [t for t in ticks if t.get("outcome") == "Up"]
        dn_tks = [t for t in ticks if t.get("outcome") == "Down"]

        # btc_up_w1 (30–60s window)
        sub_w1 = [t for t in ticks if 30 <= t.get("t_sec", 0) < 60]
        if sub_w1:
            vu = sum(t["size_usdc"] for t in sub_w1 if t.get("outcome") == "Up")
            tt = sum(t["size_usdc"] for t in sub_w1) + 1e-8
            feat["btc_up_w1"] = float(vu / tt)
        else:
            feat["btc_up_w1"] = 0.5

        # btc_size_disparity
        avg_up_sz = float(sum(t["size_usdc"] for t in up_tks) / (len(up_tks) + 1e-8))
        avg_dn_sz = float(sum(t["size_usdc"] for t in dn_tks) / (len(dn_tks) + 1e-8))
        feat["btc_size_disparity"] = float(avg_up_sz - avg_dn_sz)

        # btc_tw_up_ratio
        t_arr  = np.array([t.get("t_sec", 0) for t in ticks], dtype=np.float64)
        sz_arr = np.array([t.get("size_usdc", 1.0) for t in ticks], dtype=np.float64)
        up_arr = np.array([1.0 if t.get("outcome") == "Up" else 0.0 for t in ticks])
        w_exp  = np.exp(-0.02 * (OBSERVE_SECS - t_arr))
        feat["btc_tw_up_ratio"] = float(np.sum(up_arr * sz_arr * w_exp) / (np.sum(sz_arr * w_exp) + 1e-9))
    else:
        cur_up_ratio = 0.5
        feat.update({
            "btc_up_w1": 0.5,
            "btc_size_disparity": 0.0,
            "btc_tw_up_ratio": 0.5,
        })

    # ── Lag features (sem histórico na auditoria = neutros) ───────────────────
    feat["prev_slot_up_ratio_5"] = 0.5    # neutro — sem ring buffer
    feat["btc_up_ratio_zscore_5s"] = 0.0  # neutro — sem histórico
    feat["lag_ur_zscore_20"] = 0.0        # neutro

    # ── Spot features ──────────────────────────────────────────────────────────
    spot = compute_spot_features(slot_ts)
    feat.update(spot)

    # ── OB REST snapshot ───────────────────────────────────────────────────────
    ob = fetch_ob_rest(token_id)
    feat.update(ob)

    # ── CLOB WS features ──────────────────────────────────────────────────────
    clob = compute_clob_features(token_id, slot_ts)
    feat.update(clob)

    # ── Cross features ────────────────────────────────────────────────────────
    feat["x_imb_x_ur"]   = feat.get("ob_imbalance", 0.5) * cur_up_ratio
    feat["x_depth_x_vol"] = feat.get("ob_depth_ratio", 1.0) * feat.get("btc_vol_1h", 0.0)

    return feat


# ── Auditoria ─────────────────────────────────────────────────────────────────
def print_audit_report(feat: dict, ws_stats: dict, slot_ts: int):
    now_str = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    slot_str = datetime.fromtimestamp(slot_ts, tz=timezone.utc).strftime("%H:%M:%S")
    t_into = int(time.time()) - slot_ts

    print("\n" + "="*65)
    print(f"  AUDITORIA DE FEATURES AO VIVO — {now_str}")
    print(f"  Slot: {slot_str} | t+{t_into}s dentro do slot")
    print("="*65)

    # WS health
    print("\n[WS HEALTH]")
    for name, stats in ws_stats.items():
        age = int(time.time() - stats["last_ts"]) if stats["last_ts"] > 0 else 999
        status = "✅ OK" if age < 30 and stats["msgs"] > 0 else "❌ PROBLEMA"
        print(f"  {name:<10} msgs={stats['msgs']:>5}  última={age:>3}s atrás  {status}")

    # Features
    print(f"\n[20 FEATURES v29]")
    print(f"  {'Feature':<30} {'Valor':>12}  Status")
    print(f"  {'-'*30} {'-'*12}  ------")

    NEUTRAL = {
        "btc_inslot_ret": 0.0, "btc_pre_5m_ret": 0.0, "btc_pre_30m_ret": 0.0,
        "btc_pre_1h_ret": 0.0, "btc_inslot_range": 0.0, "btc_vol_1h": 0.0,
        "ob_depth_ratio": 0.0, "ob_imbalance": 0.0,
        "clob_spread_mean": 0.0, "clob_spread_trend": 0.0,
        "clob_mid_volatility": 0.0, "clob_ask_pressure": 0.0,
    }
    EXPECTED_NEUTRAL = {  # features que ficam neutras sem histórico
        "lag_ur_zscore_20", "btc_up_ratio_zscore_5s", "prev_slot_up_ratio_5",
    }

    ok, warn, missing = 0, 0, 0
    for f in V29_FEATURES:
        val = feat.get(f)
        if val is None:
            status = "❌ AUSENTE"
            missing += 1
            val_str = "N/A"
        else:
            val_str = f"{val:>12.5f}"
            is_neutral = abs(val - NEUTRAL.get(f, 999)) < 1e-8 or (f == "btc_dist_1k" and abs(val - 0.5) < 1e-8)
            if f in EXPECTED_NEUTRAL:
                status = "⚠️  neutro (sem hist)"
                warn += 1
            elif is_neutral and f not in ("btc_tw_up_ratio", "btc_up_w1", "btc_size_disparity"):
                status = "⚠️  ZERO/NEUTRO"
                warn += 1
            else:
                status = "✅ OK"
                ok += 1
        print(f"  {f:<30} {val_str}  {status}")

    print(f"\n[RESUMO]  ✅ OK={ok}  ⚠️ neutro/warn={warn}  ❌ ausente={missing}  total={len(V29_FEATURES)}")

    # Sanity checks
    print("\n[SANITY CHECKS]")
    checks = [
        ("btc_inslot_ret normalizado ∈ [-10, 10]",
         abs(feat.get("btc_inslot_ret", 0.0)) <= 10),
        ("btc_dist_1k ∈ [0, 0.5]",
         0.0 <= feat.get("btc_dist_1k", -1) <= 0.5),
        ("ob_imbalance ∈ [0, 1]",
         0.0 <= feat.get("ob_imbalance", -1) <= 1.0),
        ("ob_depth_ratio ∈ [0, 1]",
         0.0 <= feat.get("ob_depth_ratio", -1) <= 1.0),
        ("clob_spread_mean >= 0",
         feat.get("clob_spread_mean", -1) >= 0),
        ("clob_ask_pressure ∈ [0, 1]",
         0.0 <= feat.get("clob_ask_pressure", -1) <= 1.0),
        ("btc_tw_up_ratio ∈ [0, 1]",
         0.0 <= feat.get("btc_tw_up_ratio", -1) <= 1.0),
        ("btc_pre_30m_ret normalizado ∈ [-50, 50]",
         abs(feat.get("btc_pre_30m_ret", 0.0)) <= 50),
        ("Spot buffer tem dados (vol_1h > 0)",
         feat.get("btc_vol_1h", 0.0) > 0),
        ("OB REST respondeu (ob_imbalance > 0 ou ob_depth_ratio > 0)",
         feat.get("ob_imbalance", 0.0) > 0 or feat.get("ob_depth_ratio", 0.0) > 0),
        ("CLOB WS tem dados (spread_mean > 0)",
         feat.get("clob_spread_mean", 0.0) > 0),
    ]
    for desc, passed in checks:
        icon = "✅" if passed else "❌"
        print(f"  {icon}  {desc}")

    print("="*65 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    global _target_token, _slot_ts

    print("\n" + "="*65)
    print("  AUDITORIA DE FEED AO VIVO — v29 20 FEATURES")
    print("="*65)

    # 1. Buscar mercado atual
    log.info("Buscando mercado ativo...")
    slot_ts, up_token, dn_token = fetch_current_market()
    if not slot_ts:
        log.error("Nenhum mercado ativo encontrado. Verifique a Gamma API.")
        sys.exit(1)
    _target_token = up_token
    _slot_ts      = slot_ts

    # 2. Pré-popular spot buffer via REST
    log.info("Baixando histórico Binance (260 velas 1m)...")
    hist = fetch_binance_history(260)
    if hist:
        with _spot_lock:
            for c in hist:
                _spot_candles.append(c)
        log.info("  %d velas carregadas — span %.0fm", len(hist),
                 (hist[-1][0] - hist[0][0]) / 60)
    else:
        log.warning("  Sem histórico REST — aguardando WS acumular")

    # 3. Iniciar os 2 WS em tasks paralelas
    log.info("Iniciando WebSockets...")
    t_start = time.time()

    binance_task = asyncio.create_task(run_binance_ws())
    clob_task    = asyncio.create_task(run_clob_ws())

    # 4. Aguardar dados chegarem (30s mínimo)
    log.info("Aguardando %ds de dados (WS + dados ao vivo)...", min(40, AUDIT_SECS // 2))
    await asyncio.sleep(40)

    # 5. Snapshot 1 — com dados WS acumulados
    log.info("Computando features (snapshot 1)...")
    feat1 = compute_all_features(slot_ts, up_token)
    ws_stats = {
        "Binance": {"msgs": _spot_msgs, "last_ts": _spot_last_ts},
        "CLOB":    {"msgs": _clob_msgs, "last_ts": _clob_last_ts},
    }
    print_audit_report(feat1, ws_stats, slot_ts)

    # 6. Aguardar mais dados e fazer snapshot 2
    remaining = AUDIT_SECS - 40
    if remaining > 30:
        log.info("Aguardando mais %ds para snapshot 2...", remaining)
        await asyncio.sleep(remaining)

        log.info("Computando features (snapshot 2)...")
        feat2 = compute_all_features(slot_ts, up_token)
        ws_stats2 = {
            "Binance": {"msgs": _spot_msgs, "last_ts": _spot_last_ts},
            "CLOB":    {"msgs": _clob_msgs, "last_ts": _clob_last_ts},
        }
        print_audit_report(feat2, ws_stats2, slot_ts)

        # Diff entre snapshots
        print("[DRIFT ENTRE SNAPSHOTS]")
        for f in V29_FEATURES:
            v1 = feat1.get(f, 0.0)
            v2 = feat2.get(f, 0.0)
            if abs(v2 - v1) > 0.001:
                print(f"  {f:<30}  {v1:>10.5f} → {v2:>10.5f}  Δ={v2-v1:+.5f}")
        print()

    # Cancelar WS
    binance_task.cancel()
    clob_task.cancel()
    log.info("Auditoria concluída.")


if __name__ == "__main__":
    asyncio.run(main())
