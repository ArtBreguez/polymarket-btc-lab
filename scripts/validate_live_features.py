#!/usr/bin/env python3
"""
validate_live_features.py
=========================
Replica exatamente o fluxo do live_trader para computar as 20 features
do v29 ao vivo e valida cada uma individualmente.

Fontes de dados:
  1. Binance WS  (kline_1m) → spot features
  2. CLOB WS     (book + price_change) → clob_spread_*, clob_ask_pressure
  3. CLOB REST   (/book) → ob_imbalance, ob_depth_ratio
  4. Data-API    (/trades) → btc_up_ratio, btc_up_w1, btc_size_disparity,
                              btc_tw_up_ratio, x_imb_x_ur, x_depth_x_vol
  5. Ring buffer (simulado com 1 slot) → prev_slot_up_ratio_*, lag_ur_zscore_*

O script:
  - Espera o início de um slot (ou usa o slot atual se t < 120s)
  - Aguarda t=170s (janela de entrada do bot)
  - Busca todos os dados e computa as features
  - Imprime relatório completo com valor, fonte e status de cada feature
"""
import asyncio
import json
import logging
import math
import sys
import time
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
import websockets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("validate")

# ── constantes (espelham live_trader) ─────────────────────────────────────────
BINANCE_WS  = "wss://stream.binance.com:9443/stream?streams=btcusdt@kline_1m"
CLOB_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
GAMMA_HOST  = "https://gamma-api.polymarket.com"
CLOB_URL    = "https://clob.polymarket.com"
DATA_API    = "https://data-api.polymarket.com"
OBSERVE_SECS = 60
ENTER_AT     = 170   # segundos dentro do slot
MAX_PAGES    = 8

V29_FEATURES = [
    "btc_inslot_ret", "ob_depth_ratio", "btc_pre_5m_ret", "ob_imbalance",
    "btc_inslot_range", "clob_spread_trend", "clob_spread_mean",
    "lag_ur_zscore_20", "btc_up_w1", "x_imb_x_ur", "prev_slot_up_ratio_5",
    "btc_size_disparity", "btc_dist_1k", "clob_mid_volatility",
    "btc_up_ratio_zscore_5s", "x_depth_x_vol", "btc_pre_30m_ret",
    "clob_ask_pressure", "btc_pre_1h_ret", "btc_tw_up_ratio",
]

# Fonte de cada feature (para diagnóstico)
FEATURE_SOURCE = {
    "btc_inslot_ret":        "Binance WS",
    "btc_pre_5m_ret":        "Binance WS",
    "btc_pre_30m_ret":       "Binance WS",
    "btc_pre_1h_ret":        "Binance WS",
    "btc_inslot_range":      "Binance WS",
    "btc_dist_1k":           "Binance WS",
    "ob_imbalance":          "CLOB REST",
    "ob_depth_ratio":        "CLOB REST",
    "clob_spread_mean":      "CLOB WS",
    "clob_spread_trend":     "CLOB WS",
    "clob_mid_volatility":   "CLOB WS",
    "clob_ask_pressure":     "CLOB WS",
    "btc_up_ratio":          "Data-API",
    "btc_up_w1":             "Data-API",
    "btc_size_disparity":    "Data-API",
    "btc_tw_up_ratio":       "Data-API",
    "x_imb_x_ur":            "CLOB REST × Data-API",
    "x_depth_x_vol":         "CLOB REST × Binance WS",
    "lag_ur_zscore_20":      "Ring buffer (neutro sem hist)",
    "prev_slot_up_ratio_5":  "Ring buffer (neutro sem hist)",
    "btc_up_ratio_zscore_5s":"Ring buffer (neutro sem hist)",
}

# ── estado compartilhado ──────────────────────────────────────────────────────
_spot_candles: deque = deque(maxlen=500)
_spot_lock    = threading.Lock()
_spot_stats   = {"msgs": 0, "last": 0.0, "errors": 0}

_clob_book_snaps: dict = {}   # token_id → list of {ts, best_ask, best_bid, mid, spread}
_clob_pc_events : dict = {}   # token_id → list of {ts, side, price, size}
_clob_lock      = threading.Lock()
_clob_stats     = {"msgs": 0, "last": 0.0, "book_events": 0, "pc_events": 0}

_target_tokens: list = []
_slot_start: int = 0

# ── Binance WS ────────────────────────────────────────────────────────────────
async def run_binance():
    async for ws in websockets.connect(BINANCE_WS, ping_interval=20):
        try:
            async for raw in ws:
                _spot_stats["msgs"] += 1
                _spot_stats["last"] = time.time()
                msg = json.loads(raw)
                k = msg.get("data", {}).get("k", {})
                if not k:
                    continue
                ts = int(k["t"]) // 1000
                c = (ts, float(k["o"]), float(k["h"]), float(k["l"]),
                     float(k["c"]), float(k["v"]))
                with _spot_lock:
                    if _spot_candles and _spot_candles[-1][0] == ts:
                        _spot_candles[-1] = c
                    else:
                        _spot_candles.append(c)
        except Exception as e:
            _spot_stats["errors"] += 1
            log.warning("[Binance] %s — reconectando", e)
            await asyncio.sleep(2)

# ── CLOB WS ───────────────────────────────────────────────────────────────────
async def run_clob():
    async for ws in websockets.connect(CLOB_WS_URL, ping_interval=20):
        try:
            if _target_tokens:
                await ws.send(json.dumps({
                    "assets_ids": _target_tokens,
                    "type": "market"
                }))
                log.info("[CLOB WS] subscrito %d tokens", len(_target_tokens))
            async for raw in ws:
                _clob_stats["msgs"] += 1
                _clob_stats["last"] = time.time()
                try:
                    evs = json.loads(raw)
                    if not isinstance(evs, list):
                        evs = [evs]
                    for ev in evs:
                        _process_clob(ev)
                except Exception:
                    pass
        except Exception as e:
            log.warning("[CLOB WS] %s — reconectando", e)
            await asyncio.sleep(2)

def _process_clob(ev: dict):
    tid = ev.get("asset_id", "")
    etype = ev.get("event_type", "")
    now = time.time()
    with _clob_lock:
        if etype == "book":
            asks = ev.get("asks", [])
            bids = ev.get("bids", [])
            if asks and bids:
                # BUG FIX: asks em ordem DESC — best ask é o MENOR (último ou sorted)
                asks_sorted = sorted(asks, key=lambda x: float(x["price"]))
                bids_sorted = sorted(bids, key=lambda x: float(x["price"]), reverse=True)
                ba = float(asks_sorted[0]["price"])
                bb = float(bids_sorted[0]["price"])
                snap = {"ts": now, "best_ask": ba, "best_bid": bb,
                        "mid": (ba+bb)/2, "spread": ba-bb}
                _clob_book_snaps.setdefault(tid, []).append(snap)
                _clob_stats["book_events"] += 1
        elif etype == "price_change":
            for pc in ev.get("price_changes", []):
                _clob_pc_events.setdefault(tid, []).append({
                    "ts": now,
                    "side": pc.get("side", ""),
                    "price": float(pc.get("price", 0.5)),
                    "size": float(pc.get("size", 0.0)),
                })
                _clob_stats["pc_events"] += 1

# ── Binance REST bootstrap ────────────────────────────────────────────────────
def bootstrap_spot(minutes: int = 300):
    log.info("[Binance REST] carregando %d velas históricas...", minutes)
    try:
        r = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": "BTCUSDT", "interval": "1m", "limit": minutes},
            timeout=15,
        )
        r.raise_for_status()
        with _spot_lock:
            for c in r.json():
                _spot_candles.append((
                    int(c[0])//1000,
                    float(c[1]), float(c[2]), float(c[3]),
                    float(c[4]), float(c[5]),
                ))
        log.info("[Binance REST] %d velas  span=%.0fm",
                 len(_spot_candles),
                 (_spot_candles[-1][0] - _spot_candles[0][0]) / 60)
    except Exception as e:
        log.error("[Binance REST] falhou: %s", e)

# ── Gamma: mercado ativo ───────────────────────────────────────────────────────
def find_active_market():
    now = int(time.time())
    slot = (now // 300) * 300
    for candidate in [slot, slot + 300, slot - 300]:
        slug = f"btc-updown-5m-{candidate}"
        try:
            r = requests.get(f"{GAMMA_HOST}/markets/slug/{slug}", timeout=8)
            if not r.ok:
                continue
            m = r.json()
            if not m or m.get("closed") or m.get("archived"):
                continue
            ids = m.get("clobTokenIds", "[]")
            if isinstance(ids, str):
                ids = json.loads(ids)
            if len(ids) < 2:
                continue
            end_str = m.get("endDate") or m.get("endDateIso", "")
            end_ts = int(datetime.fromisoformat(
                end_str.replace("Z", "+00:00")).timestamp()) if end_str else candidate + 300
            if end_ts < now + 90:
                log.info("[Gamma] slot=%d já está encerrando (%ds restantes) — pulando",
                         candidate, end_ts - now)
                continue
            log.info("[Gamma] slot=%d  YES=%s…%s  termina em %ds",
                     candidate, ids[0][:8], ids[0][-6:], end_ts - now)
            return candidate, ids[0], ids[1]
        except Exception as e:
            log.warning("[Gamma] %s: %s", slug, e)
    return None, None, None

# ── Data-API: ticks inslot ────────────────────────────────────────────────────
def _fetch_token_trades(token_id: str, outcome: str, slot_ts: int, slug: str) -> list:
    trades = []
    for page in range(MAX_PAGES):
        try:
            r = requests.get(
                f"{DATA_API}/trades",
                params={"asset": token_id, "limit": 500, "offset": page * 500},
                timeout=8,
            )
            if not r.ok:
                break
            batch = r.json()
            if not batch:
                break
            filtered = []
            for t in batch:
                # BUG FIX: campo correto é 'slug' ou 'eventSlug', não 'market'
                t_slug = t.get("slug") or t.get("eventSlug") or t.get("market", "")
                if t_slug != slug:
                    continue
                try:
                    ts_raw = t.get("timestamp", t.get("created_at", 0))
                    ts_sec = int(float(ts_raw))
                except Exception:
                    continue
                t_sec = ts_sec - slot_ts
                if not (0 <= t_sec < OBSERVE_SECS):
                    continue
                filtered.append({
                    "t_sec": t_sec,
                    "outcome": outcome,
                    "size_usdc": float(t.get("usdcSize", t.get("size", 0))),
                    "side": t.get("side", ""),
                })
            trades.extend(filtered)
            if len(batch) < 500:
                break
        except Exception as e:
            log.warning("[Data-API] token=%s page=%d: %s", token_id[:12], page, e)
            break
    return trades

def fetch_inslot_trades(yes_token: str, no_token: str, slot_ts: int) -> list:
    slug = f"btc-updown-5m-{slot_ts}"
    log.info("[Data-API] buscando trades slug=%s...", slug)
    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_yes = ex.submit(_fetch_token_trades, yes_token, "Up",   slot_ts, slug)
        fut_no  = ex.submit(_fetch_token_trades, no_token,  "Down", slot_ts, slug)
        all_trades = []
        for fut in as_completed([fut_yes, fut_no]):
            try:
                all_trades.extend(fut.result(timeout=30))
            except Exception as e:
                log.warning("[Data-API] future error: %s", e)
    log.info("[Data-API] %d trades inslot encontrados", len(all_trades))
    return all_trades

# ── CLOB REST: OB snapshot com retry ─────────────────────────────────────────
def fetch_ob_rest(token_id: str, retries: int = 3, delay: float = 1.0) -> dict | None:
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(
                f"{CLOB_URL}/book",
                params={"token_id": token_id},
                timeout=5,
            )
            if not r.ok:
                log.warning("[OB REST] attempt %d HTTP %d", attempt, r.status_code)
                time.sleep(delay)
                continue
            book = r.json()
            asks = sorted(book.get("asks", []), key=lambda x: float(x["price"]))
            bids = sorted(book.get("bids", []), key=lambda x: float(x["price"]), reverse=True)
            if not asks:
                log.warning("[OB REST] attempt %d: asks vazios — bids=%d", attempt, len(bids))
                time.sleep(delay)
                continue
            if not bids:
                log.warning("[OB REST] attempt %d: bids vazios — asks=%d", attempt, len(asks))
                time.sleep(delay)
                continue
            ba, bb = float(asks[0]["price"]), float(bids[0]["price"])
            ba_sz, bb_sz = float(asks[0]["size"]), float(bids[0]["size"])
            mid    = (ba + bb) / 2
            spread = ba - bb
            imb    = (bb_sz - ba_sz) / (bb_sz + ba_sz + 1e-8)

            # Depth a 5 cents do mid
            ask_d5c = sum(float(a["size"]) for a in asks if float(a["price"]) <= mid + 0.05)
            bid_d5c = sum(float(b["size"]) for b in bids if float(b["price"]) >= mid - 0.05)
            total_d5c = ask_d5c + bid_d5c + 1e-8

            ask_tot = sum(float(a["size"]) for a in asks)
            bid_tot = sum(float(b["size"]) for b in bids)
            total_all = ask_tot + bid_tot + 1e-8

            return {
                "ob_imbalance":    float(bid_d5c / total_d5c),
                "ob_depth_ratio":  float(bid_tot / total_all),
                "ob_ask_depth_5c": ask_d5c,
                "ob_bid_depth_5c": bid_d5c,
                "ob_mid":          mid,
                "ob_spread":       spread,
                "ob_total_depth":  total_all,
                "best_ask":        ba,
                "best_bid":        bb,
                "n_ask_levels":    len(asks),
                "n_bid_levels":    len(bids),
            }
        except Exception as e:
            log.warning("[OB REST] attempt %d exception: %s", attempt, e)
            time.sleep(delay)
    log.error("[OB REST] TODAS as %d tentativas falharam para %s", retries, token_id[:20])
    return None

# ── Spot features ─────────────────────────────────────────────────────────────
def compute_spot_features(slot_ts: int) -> dict:
    with _spot_lock:
        candles = list(_spot_candles)
    if not candles:
        return {"_error": "spot buffer vazio"}

    ts_arr = np.array([c[0] for c in candles], dtype=np.int64)
    px_arr = np.array([c[4] for c in candles], dtype=np.float64)
    hi_arr = np.array([c[2] for c in candles], dtype=np.float64)
    lo_arr = np.array([c[3] for c in candles], dtype=np.float64)
    vl_arr = np.array([c[5] for c in candles], dtype=np.float64)

    feat = {}

    # inslot_ret / inslot_range
    idx_s = int(np.searchsorted(ts_arr, slot_ts, "left"))
    idx_e = int(np.searchsorted(ts_arr, slot_ts + OBSERVE_SECS, "right")) - 1
    idx_s = max(0, min(idx_s, len(px_arr) - 1))
    idx_e = max(0, min(idx_e, len(px_arr) - 1))
    if idx_e > idx_s and px_arr[idx_s] > 0:
        feat["btc_inslot_ret"] = float(px_arr[idx_e] / px_arr[idx_s] - 1)
        seg = px_arr[idx_s:idx_e + 1]
        feat["btc_inslot_vol"] = float(np.std(seg) / (np.mean(seg) + 1e-8))
        feat["btc_inslot_range"] = float(
            (hi_arr[idx_s:idx_e+1].max() - lo_arr[idx_s:idx_e+1].min())
            / (px_arr[idx_e] + 1e-8)
        )
        feat["_spot_inslot_candles"] = idx_e - idx_s + 1
    else:
        feat.update({"btc_inslot_ret": 0.0, "btc_inslot_vol": 0.0,
                     "btc_inslot_range": 0.0, "_spot_inslot_candles": 0})

    obs_end = slot_ts + OBSERVE_SECS
    idx_obs = max(0, int(np.searchsorted(ts_arr, obs_end, "right")) - 1)
    px_obs  = float(px_arr[idx_obs]) if idx_obs >= 0 else 0.0

    # Pre-slot returns
    for w_s, lbl in [(300, "5m"), (1800, "30m"), (3600, "1h")]:
        idx_h = max(0, int(np.searchsorted(ts_arr, slot_ts - w_s, "right")) - 1)
        px_h  = float(px_arr[idx_h]) if idx_h >= 0 else 0.0
        feat[f"btc_pre_{lbl}_ret"] = float(px_obs / px_h - 1) if px_h > 0 and px_obs > 0 else 0.0
        feat[f"_spot_{lbl}_candles"] = int(np.sum((ts_arr >= slot_ts - w_s) & (ts_arr < slot_ts)))

    # vol_1h
    mask_1h = (ts_arr >= slot_ts - 3600) & (ts_arr < slot_ts)
    seg_1h  = px_arr[mask_1h]
    feat["btc_vol_1h"] = float(np.std(np.diff(seg_1h) / seg_1h[:-1])) if len(seg_1h) >= 3 else 0.0

    # Normalizar por vol_1h (=live_trader)
    vol_norm = max(feat["btc_vol_1h"], 1e-4)
    for fk in ("btc_inslot_ret", "btc_pre_5m_ret", "btc_pre_30m_ret",
               "btc_pre_1h_ret", "btc_inslot_range"):
        if fk in feat:
            feat[fk] = feat[fk] / vol_norm

    # dist_1k
    if px_obs > 0:
        pk = px_obs / 1000
        feat["btc_dist_1k"] = float(min(pk - math.floor(pk), math.ceil(pk) - pk))
        feat["_btc_price"]  = px_obs
    else:
        feat["btc_dist_1k"] = 0.5

    return feat

# ── CLOB WS features ─────────────────────────────────────────────────────────
def compute_clob_ws_features(token_id: str, slot_ts: int) -> dict:
    with _clob_lock:
        books = list(_clob_book_snaps.get(token_id, []))
        pcs   = list(_clob_pc_events.get(token_id, []))

    feat = {}
    # BUG FIX: janela baseada em wall-clock absoluto, não slot_ts+offset
    # O script coleta durante ~78-170s — todos os eventos caem dentro da
    # janela de observação [slot_ts, slot_ts+168s).
    # Filtramos por wall-clock: eventos gerados após o início do slot.
    slot_wall_start = float(slot_ts)          # slot_ts é unix timestamp
    window_end_wall = slot_wall_start + 168.0
    window_start_wall = slot_wall_start + 108.0  # últimos 60s

    valid_books = [b for b in books if slot_wall_start <= b["ts"] <= window_end_wall]
    # Se não há books na janela restrita, usar todos coletados (o script iniciou depois do slot)
    if not valid_books:
        valid_books = books
        feat["_clob_fallback_all_books"] = True

    feat["_clob_book_snaps"] = len(valid_books)
    feat["_clob_pc_events"]  = len(pcs)  # todos coletados desde subscrição

    if len(valid_books) >= 2:
        spreads = [b["spread"] for b in valid_books]
        mids    = [b["mid"]    for b in valid_books]
        feat["clob_spread_mean"]    = float(np.mean(spreads))
        feat["clob_spread_trend"]   = float(spreads[-1] - spreads[0])
        feat["clob_mid_volatility"] = float(np.std(mids))
    elif valid_books:
        b = valid_books[-1]
        feat["clob_spread_mean"]    = b["spread"]
        feat["clob_spread_trend"]   = 0.0
        feat["clob_mid_volatility"] = 0.0
    else:
        feat.update({"clob_spread_mean": 0.0, "clob_spread_trend": 0.0,
                     "clob_mid_volatility": 0.0, "_clob_no_data": True})

    # ask_pressure: usar todos os pc events coletados (janela é desde a subscrição)
    win_pcs = [p for p in pcs if p["ts"] >= window_start_wall]
    if not win_pcs:
        win_pcs = pcs  # fallback: usar todos
    if win_pcs:
        n_ask = sum(1 for p in win_pcs if p["side"] in ("SELL", "ASK"))
        feat["clob_ask_pressure"] = float(n_ask / len(win_pcs))
        feat["_clob_ask_win_pcs"]  = len(win_pcs)
    else:
        feat["clob_ask_pressure"] = 0.5
        feat["_clob_ask_pressure_neutral"] = True

    return feat

# ── Tick features (Data-API) ─────────────────────────────────────────────────
def compute_tick_features(ticks: list) -> dict:
    feat = {"_n_ticks": len(ticks)}
    if not ticks:
        return {**feat, "btc_up_ratio": 0.5, "btc_up_w1": 0.5,
                "btc_size_disparity": 0.0, "btc_tw_up_ratio": 0.5}

    vol_up = sum(t["size_usdc"] for t in ticks if t["outcome"] == "Up")
    vol_dn = sum(t["size_usdc"] for t in ticks if t["outcome"] == "Down")
    total  = vol_up + vol_dn + 1e-8
    cur_up = float(vol_up / total)
    feat["btc_up_ratio"] = cur_up

    # btc_up_w1: janela 30–60s
    sub_w1 = [t for t in ticks if 30 <= t["t_sec"] < 60]
    if sub_w1:
        vu = sum(t["size_usdc"] for t in sub_w1 if t["outcome"] == "Up")
        tt = sum(t["size_usdc"] for t in sub_w1) + 1e-8
        feat["btc_up_w1"] = float(vu / tt)
    else:
        feat["btc_up_w1"] = 0.5
    feat["_w1_ticks"] = len(sub_w1)

    # btc_size_disparity
    up_tks = [t for t in ticks if t["outcome"] == "Up"]
    dn_tks = [t for t in ticks if t["outcome"] == "Down"]
    avg_up = float(sum(t["size_usdc"] for t in up_tks) / (len(up_tks) + 1e-8))
    avg_dn = float(sum(t["size_usdc"] for t in dn_tks) / (len(dn_tks) + 1e-8))
    feat["btc_size_disparity"] = float(avg_up - avg_dn)

    # btc_tw_up_ratio (exponential decay)
    t_arr  = np.array([t["t_sec"]     for t in ticks], dtype=np.float64)
    sz_arr = np.array([t["size_usdc"] for t in ticks], dtype=np.float64)
    up_arr = np.array([1.0 if t["outcome"] == "Up" else 0.0 for t in ticks])
    w_exp  = np.exp(-0.02 * (OBSERVE_SECS - t_arr))
    feat["btc_tw_up_ratio"] = float(
        np.sum(up_arr * sz_arr * w_exp) / (np.sum(sz_arr * w_exp) + 1e-9)
    )

    feat["_vol_up_usdc"] = float(vol_up)
    feat["_vol_dn_usdc"] = float(vol_dn)
    return feat

# ── Relatório ─────────────────────────────────────────────────────────────────
def print_report(feat: dict, slot_ts: int, t_into: int):
    now_str = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    slot_str = datetime.fromtimestamp(slot_ts, tz=timezone.utc).strftime("%H:%M:%S")

    print("\n" + "="*72)
    print(f"  VALIDAÇÃO DE FEATURES AO VIVO — {now_str}")
    print(f"  Slot: {slot_str}  |  t+{t_into}s dentro do slot")
    print("="*72)

    # WS health
    print("\n[WS HEALTH]")
    b_age = int(time.time() - _spot_stats["last"]) if _spot_stats["last"] else 999
    c_age = int(time.time() - _clob_stats["last"]) if _clob_stats["last"] else 999
    print(f"  Binance   msgs={_spot_stats['msgs']:>5}  última={b_age:>3}s  "
          f"candles_buf={len(_spot_candles)}  "
          f"{'✅' if b_age < 90 and _spot_stats['msgs'] > 0 else '❌'}")
    print(f"  CLOB WS   msgs={_clob_stats['msgs']:>6}  última={c_age:>3}s  "
          f"book_evs={_clob_stats['book_events']}  pc_evs={_clob_stats['pc_events']}  "
          f"{'✅' if c_age < 30 else '❌'}")

    # Spot details
    print(f"\n[SPOT DETAILS]  BTC={feat.get('_btc_price', 0):.2f}  "
          f"vol_1h={feat.get('btc_vol_1h', 0):.6f}  "
          f"inslot_candles={feat.get('_spot_inslot_candles', 0)}")
    for lbl in ["5m", "30m", "1h"]:
        n = feat.get(f"_spot_{lbl}_candles", 0)
        print(f"  {lbl} window: {n} velas no buffer")

    # CLOB WS details
    print(f"\n[CLOB WS DETAILS]  "
          f"book_snaps={feat.get('_clob_book_snaps', 0)}  "
          f"pc_evs={feat.get('_clob_pc_events', 0)}  "
          f"fallback={feat.get('_clob_fallback', False)}")

    # Data-API ticks
    print(f"\n[DATA-API TICKS]  n={feat.get('_n_ticks', 0)}  "
          f"vol_up=${feat.get('_vol_up_usdc', 0):.2f}  "
          f"vol_dn=${feat.get('_vol_dn_usdc', 0):.2f}  "
          f"w1_ticks={feat.get('_w1_ticks', 0)}")

    # OB REST details
    ob_ok = feat.get("ob_imbalance", None) is not None
    print(f"\n[OB REST]  "
          f"{'✅ OK' if ob_ok else '❌ FALHOU'}  "
          f"ask={feat.get('best_ask', '?')}  bid={feat.get('best_bid', '?')}  "
          f"levels={feat.get('n_ask_levels', '?')}/{feat.get('n_bid_levels', '?')}")

    # 20 features
    print(f"\n[20 FEATURES v29]\n  {'Feature':<30} {'Valor':>12}  {'Fonte':<25} Status")
    print(f"  {'-'*30} {'-'*12}  {'-'*25} ------")

    NEUTRAL_EXPECTED = {"lag_ur_zscore_20", "prev_slot_up_ratio_5", "btc_up_ratio_zscore_5s"}

    ok = warn = missing = 0
    for f in V29_FEATURES:
        val  = feat.get(f)
        src  = FEATURE_SOURCE.get(f, "?")
        if val is None:
            status = "❌ AUSENTE"
            val_str = "         N/A"
            missing += 1
        else:
            val_str = f"{val:>12.5f}"
            if f in NEUTRAL_EXPECTED:
                status = "⚠️  neutro (sem hist)"
                warn += 1
            elif f == "btc_dist_1k" and 0 < val <= 0.5:
                status = "✅"
                ok += 1
            elif f in ("btc_up_w1", "btc_tw_up_ratio", "btc_up_ratio") and abs(val - 0.5) < 0.001:
                status = "⚠️  neutro (sem ticks?)" if feat.get("_n_ticks", 0) == 0 else "⚠️  mercado 50/50"
                warn += 1
            elif f == "clob_spread_trend" and val == 0.0:
                snaps = feat.get("_clob_book_snaps", 0)
                status = f"⚠️  só {snaps} snap(s) no slot" if snaps < 2 else "✅ spread estável"
                warn += 1
            elif val == 0.0 and f not in ("btc_size_disparity",):
                status = "⚠️  ZERO"
                warn += 1
            else:
                status = "✅"
                ok += 1
        print(f"  {f:<30} {val_str}  {src:<25} {status}")

    print(f"\n[RESUMO]  ✅ OK={ok}  ⚠️ warn={warn}  ❌ ausente={missing}  total={len(V29_FEATURES)}")

    # Sanity checks
    print("\n[SANITY CHECKS]")
    checks = [
        ("Spot buffer ≥ 60 velas",            len(_spot_candles) >= 60),
        ("vol_1h > 0 (mercado ativo)",         feat.get("btc_vol_1h", 0) > 0),
        ("btc_inslot_ret normalizado ∈ [-10,10]", abs(feat.get("btc_inslot_ret", 0)) <= 10),
        ("btc_dist_1k ∈ (0, 0.5]",            0 < feat.get("btc_dist_1k", 0) <= 0.5),
        ("ob_imbalance presente e ∈ [0,1]",   0 <= feat.get("ob_imbalance", -1) <= 1),
        ("ob_depth_ratio presente e ∈ [0,1]", 0 <= feat.get("ob_depth_ratio", -1) <= 1),
        ("clob_spread_mean > 0",               feat.get("clob_spread_mean", 0) > 0),
        ("clob_ask_pressure ∈ [0,1]",         0 <= feat.get("clob_ask_pressure", -1) <= 1),
        ("Data-API retornou ticks",            feat.get("_n_ticks", 0) > 0),
        ("btc_tw_up_ratio ∈ [0,1]",           0 <= feat.get("btc_tw_up_ratio", -1) <= 1),
        ("CLOB WS recebeu book events",        _clob_stats["book_events"] > 0),
        ("CLOB WS recebeu price_change events", _clob_stats["pc_events"] > 0),
        ("pre_1h_ret normalizado ∈ [-50,50]", abs(feat.get("btc_pre_1h_ret", 0)) <= 50),
    ]
    all_pass = True
    for desc, passed in checks:
        icon = "✅" if passed else "❌"
        if not passed:
            all_pass = False
        print(f"  {icon}  {desc}")

    verdict = "🟢 FEED VÁLIDO — pode religar o bot" if all_pass else \
              "🔴 PROBLEMAS ENCONTRADOS — revisar antes de religar"
    print(f"\n  VEREDICTO: {verdict}")
    print("="*72 + "\n")

    return ok, warn, missing

# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    global _target_tokens, _slot_start

    print("\n" + "="*72)
    print("  VALIDAÇÃO COMPLETA DE FEED AO VIVO — v29 20 FEATURES")
    print("  Replica exatamente o fluxo do live_trader")
    print("="*72 + "\n")

    # 1. Mercado ativo
    log.info("Procurando mercado ativo...")
    slot_ts, yes_token, no_token = find_active_market()
    if not slot_ts:
        log.error("Nenhum mercado ativo. Saindo.")
        sys.exit(1)

    _slot_start    = slot_ts
    _target_tokens = [yes_token, no_token]

    # 2. Bootstrap spot
    bootstrap_spot(300)

    # 3. Iniciar WS
    log.info("Iniciando WebSockets...")
    ws_tasks = [
        asyncio.create_task(run_binance()),
        asyncio.create_task(run_clob()),
    ]
    await asyncio.sleep(3)   # aguardar subscriptions

    # 4. Calcular quanto tempo falta para t=ENTER_AT dentro do slot
    now = int(time.time())
    t_into = now - slot_ts
    wait_for = max(0, ENTER_AT - t_into)

    if t_into >= ENTER_AT + 60:
        log.warning("Slot já passou da janela de entrada (t=%ds). Calculando no momento atual.", t_into)
        wait_for = 0
    elif wait_for > 0:
        log.info("t=%ds no slot. Aguardando %ds até t=%ds (janela de entrada)...",
                 t_into, wait_for, ENTER_AT)
        # Log a cada 30s enquanto aguarda
        remaining = wait_for
        while remaining > 0:
            await asyncio.sleep(min(30, remaining))
            remaining -= 30
            t_now = int(time.time()) - slot_ts
            log.info("  t=%ds | Binance msgs=%d | CLOB msgs=%d | aguardando...",
                     t_now, _spot_stats["msgs"], _clob_stats["msgs"])
    else:
        log.info("t=%ds — já na janela de entrada. Computando agora.", t_into)

    t_compute = int(time.time())
    t_into_final = t_compute - slot_ts
    log.info("Computando features (t=%ds no slot)...", t_into_final)

    # 5. Coletar dados em paralelo
    feat = {}

    # 5a. Spot (já no buffer)
    log.info("  [1/4] Spot features...")
    spot = compute_spot_features(slot_ts)
    feat.update(spot)

    # 5b. OB REST com retry
    log.info("  [2/4] OB REST (com retry)...")
    ob = fetch_ob_rest(yes_token, retries=3, delay=1.0)
    if ob:
        feat.update(ob)
        log.info("  OB OK: ask=%.3f bid=%.3f imb=%.3f depth_ratio=%.3f",
                 ob["best_ask"], ob["best_bid"], ob["ob_imbalance"], ob["ob_depth_ratio"])
    else:
        log.error("  OB FALHOU após 3 tentativas")

    # 5c. CLOB WS features
    log.info("  [3/4] CLOB WS features...")
    clob = compute_clob_ws_features(yes_token, slot_ts)
    feat.update(clob)
    log.info("  CLOB: spread_mean=%.4f trend=%.4f mid_vol=%.4f ask_pres=%.3f",
             clob.get("clob_spread_mean", 0), clob.get("clob_spread_trend", 0),
             clob.get("clob_mid_volatility", 0), clob.get("clob_ask_pressure", 0.5))

    # 5d. Data-API ticks
    log.info("  [4/4] Data-API trades...")
    ticks = fetch_inslot_trades(yes_token, no_token, slot_ts)
    tick_feats = compute_tick_features(ticks)
    feat.update(tick_feats)

    # 5e. Cross features
    feat["x_imb_x_ur"]    = feat.get("ob_imbalance", 0.5) * feat.get("btc_up_ratio", 0.5)
    feat["x_depth_x_vol"] = feat.get("ob_depth_ratio", 1.0) * feat.get("btc_vol_1h", 0.0)

    # 5f. Lag features (neutros sem histórico — documentado)
    feat["lag_ur_zscore_20"]    = 0.0
    feat["prev_slot_up_ratio_5"] = 0.5
    feat["btc_up_ratio_zscore_5s"] = 0.0

    # 6. Relatório
    ok, warn, missing = print_report(feat, slot_ts, t_into_final)

    # 7. Cancelar WS
    for t in ws_tasks:
        t.cancel()

    sys.exit(0 if missing == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
