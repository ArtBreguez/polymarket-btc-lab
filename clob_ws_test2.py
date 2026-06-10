"""
Diagnostic v2: Connects exactly like ws_manager.py does.
Tests both URL variants and subscribe formats.
"""
import asyncio
import json
import time
import requests
import websockets

CLOB_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

def get_btc_token():
    """Get any BTC Polymarket token that's active."""
    try:
        # Try direct CLOB markets endpoint
        resp = requests.get(
            "https://clob.polymarket.com/markets",
            params={"next_cursor": ""},
            timeout=10,
        )
        data = resp.json()
        markets = data.get("data", [])
        print(f"CLOB markets count: {len(markets)}")
        for m in markets[:5]:
            print(f"  condition: {m.get('condition_id','')[:16]}... tokens: {m.get('tokens','')}")
    except Exception as e:
        print(f"CLOB markets error: {e}")

    # Try gamma API for BTC active
    try:
        resp = requests.get(
            "https://gamma-api.polymarket.com/markets",
            params={"active": "true", "closed": "false", "limit": 50, "tag_slug": "crypto"},
            timeout=10,
        )
        markets = resp.json()
        print(f"\nGamma markets: {len(markets)}")
        btc = [m for m in markets if "bitcoin" in m.get("question","").lower() or "btc" in m.get("question","").lower()]
        print(f"BTC markets: {len(btc)}")
        for m in btc[:5]:
            print(f"  Q: {m.get('question','')[:80]}")
            tokens = m.get("clobTokenIds", [])
            print(f"  tokens: {tokens[:2]}")
        if btc:
            return btc[0].get("clobTokenIds", [])[:2]
    except Exception as e:
        print(f"Gamma error: {e}")

    return []

async def test_ws(token_ids):
    print(f"\n=== Testing connection to {CLOB_WS_URL} ===")
    print(f"Token IDs: {token_ids[:2]}\n")

    try:
        async with websockets.connect(
            CLOB_WS_URL,
            ping_interval=None,
            ping_timeout=None,
            open_timeout=10,
        ) as ws:
            print("Connected!")
            # Use the EXACT same format as live_trader.py
            sub_msg = {"type": "Market", "assets_ids": token_ids}
            await ws.send(json.dumps(sub_msg))
            print(f"Sent subscribe: {sub_msg}\n")

            deadline = time.time() + 30
            msg_count = 0
            all_keys = {}

            while time.time() < deadline:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=5)
                    data = json.loads(raw)
                    events = [data] if isinstance(data, dict) else data

                    for ev in (events if isinstance(events, list) else [events]):
                        msg_count += 1
                        etype = ev.get("event_type", "unknown")

                        # Record keys per event type
                        if etype not in all_keys:
                            all_keys[etype] = set(ev.keys())
                        else:
                            all_keys[etype].update(ev.keys())

                        if msg_count <= 5:
                            print(f"MSG #{msg_count} type={etype}")
                            print(f"  keys: {list(ev.keys())}")
                            if etype == "book":
                                print(f"  asks[:2]: {ev.get('asks',[])[:2]}")
                                print(f"  bids[:2]: {ev.get('bids',[])[:2]}")
                            elif etype == "price_change":
                                # Check BOTH possible field names
                                pc_v1 = ev.get("price_changes", [])
                                pc_v2 = ev.get("changes", [])
                                print(f"  'price_changes' field: {pc_v1[:2]} (len={len(pc_v1)})")
                                print(f"  'changes' field:       {pc_v2[:2]} (len={len(pc_v2)})")
                                print(f"  FULL event: {json.dumps(ev)[:500]}")
                            print()

                except asyncio.TimeoutError:
                    print(f"  (timeout, {msg_count} msgs so far)")

            print(f"\n=== RESULTS ===")
            print(f"Total events received: {msg_count}")
            for etype, keys in all_keys.items():
                print(f"  {etype}: keys={keys}")

    except Exception as e:
        print(f"Connection error: {type(e).__name__}: {e}")
        # Try alternate URL
        print("\nTrying alternate URL: wss://ws-subscriptions-clob.polymarket.com/ws/")
        try:
            async with websockets.connect(
                "wss://ws-subscriptions-clob.polymarket.com/ws/",
                ping_interval=None,
                open_timeout=10,
            ) as ws:
                print("Alternate connected!")
                sub_msg = {
                    "auth": {},
                    "markets": [],
                    "assets_ids": token_ids,
                    "type": "market"
                }
                await ws.send(json.dumps(sub_msg))
                print(f"Sent: {sub_msg}\n")

                deadline = time.time() + 20
                msg_count = 0
                while time.time() < deadline:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=5)
                        data = json.loads(raw)
                        events = [data] if isinstance(data, dict) else data
                        for ev in (events if isinstance(events, list) else [events]):
                            msg_count += 1
                            etype = ev.get("event_type", "unknown")
                            print(f"MSG #{msg_count} type={etype} keys={list(ev.keys())}")
                            if etype == "price_change":
                                print(f"  'changes': {ev.get('changes',[])[:2]}")
                                print(f"  'price_changes': {ev.get('price_changes',[])[:2]}")
                    except asyncio.TimeoutError:
                        pass
        except Exception as e2:
            print(f"Alternate also failed: {e2}")

if __name__ == "__main__":
    tokens = get_btc_token()
    if not tokens:
        print("No tokens found — using hardcoded test token")
        # Use any known polymarket token
        tokens = ["21742633143463906290569050155826241533067272736897614950488156847949938836455"]
    asyncio.run(test_ws(tokens))
