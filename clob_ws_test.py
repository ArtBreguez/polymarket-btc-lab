"""
Diagnostic: Connect to Polymarket CLOB WS and log raw messages for 60s.
Goal: Confirm the exact JSON keys in 'price_change' events.
"""
import asyncio
import json
import time
import requests
import websockets

CLOB_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
GAMMA_URL   = "https://gamma-api.polymarket.com"

def get_btc_tokens():
    """Get the current active BTC 5-min market token IDs."""
    resp = requests.get(
        f"{GAMMA_URL}/markets",
        params={"tag": "BTC", "active": "true", "closed": "false", "limit": 20},
        timeout=10,
    )
    markets = resp.json()
    btc_5min = []
    for m in markets:
        question = m.get("question", "").lower()
        if "bitcoin" in question or "btc" in question:
            tokens = m.get("clobTokenIds", [])
            if tokens:
                btc_5min.append({
                    "question": m.get("question", ""),
                    "tokens": tokens,
                    "endDate": m.get("endDateIso", ""),
                })
    # sort by endDate ascending to get the soonest-expiring (most active)
    btc_5min.sort(key=lambda x: x["endDate"])
    print(f"Found {len(btc_5min)} BTC markets")
    for m in btc_5min[:3]:
        print(f"  Q: {m['question'][:80]}")
        print(f"     tokens: {m['tokens'][:2]}")
        print(f"     end: {m['endDate']}")
    return btc_5min[:1][0]["tokens"][:2] if btc_5min else []

async def test_clob_ws(token_ids):
    print(f"\nConnecting to {CLOB_WS_URL}")
    print(f"Subscribing to tokens: {token_ids}\n")

    msg_count = 0
    book_count = 0
    price_change_count = 0
    price_change_keys = set()
    price_change_sample = None

    async with websockets.connect(CLOB_WS_URL, ping_interval=None) as ws:
        sub_msg = {
            "auth": {},
            "markets": [],
            "assets_ids": token_ids,
            "type": "market"
        }
        await ws.send(json.dumps(sub_msg))
        print("Subscribe sent. Listening for 60s...\n")

        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=5)
                data = json.loads(raw)
                events = [data] if isinstance(data, dict) else data

                for ev in events:
                    msg_count += 1
                    etype = ev.get("event_type", "unknown")

                    if etype == "book":
                        book_count += 1
                        if book_count == 1:
                            print("=== BOOK EVENT (first) ===")
                            print(f"  Top-level keys: {list(ev.keys())}")
                            asks = ev.get("asks", [])[:2]
                            bids = ev.get("bids", [])[:2]
                            print(f"  asks[:2]: {asks}")
                            print(f"  bids[:2]: {bids}")
                            print()

                    elif etype == "price_change":
                        price_change_count += 1
                        # Record all keys seen
                        price_change_keys.update(ev.keys())
                        if price_change_count <= 3:
                            print(f"=== PRICE_CHANGE EVENT #{price_change_count} ===")
                            print(f"  Full event: {json.dumps(ev, indent=2)}")
                            print()
                        elif price_change_sample is None:
                            price_change_sample = ev

            except asyncio.TimeoutError:
                pass
            except Exception as e:
                print(f"Error: {e}")
                break

    print("\n=== SUMMARY ===")
    print(f"Total msgs: {msg_count}")
    print(f"  book events: {book_count}")
    print(f"  price_change events: {price_change_count}")
    print(f"  price_change top-level keys seen: {price_change_keys}")

    if price_change_count > 0:
        print("\n>>> KEY FINDING <<<")
        # Check which key holds the changes
        if "price_changes" in price_change_keys:
            print("  Uses 'price_changes' (plural) — code is CORRECT")
        elif "changes" in price_change_keys:
            print("  Uses 'changes' — code is WRONG (should be 'changes', not 'price_changes')")
        else:
            print(f"  Unknown key structure: {price_change_keys}")

if __name__ == "__main__":
    tokens = get_btc_tokens()
    if not tokens:
        print("ERROR: No BTC tokens found")
    else:
        asyncio.run(test_clob_ws(tokens))
