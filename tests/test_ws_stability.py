"""
Test: Polymarket CLOB WebSocket connection stability
=====================================================
Compares two configurations:
  A) ping_interval=20, ping_timeout=10  (our current config — client pings)
  B) ping_interval=None, ping_timeout=None  (server controls ping/pong)

Hypothesis: Config A causes "double-ping" conflict with the Polymarket server,
leading to code 1006 disconnects every ~60s. Config B should be stable.

Reference: https://github.com/Polymarket/py-clob-client/issues/82
"""

import asyncio
import json
import time
import sys

WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

# Use a known active BTC 5-min market token — just needs any valid asset_id
# We'll subscribe to a dummy one; the server sends book snapshots regardless
TEST_ASSET_ID = "21742633143463906290209502535071711472481901910614024017207968569310580469319"


async def test_connection(name: str, ping_interval, ping_timeout, duration: int = 180):
    """Test a WS connection configuration for `duration` seconds."""
    import websockets

    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"  ping_interval={ping_interval}, ping_timeout={ping_timeout}")
    print(f"  duration={duration}s")
    print(f"{'='*60}")

    connects = 0
    disconnects = 0
    total_msgs = 0
    start = time.time()

    while time.time() - start < duration:
        try:
            async with websockets.connect(
                WS_URL,
                ping_interval=ping_interval,
                ping_timeout=ping_timeout,
                close_timeout=5,
            ) as ws:
                connects += 1
                connect_at = time.time()
                elapsed_total = connect_at - start
                print(f"  [{elapsed_total:6.1f}s] Connected (#{connects})")

                # Subscribe
                await ws.send(json.dumps({
                    "type": "Market",
                    "assets_ids": [TEST_ASSET_ID],
                }))

                msg_count = 0
                try:
                    async for raw in ws:
                        msg_count += 1
                        total_msgs += 1

                        # Log every 500 messages
                        if msg_count % 500 == 0:
                            uptime = time.time() - connect_at
                            elapsed_total = time.time() - start
                            print(f"  [{elapsed_total:6.1f}s] ... {msg_count} msgs, uptime={uptime:.0f}s")

                        if time.time() - start >= duration:
                            print(f"  [{time.time()-start:6.1f}s] Duration reached, closing")
                            await ws.close()
                            break
                except Exception:
                    pass

                uptime = time.time() - connect_at
                elapsed_total = time.time() - start
                print(f"  [{elapsed_total:6.1f}s] Connection ended — uptime={uptime:.1f}s, msgs={msg_count}")

        except Exception as e:
            disconnects += 1
            elapsed_total = time.time() - start
            code = getattr(e, 'code', '?')
            print(f"  [{elapsed_total:6.1f}s] DISCONNECT #{disconnects}: code={code} {type(e).__name__}: {e}")
            await asyncio.sleep(2)

    print(f"\n  RESULT: {name}")
    print(f"    Total connects:    {connects}")
    print(f"    Total disconnects: {disconnects}")
    print(f"    Total messages:    {total_msgs}")
    print(f"    Avg uptime:        {duration / max(connects, 1):.1f}s per connection")
    print(f"    Stability:         {'GOOD' if disconnects <= 1 else 'BAD'} ({disconnects} drops in {duration}s)")

    return {
        "name": name,
        "connects": connects,
        "disconnects": disconnects,
        "total_msgs": total_msgs,
        "avg_uptime": duration / max(connects, 1),
    }


async def main():
    duration = 120  # 2 minutes per test

    # Test A: Current config (client pings — expected to fail)
    result_a = await test_connection(
        name="A: Client pings (current)",
        ping_interval=20,
        ping_timeout=10,
        duration=duration,
    )

    # Brief pause between tests
    print("\n\n--- Pause 5s between tests ---\n")
    await asyncio.sleep(5)

    # Test B: Server controls ping/pong (expected fix)
    result_b = await test_connection(
        name="B: Server pings (fix)",
        ping_interval=None,
        ping_timeout=None,
        duration=duration,
    )

    # Summary
    print(f"\n\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    for r in [result_a, result_b]:
        print(f"  {r['name']:30s} | connects={r['connects']:2d} disconnects={r['disconnects']:2d} avg_uptime={r['avg_uptime']:6.1f}s msgs={r['total_msgs']}")
    print(f"{'='*60}")

    winner = "B (server pings)" if result_b["disconnects"] < result_a["disconnects"] else "A (client pings)" if result_a["disconnects"] < result_b["disconnects"] else "TIE"
    print(f"  WINNER: {winner}")


if __name__ == "__main__":
    asyncio.run(main())
