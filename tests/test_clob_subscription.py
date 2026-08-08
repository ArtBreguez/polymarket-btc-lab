"""Regression tests for the Polymarket CLOB subscription protocol.

Root cause found 2026-08-08: the CLOB market channel accepts EXACTLY ONE
{"type":"Market"} message per connection. A second one is answered with the
plain-text payload "INVALID OPERATION" and the server then stops delivering
book/price_change events while ping/pong keeps succeeding.

Measured from the prod machine (fly.io ams):
    1 subscribe message  -> 417 events/sec
    2 subscribe messages ->   6.6 events/sec  (feed dead)

Consequence: every clob_* feature was 0.0 in ~91% of live decisions, i.e.
20.65% of the champion's feature importance was served as zeros.

These tests pin the invariant so it cannot regress.
"""
from __future__ import annotations

import asyncio
import time
import json
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy"))

import live_trader  # noqa: E402


class FakeWS:
    """Records every payload sent on a connection."""

    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send(self, payload: str) -> None:
        self.sent.append(json.loads(payload))

    def market_messages(self) -> list[dict]:
        return [m for m in self.sent if m.get("type") == "Market"]


@pytest.fixture(autouse=True)
def _clean_state():
    """Reset the module-level subscription state around each test."""
    live_trader._clob_subscribed.clear()
    live_trader._token_slot.clear()
    while True:
        try:
            live_trader._subscribe_queue.get_nowait()
        except Exception:
            break
    yield
    live_trader._clob_subscribed.clear()
    live_trader._token_slot.clear()


def _run(coro):
    return asyncio.run(coro)


class TestOneMarketMessagePerConnection:
    def test_on_connect_sends_exactly_one_market_message(self):
        """Existing + queued tokens must go out in a single Market message."""
        # Already-subscribed tokens carry a current slot_ts, as clob_subscribe()
        # always sets — otherwise _clob_prune_stale() legitimately drops them.
        cur = (int(time.time()) // live_trader.SLOT_DURATION) * live_trader.SLOT_DURATION
        live_trader._clob_subscribed.update({"tok_a", "tok_b"})
        live_trader._token_slot.update({"tok_a": cur, "tok_b": cur})
        live_trader.clob_subscribe(["tok_c", "tok_d"], slot_ts=cur)

        ws = FakeWS()
        with mock.patch.object(live_trader, "_http") as http:
            http.get.side_effect = AssertionError("REST hydration must not block on_connect")
            _run(live_trader._clob_on_connect(ws))

        msgs = ws.market_messages()
        assert len(msgs) == 1, f"expected exactly 1 Market message, got {len(msgs)}: {msgs}"
        assert set(msgs[0]["assets_ids"]) == {"tok_a", "tok_b", "tok_c", "tok_d"}

    def test_on_connect_with_no_tokens_sends_nothing(self):
        ws = FakeWS()
        _run(live_trader._clob_on_connect(ws))
        assert ws.market_messages() == []

    def test_queued_tokens_are_registered_as_subscribed(self):
        cur = (int(time.time()) // live_trader.SLOT_DURATION) * live_trader.SLOT_DURATION
        live_trader.clob_subscribe(["tok_x"], slot_ts=cur)
        ws = FakeWS()
        with mock.patch.object(live_trader, "_http"):
            _run(live_trader._clob_on_connect(ws))
        assert "tok_x" in live_trader._clob_subscribed
        assert live_trader._token_slot["tok_x"] == cur


class TestNoIncrementalSubscribe:
    """New tokens must trigger a reconnect, never a second Market message."""

    def test_drain_forces_reconnect_and_does_not_send(self):
        mgr = mock.MagicMock()
        live_trader.clob_subscribe(["tok_new"], slot_ts=1786225500)

        live_trader._clob_drain_and_subscribe(mgr)

        mgr.send_sync.assert_not_called()
        mgr.force_reconnect.assert_called_once()
        assert "tok_new" in live_trader._clob_subscribed

    def test_drain_is_a_noop_for_already_subscribed_tokens(self):
        """No reconnect storm when the entry window re-requests known tokens."""
        live_trader._clob_subscribed.add("tok_known")
        mgr = mock.MagicMock()
        live_trader.clob_subscribe(["tok_known"], slot_ts=1786225500)

        live_trader._clob_drain_and_subscribe(mgr)

        mgr.force_reconnect.assert_not_called()
        mgr.send_sync.assert_not_called()

    def test_empty_queue_does_nothing(self):
        mgr = mock.MagicMock()
        live_trader._clob_drain_and_subscribe(mgr)
        mgr.force_reconnect.assert_not_called()
        mgr.send_sync.assert_not_called()

    def test_on_message_never_sends_a_market_message(self):
        """The message handler drained the queue and used to send inline."""
        mgr = mock.MagicMock()
        live_trader.clob_subscribe(["tok_inline"], slot_ts=1786225500)

        with mock.patch.object(live_trader, "_clob_ws_manager", mgr):
            _run(live_trader._clob_on_message([{"event_type": "unknown"}]))

        mgr.send.assert_not_called()
        mgr.force_reconnect.assert_called_once()
