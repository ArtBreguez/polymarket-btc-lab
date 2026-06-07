"""CLOB Feature Logger — records real-time CLOB features with ground truth for future training.

Logs one row per slot with:
  - slot_ts, timestamp
  - 10 CLOB real-time features (from ClobFeatureAccumulator)
  - direction_resolved (UP/DOWN) — filled at settlement time
  - target (1.0 = UP won, 0.0 = DOWN won) — filled at settlement time

Data is stored as JSONL in /tmp/clob_features_log.jsonl and periodically
uploaded to HuggingFace for training v25+.

Usage:
    from clob_feature_logger import log_clob_features, resolve_clob_features
    log_clob_features(slot_ts, token_id, features_dict)
    resolve_clob_features(slot_ts, target)  # called at settlement
"""

import json
import logging
import os
import threading
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

LOG_FILE = Path("/tmp/clob_features_log.jsonl")
_lock = threading.Lock()


def log_clob_features(
    slot_ts: int,
    token_id: str,
    clob_features: dict,
    model_prob: float = 0.5,
    ask_price: float = 0.5,
    t_in_slot: int = 0,
) -> None:
    """Record CLOB features for a slot at prediction time.
    
    Args:
        slot_ts: Slot timestamp
        token_id: YES token ID
        clob_features: Dict of clob_* features from accumulator
        model_prob: Model's predicted probability of UP
        ask_price: Best ask at prediction time
        t_in_slot: Seconds elapsed in slot at prediction time
    """
    import time
    row = {
        "slot_ts": slot_ts,
        "token_id": token_id,
        "logged_at": time.time(),
        "t_in_slot": t_in_slot,
        "model_prob": round(model_prob, 4),
        "ask_price": round(ask_price, 4),
        "target": None,  # filled at settlement
        **{k: round(v, 6) for k, v in clob_features.items()},
    }
    with _lock:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(row) + "\n")
    log.debug("CLOB features logged for slot=%d (%d features)", slot_ts, len(clob_features))


def resolve_clob_features(slot_ts: int, target: float) -> None:
    """Backfill target for a resolved slot.
    
    Reads the log, finds matching slot_ts, updates target, rewrites.
    Called at settlement time.
    """
    if not LOG_FILE.exists():
        return
    
    with _lock:
        lines = LOG_FILE.read_text().strip().split("\n")
        updated = False
        new_lines = []
        for line in lines:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if row.get("slot_ts") == slot_ts and row.get("target") is None:
                    row["target"] = target
                    updated = True
                new_lines.append(json.dumps(row))
            except json.JSONDecodeError:
                new_lines.append(line)
        
        if updated:
            LOG_FILE.write_text("\n".join(new_lines) + "\n")
            log.info("CLOB features resolved for slot=%d target=%.1f", slot_ts, target)


def get_log_stats() -> dict:
    """Return stats about the collected data."""
    if not LOG_FILE.exists():
        return {"total": 0, "resolved": 0, "pending": 0}
    
    total = 0
    resolved = 0
    with open(LOG_FILE) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                total += 1
                if row.get("target") is not None:
                    resolved += 1
            except json.JSONDecodeError:
                pass
    return {"total": total, "resolved": resolved, "pending": total - resolved}
