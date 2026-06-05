#!/bin/bash
# BTC Bot Monitor — checks Fly.io logs for errors, trades, and health
# Location in repo: scripts/monitor_bot.sh
# Called by Hermes cron (btc-bot-monitor) every 10 minutes
#
# Usage: bash /home/ubuntu/polymarket-btc-lab/scripts/monitor_bot.sh
#
# Output: structured health report suitable for LLM analysis
# The cron agent reads this output and decides whether to alert the user

FLYCTL="/home/ubuntu/.fly/bin/flyctl"
APP="polymarket-maker-mm"

LOGS=$($FLYCTL logs -a "$APP" --no-tail 2>&1 | tail -200)

if [ -z "$LOGS" ]; then
    echo "WARNING: No logs retrieved — bot may be down"
    exit 0
fi

LAST_ENTRY=$(echo "$LOGS" | grep "Entry window" | tail -1)
LAST_PREDICTION=$(echo "$LOGS" | grep "Prediction:" | tail -1)
LAST_TIME=$(echo "$LAST_ENTRY" | grep -oP '\d{2}:\d{2}:\d{2}' | head -1)

ERRORS=$(echo "$LOGS" | grep -c "ERROR")
WARNINGS=$(echo "$LOGS" | grep -c "WARNING")
TRADES=$(echo "$LOGS" | grep -c "Order placed")
FILLS=$(echo "$LOGS" | grep -c "FILL\|SETTLED\|filled")
SKIPS_CONF=$(echo "$LOGS" | grep -c "Skip — conf")
SKIPS_BALANCE=$(echo "$LOGS" | grep -c "Insufficient balance")
SKIPS_DIVERGE=$(echo "$LOGS" | grep -c "diverges.*mid")
SKIPS_EDGE=$(echo "$LOGS" | grep -c "Skip — edge")
SKIPS_RANGE=$(echo "$LOGS" | grep -c "outside.*valid range\|outside.*0.38")
SANITY_FAIL=$(echo "$LOGS" | grep -c "SANITY")
FEATURE_FAIL=$(echo "$LOGS" | grep -c "FEATURE")
WS_ERRORS=$(echo "$LOGS" | grep -c "CLOB WS error")

BOT_STATUS=$($FLYCTL status -a "$APP" 2>&1 | grep -oP 'started|stopped' | head -1)
BALANCE_LINE=$(echo "$LOGS" | grep "Insufficient balance" | tail -1)
BALANCE=$(echo "$BALANCE_LINE" | grep -oP '\$[\d.]+' | head -1)

echo "=== BTC Bot Monitor ==="
echo "Status: ${BOT_STATUS:-unknown}"
echo "Last entry: $LAST_TIME UTC"
[ -n "$LAST_PREDICTION" ] && echo "Last prediction: $(echo "$LAST_PREDICTION" | grep -oP 'Prediction: \w+\s+conf=[\d.]+%')"
echo ""
echo "Predictions: $(echo "$LOGS" | grep -c 'Prediction:') | Trades: $TRADES | Fills: $FILLS"
echo "Skips — conf: $SKIPS_CONF | balance: $SKIPS_BALANCE | diverge: $SKIPS_DIVERGE | edge: $SKIPS_EDGE | range: $SKIPS_RANGE"
echo "Errors: $ERRORS | Warnings: $WARNINGS | Sanity: $SANITY_FAIL | Feature: $FEATURE_FAIL | WS drops: $WS_ERRORS"
[ -n "$BALANCE" ] && echo "Balance: $BALANCE"

# Recent errors
RECENT_ERRORS=$(echo "$LOGS" | grep "ERROR" | tail -3 | sort -u)
[ -n "$RECENT_ERRORS" ] && echo -e "\nRecent errors:\n$RECENT_ERRORS"

# Recent trades
RECENT_TRADES=$(echo "$LOGS" | grep -E "Order placed|FILL|SETTLED" | tail -5)
[ -n "$RECENT_TRADES" ] && echo -e "\nRecent trades:\n$RECENT_TRADES"
