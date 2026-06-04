#!/bin/bash
# BTC Bot Monitor — checks Fly.io logs for errors, trades, and health
# Called by Hermes cron every 10 minutes

FLYCTL="/home/ubuntu/.fly/bin/flyctl"
APP="polymarket-maker-mm"

# Grab last ~200 lines of logs
LOGS=$($FLYCTL logs -a "$APP" --no-tail 2>&1 | tail -200)

if [ -z "$LOGS" ]; then
    echo "WARNING: No logs retrieved from Fly.io — bot may be down"
    exit 0
fi

# Extract key metrics
LAST_ENTRY=$(echo "$LOGS" | grep "Entry window" | tail -1)
LAST_PREDICTION=$(echo "$LOGS" | grep "Prediction:" | tail -1)
LAST_TIME=$(echo "$LAST_ENTRY" | grep -oP '\d{2}:\d{2}:\d{2}' | head -1)

# Count events
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
RECONNECTS=$(echo "$LOGS" | grep -c "WS daemon connected")

# Get balance info
BALANCE_LINE=$(echo "$LOGS" | grep "Insufficient balance" | tail -1)
BALANCE=$(echo "$BALANCE_LINE" | grep -oP '\$[\d.]+' | head -1)

# Bot status
BOT_STATUS=$($FLYCTL status -a "$APP" 2>&1 | grep -E "STATE|started|stopped")

# Build report
echo "=== BTC Bot Monitor ==="
echo "Bot: $APP"
echo "Status: $(echo "$BOT_STATUS" | grep -oP 'started|stopped' | head -1)"
echo "Last entry: $LAST_TIME UTC"
echo ""

# Last prediction details
if [ -n "$LAST_PREDICTION" ]; then
    echo "Last prediction: $(echo "$LAST_PREDICTION" | grep -oP 'Prediction: \w+\s+conf=[\d.]+%')"
fi

echo ""
echo "--- Slot Activity ---"
echo "Predictions made: $(echo "$LOGS" | grep -c 'Prediction:')"
echo "Trades executed: $TRADES"
echo "Fills/Settlements: $FILLS"
echo "Skips (low conf): $SKIPS_CONF"
echo "Skips (no balance): $SKIPS_BALANCE"
echo "Skips (ask diverge): $SKIPS_DIVERGE"
echo "Skips (no edge): $SKIPS_EDGE"
echo "Skips (out of range): $SKIPS_RANGE"
echo ""
echo "--- Health ---"
echo "Errors: $ERRORS"
echo "Warnings: $WARNINGS"
echo "Sanity violations: $SANITY_FAIL"
echo "Feature violations: $FEATURE_FAIL"
echo "WS disconnects: $WS_ERRORS"
echo "WS reconnects: $RECONNECTS"

if [ -n "$BALANCE" ]; then
    echo ""
    echo "Last known balance: $BALANCE"
fi

# Alert conditions
ALERT=""
if echo "$BOT_STATUS" | grep -q "stopped"; then
    ALERT="$ALERT\nALERT: Bot is STOPPED!"
fi
if [ "$SANITY_FAIL" -gt 0 ]; then
    ALERT="$ALERT\nALERT: $SANITY_FAIL sanity violations detected!"
fi
if [ "$FEATURE_FAIL" -gt 0 ]; then
    ALERT="$ALERT\nALERT: $FEATURE_FAIL feature violations detected!"
fi
if [ "$SKIPS_BALANCE" -gt 5 ]; then
    ALERT="$ALERT\nALERT: $SKIPS_BALANCE balance skips — deposit may not have landed yet"
fi

# Show recent errors (deduped)
RECENT_ERRORS=$(echo "$LOGS" | grep "ERROR" | tail -5 | sort -u)
if [ -n "$RECENT_ERRORS" ]; then
    echo ""
    echo "--- Recent Errors ---"
    echo "$RECENT_ERRORS"
fi

# Show any trades
RECENT_TRADES=$(echo "$LOGS" | grep -E "Order placed|FILL|SETTLED" | tail -5)
if [ -n "$RECENT_TRADES" ]; then
    echo ""
    echo "--- Recent Trades ---"
    echo "$RECENT_TRADES"
fi

if [ -n "$ALERT" ]; then
    echo ""
    echo -e "$ALERT"
fi
