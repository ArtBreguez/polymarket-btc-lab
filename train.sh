#!/usr/bin/env bash
# train.sh — Run the full BTC model training pipeline locally.
#
# Usage:
#   ./train.sh v8               # train v8, auto-promote if beats champion
#   ./train.sh v9               # train v9
#   ./train.sh v8 --dry-run     # train but skip promotion and model card upload
#   ./train.sh v8 --notes "my notes"
#
# Requirements:
#   - HF_TOKEN in env (or set below)
#   - Modal authenticated: modal token set --token-id ... --token-secret ...
#   - uv / python3 available

set -euo pipefail

VERSION="${1:-v8}"
DRY_RUN=0
NOTES=""

# Parse extra flags
shift || true
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)    DRY_RUN=1 ;;
    --notes)      NOTES="$2"; shift ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
  shift
done

SCRIPT="scripts/train_${VERSION}_modal.py"

if [[ ! -f "$SCRIPT" ]]; then
  echo "ERROR: training script not found: $SCRIPT"
  echo "Available versions:"
  ls scripts/train_v*_modal.py 2>/dev/null | sed 's/scripts\/train_/  /' | sed 's/_modal.py//'
  exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════"
echo "  BTC ML Pipeline — Training $VERSION"
echo "  Script : $SCRIPT"
echo "  Dry run: $DRY_RUN"
[[ -n "$NOTES" ]] && echo "  Notes  : $NOTES"
echo "════════════════════════════════════════════════════════"
echo ""

# ── 1. Train on Modal ────────────────────────────────────────────────────────
echo "[1/3] Submitting training job to Modal..."
TRAIN_OUTPUT=$(modal run "$SCRIPT" 2>&1 | tee /dev/stderr)
echo ""

# Check if a new champion was promoted (parse modal output)
if echo "$TRAIN_OUTPUT" | grep -q "Promoted.*YES\|PROMOTED\|promoted.*true"; then
  PROMOTED=1
  echo "✅ New champion promoted!"
else
  PROMOTED=0
  echo "ℹ️  No promotion — current champion retained."
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo ""
  echo "DRY RUN — skipping model card update and deploy trigger."
  exit 0
fi

if [[ "$PROMOTED" -eq 0 ]]; then
  echo "Nothing to update. Done."
  exit 0
fi

# ── 2. Update HF model card ──────────────────────────────────────────────────
echo ""
echo "[2/3] Updating HuggingFace model card..."
python3 scripts/update_model_card.py --hf-token "${HF_TOKEN:-}"
echo "✅ Model card updated → https://huggingface.co/artbreguez/polymarket-btc-model"

# ── 3. Trigger Fly.io deploy via GitHub Actions ──────────────────────────────
echo ""
echo "[3/3] Triggering deploy workflow on GitHub..."
if command -v gh &>/dev/null; then
  gh workflow run deploy.yml \
    --repo ArtBreguez/polymarket-btc-lab \
    --ref main \
    --field reason="Local train.sh promoted $VERSION"
  echo "✅ Deploy workflow triggered — check: https://github.com/ArtBreguez/polymarket-btc-lab/actions"
else
  echo "⚠️  gh CLI not found — trigger deploy manually:"
  echo "    gh workflow run deploy.yml --repo ArtBreguez/polymarket-btc-lab --ref main"
fi

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Pipeline complete!"
echo "  Champion: https://huggingface.co/artbreguez/polymarket-btc-model"
echo "  Deploy:   https://github.com/ArtBreguez/polymarket-btc-lab/actions"
echo "════════════════════════════════════════════════════════"
