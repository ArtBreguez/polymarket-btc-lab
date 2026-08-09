FROM python:3.12-slim
WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY live_trader.py .
COPY ws_manager.py .
COPY data_quality_gate.py .
COPY clob_features.py .
COPY clob_feature_logger.py .

# Model is downloaded from HuggingFace at startup via the HF_TOKEN secret.
# champion_fallback.pkl is a last-resort copy used ONLY when HF is unreachable
# and /tmp has no cached model — otherwise an HF outage bricks the bot on boot
# (hit for real on 2026-08-08: private-storage quota => every HF read 403'd).
# Keep it in sync when promoting a new champion.
COPY champion_fallback.pkl .

CMD ["python", "-u", "live_trader.py"]
