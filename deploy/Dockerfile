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

# Model is downloaded from HuggingFace at startup via HF_TOKEN secret.
# No pkl bundled in image — always pulls champion.pkl from artbreguez/polymarket-btc-model.

CMD ["python", "-u", "live_trader.py"]
