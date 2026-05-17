FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NLTK_DATA=/usr/local/share/nltk_data \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMER_MODEL=unitary/toxic-bert \
    MODEL_PATH=/app/models/LogisticRegression.pkl \
    DATASET_PATH=/app/data/cleaned_dataset.csv \
    LOG_DB_PATH=/app/runtime/logs.db \
    MAX_INPUT_CHARS=2000 \
    S3_SYNC_ON_STARTUP=false

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

RUN python -m nltk.downloader -d "$NLTK_DATA" wordnet omw-1.4

COPY . .

RUN mkdir -p /app/runtime \
    && python - <<'PY'
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = os.environ["TRANSFORMER_MODEL"]
AutoTokenizer.from_pretrained(model_name)
AutoModelForSequenceClassification.from_pretrained(model_name)
PY

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:5000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5000"]
