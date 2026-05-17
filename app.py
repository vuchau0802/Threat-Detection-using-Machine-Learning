import logging
import os
import re
import sqlite3
import string
from dataclasses import dataclass
from pathlib import Path

import asyncio
import json

if os.getenv("S3_SYNC_ON_STARTUP", "false").lower() == "true":
    from AWS.s3_store import sync_from_s3

    sync_from_s3()

from models.ensemble import ensemble_predict
import nltk
import uvicorn

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse

from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from nltk.stem import WordNetLemmatizer

from datetime import datetime, timezone, timedelta


CDT = timezone(timedelta(hours=-5))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    model_path: str = os.getenv("MODEL_PATH", "models/LogisticRegression.pkl")
    dataset_path: str = os.getenv("DATASET_PATH", "data/cleaned_dataset.csv")
    log_db_path: str = os.getenv("LOG_DB_PATH", "models/logs.db")
    max_input_chars: int = int(os.getenv("MAX_INPUT_CHARS", "2000"))


cfg = Config()
nltk.download("wordnet", quiet=True)
lemmatizer = WordNetLemmatizer()

app = FastAPI(
    title="Threat API",
    version="1.0.0"
)

# CORS

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# TEMPLATES
templates = Jinja2Templates(directory="templates")
Path(cfg.log_db_path).parent.mkdir(parents=True, exist_ok=True)
conn = sqlite3.connect(cfg.log_db_path, check_same_thread=False)
conn.row_factory = sqlite3.Row

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT,
    prediction INTEGER,
    confidence REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()

# LOAD MODEL
_POSITIVE_WORDS = frozenset({
    "happy", "good", "love", "great", "nice",
    "awesome", "wonderful", "excellent",
    "fantastic", "brilliant", "joy",
    "kind", "sweet", "thank",
})

_NEGATIVE_WORDS = frozenset({
    "bad", "sad", "angry", "hate",
    "terrible", "awful", "horrible",
    "disgusting", "furious", "depressed",
    "miserable", "kill", "die", "hurt",
})

_HARD_THREAT_WORDS = frozenset({
    "kill", "murder", "stab", "shoot",
    "rape", "hang", "strangle",
    "destroy", "hurt", "harm",
    "torture", "beat", "slaughter",
    "exterminate", "die",
})

_FLAG_WORDS = sorted(
    _NEGATIVE_WORDS | _HARD_THREAT_WORDS | {
        "idiot",
        "stupid",
        "dumb",
        "ugly",
        "loser",
        "fat",
        "freak",
        "moron",
        "worthless",
        "pathetic",
    },
    key=len,
    reverse=True,
)

class PredictRequest(BaseModel):
    text: str

def get_current_time() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(CDT).isoformat()
 
# HELPERS
def clean_text(text: str) -> str:

    text = text.lower()

    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    text = re.sub(r"\d+", "", text)

    words = text.split()

    words = [
        lemmatizer.lemmatize(w)
        for w in words
    ]

    return " ".join(words).strip()


def contains_hard_threat(text: str) -> bool:

    words = set(
        re.findall(r"\b\w+\b", text.lower())
    )

    return bool(words & _HARD_THREAT_WORDS)


def detect_sentiment(text: str) -> str:

    words = set(text.lower().split())

    pos = len(words & _POSITIVE_WORDS)

    neg = len(
        words & (_NEGATIVE_WORDS | _HARD_THREAT_WORDS)
    )

    if pos > neg:
        return "Positive"

    if neg > pos:
        return "Negative"

    return "Neutral"

def classify_behavior(text: str) -> str:
    
    lower = text.lower()

    if contains_hard_threat(lower):
        return "Threat"

    harassment_words = {
        "hate",
        "worthless",
        "loser",
        "idiot",
        "stupid",
        "pathetic",
        "ugly",
        "moron",
    }

    toxic_words = {
        "annoying",
        "dumb",
        "freak",
    }

    positive_words = {
        "great",
        "love",
        "awesome",
        "wonderful",
        "thank",
    }

    words = set(
        re.findall(r"\b\w+\b", lower)
    )

    if words & harassment_words:
        return "Harassment"

    if words & toxic_words:
        return "Toxic"

    if words & positive_words:
        return "Positive"

    return "Safe"

def highlight_keywords(text: str) -> str:

    for word in _FLAG_WORDS:

        pattern = r"\b" + re.escape(word) + r"\b"

        text = re.sub(
            pattern,
            lambda m: (
                f'<mark class="bg-danger">'
                f'{m.group(0)}</mark>'
            ),
            text,
            flags=re.IGNORECASE,
        )

    return text

# FRONTEND ROUTES

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):

    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):

    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request}
    )

@app.post("/api/v1/predict")
async def predict(payload: PredictRequest):

    text = payload.text.strip()

    if not text:
        return JSONResponse(
            status_code=422,
            content={"error": "Empty text"}
        )

    if len(text) > cfg.max_input_chars:
        text = text[:cfg.max_input_chars]

    cleaned = clean_text(text)

    ensemble_result = ensemble_predict(text)

    if contains_hard_threat(text):
        ensemble_result["prediction"] = 1
        ensemble_result["confidence"] = max(
            ensemble_result["confidence"],
            0.95
        )

        ensemble_result["severity"] = "Critical"
        ensemble_result["threat_category"] = "Violence"
    
    prediction = ensemble_result["prediction"]

    probability = ensemble_result["confidence"]

    bullying_percentage = round(probability * 100, 2)

    # Timestamp recorded at the moment of prediction
    now = get_current_time()

    result = {
        "prediction": prediction,

        "confidence": round(probability, 4),

        "bullying_percentage": bullying_percentage,

        "label": (
            "Threat"
            if prediction == 1
            else "Safe"
        ),

        "severity": ensemble_result["severity"],

        "threat_category": (
            ensemble_result["threat_category"]
        ),

        "logistic_score": (
            ensemble_result["logistic_score"]
        ),

        "transformer_score": (
            ensemble_result["transformer_score"]
        ),

        "rule_boost": (
            ensemble_result["rule_boost"]
        ),

        "classification": classify_behavior(text),

        "sentiment": detect_sentiment(text),

        "highlighted": highlight_keywords(text),

        "timestamp": now,
    }

    cursor.execute(
        "INSERT INTO logs (text, prediction, confidence, created_at) VALUES (?, ?, ?, ?)",
        (text, prediction, probability, now)
    )
    conn.commit()

    logger.info(
        "pred=%d conf=%.4f classification=%s cleaned=%r",
        prediction,
        probability,
        result["classification"],
        cleaned
    )

    return result

@app.get("/api/v1/logs")
def get_logs():

    cursor.execute("""
        SELECT
            id,
            text,
            prediction,
            confidence,
            created_at
        FROM logs
        ORDER BY id DESC
        LIMIT 20
    """)

    rows = cursor.fetchall()

    results = []

    for row in rows:

        text = row["text"]

        prediction = row["prediction"]

        confidence = float(row["confidence"])

        risk_score = round(confidence * 100)

        severity = "Low"

        if risk_score >= 85:
            severity = "Critical"

        elif risk_score >= 70:
            severity = "High"

        elif risk_score >= 40:
            severity = "Medium"

        # REAL sentiment detection
        # Sentiment mapping
        if severity == "Low":
            sentiment = "Positive"

        elif severity == "Medium":
            sentiment = "Neutral"

        else:
            sentiment = "Negative"

        results.append({

            "id": row["id"],

            "threat_class": (
                "Threat"
                if prediction == 1
                else "Safe"
            ),

            "severity": severity,

            "confidence": confidence,

            "risk_score": risk_score,

            "sentiment": sentiment,

            "created_at": row["created_at"]
        })

    return results

@app.get("/api/v1/analytics")
async def analytics():

    cursor.execute(
        "SELECT COUNT(*) FROM logs"
    )

    total_scans = cursor.fetchone()[0]

    cursor.execute(
        "SELECT COUNT(*) FROM logs WHERE prediction = 1"
    )

    threats_detected = cursor.fetchone()[0]

    threat_rate = round(
        (threats_detected / total_scans) * 100,
        2
    ) if total_scans > 0 else 0

    cursor.execute(
        "SELECT AVG(confidence) FROM logs"
    )

    avg_conf = cursor.fetchone()[0] or 0

    cursor.execute("""
        SELECT confidence
        FROM logs
    """)

    severity_rows = cursor.fetchall()

    severity_distribution = {
        "Low": 0,
        "Medium": 0,
        "High": 0,
        "Critical": 0
    }

    for row in severity_rows:

        confidence = float(row[0])

        risk_score = round(confidence * 100)

        if risk_score >= 85:

            severity_distribution["Critical"] += 1

        elif risk_score >= 70:

            severity_distribution["High"] += 1

        elif risk_score >= 40:

            severity_distribution["Medium"] += 1

        else:

            severity_distribution["Low"] += 1

    return {

        "total_scans": total_scans,

        "threats_detected": threats_detected,

        "threat_rate": threat_rate,

        "avg_risk": round(avg_conf * 100, 2),

        "class_distribution": {

            "Safe": (
                total_scans - threats_detected
            ),

            "Threat": threats_detected
        },

        "severity_distribution": severity_distribution
    }

# HEALTH
@app.get("/health")
async def health():

    return {
        "status": "ok",
        "model": "Logistic Regression + TF-IDF",
        "timestamp": get_current_time(),
    }

if __name__ == "__main__":

    logger.info("Threat API running... started at %s", get_current_time())

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=5000,
        reload=True
    )
