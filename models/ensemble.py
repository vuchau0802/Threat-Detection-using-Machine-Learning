import re
import string
import joblib
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# =========================================================
# LOAD CLASSICAL MODEL
# =========================================================
lr_model = joblib.load(
    "models/LogisticRegression.pkl"
)

# =========================================================
# LOAD TRANSFORMER
# =========================================================
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

transformer_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME
)

transformer_model.eval()

# =========================================================
# THREAT WORDS
# =========================================================
HARD_THREAT_WORDS = {
    "kill",
    "murder",
    "shoot",
    "stab",
    "rape",
    "harm",
    "hurt",
    "destroy",
    "slaughter",
    "die",
}

# =========================================================
# CLEAN TEXT
# =========================================================
def clean_text(text: str) -> str:

    text = text.lower()

    text = re.sub(r"http\S+|www\S+", "", text)

    text = re.sub(r"@\w+", "", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    text = re.sub(r"\d+", "", text)

    return text.strip()

# =========================================================
# LOGISTIC REGRESSION SCORE
# =========================================================
def get_lr_score(text: str) -> float:

    cleaned = clean_text(text)

    score = lr_model.predict_proba(
        [cleaned]
    )[0][1]

    return float(score)

# =========================================================
# TRANSFORMER SCORE
# =========================================================
def get_transformer_score(text: str) -> float:

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():

        outputs = transformer_model(**inputs)

        probs = torch.softmax(
            outputs.logits,
            dim=1
        )

    negative_score = probs[0][0].item()

    return float(negative_score)

# =========================================================
# RULE ENGINE
# =========================================================
def contains_hard_threat(text: str) -> bool:

    words = set(
        re.findall(r"\b\w+\b", text.lower())
    )

    return bool(
        words & HARD_THREAT_WORDS
    )

# =========================================================
# ENSEMBLE ENGINE
# =========================================================
def ensemble_predict(text: str):

    lr_score = get_lr_score(text)

    transformer_score = get_transformer_score(text)

    rule_boost = 0.0

    if contains_hard_threat(text):
        rule_boost = 0.25

    # Weighted ensemble
    final_score = (
        (0.30 * lr_score) +
        (0.60 * transformer_score) +
        (0.10 * rule_boost)
    )

    final_score = min(final_score, 1.0)

    prediction = 1 if final_score >= 0.50 else 0

    # Threat category
    threat_category = "Safe"

    lowered = text.lower()

    if any(word in lowered for word in ["kill", "murder", "shoot", "stab"]):
        threat_category = "Violence"

    elif any(word in lowered for word in ["hate", "idiot", "moron"]):
        threat_category = "Harassment"

    elif any(word in lowered for word in ["rape"]):
        threat_category = "Sexual Threat"

    # Severity
    severity = "Low"

    if final_score >= 0.90:
        severity = "Critical"

    elif final_score >= 0.75:
        severity = "High"

    elif final_score >= 0.50:
        severity = "Medium"

    return {
        "prediction": prediction,
        "confidence": round(final_score, 4),

        "logistic_score": round(lr_score, 4),

        "transformer_score": round(
            transformer_score,
            4
        ),

        "rule_boost": round(rule_boost, 4),

        "severity": severity,

        "threat_category": threat_category
    }