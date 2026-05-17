import re
import string
import os
import joblib
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

lr_model = joblib.load(
    os.getenv("MODEL_PATH", "models/LogisticRegression.pkl")
)

MODEL_NAME = os.getenv(
    "TRANSFORMER_MODEL",
    "unitary/toxic-bert"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

transformer_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME
)

transformer_model.eval()

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

def clean_text(text: str) -> str:

    text = text.lower()

    text = re.sub(r"http\S+|www\S+", "", text)

    text = re.sub(r"@\w+", "", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    text = re.sub(r"\d+", "", text)

    return text.strip()

def get_lr_score(text: str) -> float:

    cleaned = clean_text(text)

    score = lr_model.predict_proba(
        [cleaned]
    )[0][1]

    return float(score)

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

        logits = outputs.logits

        if logits.shape[-1] > 2:
            probs = torch.sigmoid(logits)[0]
            toxic_scores = []

            for idx, prob in enumerate(probs):
                label = transformer_model.config.id2label.get(
                    idx,
                    str(idx)
                ).lower()

                if any(
                    marker in label
                    for marker in (
                        "toxic",
                        "threat",
                        "insult",
                        "obscene",
                        "hate",
                        "identity",
                    )
                ):
                    toxic_scores.append(prob.item())

            if toxic_scores:
                return float(max(toxic_scores))

            return float(probs.max().item())

        probs = torch.softmax(logits, dim=1)[0]

    label_scores = {
        transformer_model.config.id2label.get(idx, str(idx)).lower(): prob.item()
        for idx, prob in enumerate(probs)
    }

    for label, score in label_scores.items():
        if any(
            marker in label
            for marker in ("toxic", "threat", "negative", "unsafe", "harm")
        ):
            return float(score)

    for label, score in label_scores.items():
        if any(marker in label for marker in ("safe", "neutral", "positive")):
            return float(1 - score)

    return float(max(label_scores.values()))

def contains_hard_threat(text: str) -> bool:

    words = set(
        re.findall(r"\b\w+\b", text.lower())
    )

    return bool(
        words & HARD_THREAT_WORDS
    )

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
