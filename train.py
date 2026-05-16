import logging
import re
import string
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    dataset_path: str = "data/cleaned_dataset.csv"
    test_size: float = 0.2
    random_state: int = 42
    model_path: str = "models/LogisticRegression.pkl"
    tfidf_max_features: int = 10000

cfg = Config()

nltk.download("wordnet", quiet=True)
lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words).strip()

def load_dataset(path: str):
    df = pd.read_csv(path)

    df = df[df["label"].isin([0, 1, "0", "1"])].copy()
    df = df[df["headline"].notna()].copy()

    df["label"] = df["label"].astype(int)
    df["headline"] = df["headline"].astype(str).apply(clean_text)

    return df

def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=cfg.tfidf_max_features,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

def train():
    df = load_dataset(cfg.dataset_path)

    X_train, X_test, y_train, y_test = train_test_split(
        df["headline"],
        df["label"],
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=df["label"]
    )

    model = build_pipeline()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")

    logger.info("Accuracy: %.4f | F1: %.4f", acc, f1)
    logger.info("\n%s", classification_report(y_test, preds))

    joblib.dump(model, cfg.model_path)
    logger.info("Model saved -> %s", cfg.model_path)

    return model

if __name__ == "__main__":
    train()