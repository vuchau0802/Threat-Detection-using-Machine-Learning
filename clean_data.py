"""
clean_data.py
=============
Loads the raw dataset, cleans and preprocesses text,
and saves the result as cleaned_dataset.csv.

Usage:
------
pip install pandas nltk
python clean_data.py
"""

import logging
import re
import string

import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer

# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# =========================================================
# NLTK SETUP
# =========================================================

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

lemmatizer = WordNetLemmatizer()

# =========================================================
# LEETSPEAK NORMALIZATION
# =========================================================

LEET_MAP = {
    "1": "i",
    "3": "e",
    "4": "a",
    "@": "a",
    "0": "o",
    "$": "s",
    "5": "s",
    "7": "t",
}


def normalize_leetspeak(text: str) -> str:
    for k, v in LEET_MAP.items():
        text = text.replace(k, v)
    return text


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text: str) -> str:
    """Full preprocessing pipeline for a single text string."""

    # Lowercase
    text = text.lower()

    # Normalize leetspeak
    text = normalize_leetspeak(text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove @mentions
    text = re.sub(r"@\w+", "", text)

    # Remove non-ASCII (emojis, special chars)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Normalize repeated letters (e.g. "haaaate" → "haate")
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Lemmatize
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words).strip()


# =========================================================
# DATASET LOADING & CLEANING
# =========================================================

def load_and_clean_dataset(
    input_path: str = "data/dataset.csv",
    output_path: str = "data/cleaned_dataset.csv",
) -> pd.DataFrame:

    logger.info("Reading raw dataset from: %s", input_path)
    df = pd.read_csv(input_path)

    logger.info("Raw shape: %s", df.shape)

    # Keep only valid label rows
    df = df[df["label"].isin(["0", "1", 0, 1])].copy()

    # Drop unnamed/junk columns
    df.drop(
        columns=["Unnamed: 2", "Unnamed: 3"],
        errors="ignore",
        inplace=True,
    )

    # Drop rows with missing headline
    before = len(df)
    df = df[df["headline"].notna()].copy()
    logger.info(
        "Dropped %d rows with missing headlines.",
        before - len(df),
    )

    # Cast label to int
    df["label"] = df["label"].astype(int)

    # Apply text cleaning
    logger.info("Cleaning text...")
    df["headline"] = df["headline"].astype(str).apply(clean_text)

    # Drop duplicate cleaned headlines
    before = len(df)
    df.drop_duplicates(subset=["headline"], inplace=True)
    logger.info("Dropped %d duplicate rows.", before - len(df))

    # Drop empty headlines after cleaning
    before = len(df)
    df = df[df["headline"].str.len() > 0]
    logger.info("Dropped %d empty rows after cleaning.", before - len(df))

    # Reset index
    df.reset_index(drop=True, inplace=True)

    logger.info("Final dataset shape: %s", df.shape)
    logger.info("Label distribution:\n%s", df["label"].value_counts().to_string())

    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info("Cleaned dataset saved to: %s", output_path)

    return df


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    df = load_and_clean_dataset(
        input_path="data/dataset.csv",
        output_path="data/cleaned_dataset.csv",
    )
