"""Train and score review classification models.

Input CSV needs `review` and `label` (fake/genuine or 1/0).
Optional features: rating, verified_purchase, reviewer_review_count.
The included dataset is synthetic and strictly for demonstration.
"""
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "reviews.csv"
MODEL_PATH = ROOT / "review_model.joblib"
NUMERIC = ["rating", "verified_purchase", "reviewer_review_count", "review_length", "exclamation_count"]


def prepare_features(df):
    """Accept optional metadata, avoiding text/csv missing-value failures."""
    if "review" not in df.columns:
        raise ValueError("CSV must contain a 'review' column.")
    x = df.copy()
    x["review"] = x["review"].fillna("").astype(str).str.strip()
    x["rating"] = pd.to_numeric(x["rating"], errors="coerce") if "rating" in x else np.nan
    if "verified_purchase" in x:
        x["verified_purchase"] = x["verified_purchase"].astype(str).str.lower().map(
            {"true": 1, "false": 0, "yes": 1, "no": 0, "1": 1, "0": 0})
    else:
        x["verified_purchase"] = np.nan
    x["reviewer_review_count"] = (pd.to_numeric(x["reviewer_review_count"], errors="coerce")
                                  if "reviewer_review_count" in x else np.nan)
    x["review_length"] = x["review"].str.len()
    x["exclamation_count"] = x["review"].str.count("!")
    return x[["review", *NUMERIC]]


def parse_labels(series):
    y = series.astype(str).str.strip().str.lower().map(
        {"fake": 1, "genuine": 0, "real": 0, "1": 1, "0": 0})
    if y.isna().any():
        raise ValueError("Labels must be fake/genuine (or 1/0).")
    return y.astype(int)


def feature_transformer():
    return ColumnTransformer([
        ("text", TfidfVectorizer(ngram_range=(1, 2), max_features=3000,
                                 min_df=1, strip_accents="unicode", sublinear_tf=True), "review"),
        ("numeric", Pipeline([
            ("fill", SimpleImputer(strategy="constant", fill_value=0)),
            ("scale", StandardScaler(with_mean=False)),
        ]), NUMERIC),
    ])


def train(df, save=False):
    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column.")
    df = df.dropna(subset=["review", "label"]).copy()
    df = df[df["review"].astype(str).str.strip().ne("")]
    y = parse_labels(df["label"])
    if len(df) < 20 or y.value_counts().min() < 5:
        raise ValueError("Use at least 20 reviews, including at least 5 examples of each class.")
    x = prepare_features(df)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, stratify=y, random_state=42)
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=120, class_weight="balanced",
                                                min_samples_leaf=2, random_state=42, n_jobs=-1),
    }
    best = None
    results = {}
    for name, estimator in candidates.items():
        pipe = Pipeline([("features", feature_transformer()), ("classifier", estimator)])
        pipe.fit(x_train, y_train)
        predicted = pipe.predict(x_test)
        score = f1_score(y_test, predicted, zero_division=0)
        results[name] = round(float(score), 3)
        if best is None or score > best[0]:
            best = (score, name, pipe, predicted)
    _, name, pipe, predicted = best
    metrics = classification_report(y_test, predicted, labels=[0, 1],
                                    target_names=["Genuine", "Fake"], output_dict=True,
                                    zero_division=0)
    payload = {"pipeline": pipe, "model_name": name, "scores": results,
               "report": metrics, "confusion_matrix": confusion_matrix(y_test, predicted, labels=[0, 1]).tolist(),
               "test_size": len(y_test), "train_size": len(y_train)}
    if save:
        joblib.dump(payload, MODEL_PATH)
    return payload


def predict(payload, review, rating=None, verified_purchase=None, reviewer_review_count=None):
    sample = pd.DataFrame([{"review": review, "rating": rating,
                            "verified_purchase": verified_purchase,
                            "reviewer_review_count": reviewer_review_count}])
    x = prepare_features(sample)
    proba = float(payload["pipeline"].predict_proba(x)[0, 1])
    return proba, "Potentially fake" if proba >= 0.5 else "Likely genuine"


if __name__ == "__main__":
    result = train(pd.read_csv(DATA_PATH), save=True)
    print("Model:", result["model_name"])
    print("Held-out F1 scores:", result["scores"])
    print("Saved to:", MODEL_PATH)
