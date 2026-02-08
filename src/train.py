# src/train.py
"""
Train script mirroring new-data.ipynb (AskScienceFiction pipeline)

Reads a post-level parquet dataset (created in your notebook),
builds the same engineered features, trains a LogisticRegression model
with:
- word TF-IDF (min_df=5, max_features=15000, ngram 1-2)
- char TF-IDF (char_wb, 3-5, max_features=5000)
- numeric features: day_of_week, has_question_mark, hour_sin, title_len_log, hour_cos
- categorical: conv_author_flair_text, universe_tag

Saves model + metrics into artifacts/.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report


# ----------------------------
# Configs for model
# ----------------------------
@dataclass
class TrainConfig:
    # data
    data_path: str
    train_end_year: int = 2016
    live_year: int = 2017

    # split
    val_size: float = 0.2
    random_state: int = 42

    # vectorizers
    word_min_df: int = 5
    word_max_features: int = 15000
    word_ngram_min: int = 1
    word_ngram_max: int = 2

    char_max_features: int = 5000
    char_ngram_min: int = 3
    char_ngram_max: int = 5

    # model
    max_iter: int = 1000
    class_weight: str = "balanced"


# ----------------------------
# deatures
# ----------------------------
def extract_tag(title: str) -> str:
    """
    e.g.: "[Marvel] Why..." -> "marvel"
    """
    match = re.search(r"\[(.*?)\]", title)
    return match.group(1).lower() if match else "no_tag"


def build_features(df: pd.DataFrame) -> pd.DataFrame:

    # Basic column expectations from your parquet
    required = {"created_datetime", "utt_score", "conv_title"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in parquet: {missing}")

    # Ensure datetime
    if not np.issubdtype(df["created_datetime"].dtype, np.datetime64):
        df["created_datetime"] = pd.to_datetime(df["created_datetime"], errors="coerce")

    df["conv_title"] = df["conv_title"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[df["conv_title"] != ""].copy()


    df["title_len"] = df["conv_title"].str.len()
    df["hour"] = df["created_datetime"].dt.hour
    df["day_of_week"] = df["created_datetime"].dt.dayofweek
    df["has_question_mark"] = df["conv_title"].str.contains(r"\?", na=False).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["title_len_log"] = np.log1p(df["title_len"])

    df["universe_tag"] = df["conv_title"].apply(extract_tag)

    if "conv_author_flair_text" not in df.columns:
        df["conv_author_flair_text"] = None

    return df


# ----------------------------
# labels
# ----------------------------
def label_popularity(df: pd.DataFrame, train_end_year: int) -> pd.DataFrame:
    """
    -  90th percentile of utt_score using <= train_end_year only
    - label all rows using that fixed threshold
    """
    past = df[df["created_datetime"].dt.year <= train_end_year].copy()
    if past.empty:
        raise ValueError(f"No rows found for years <= {train_end_year}. Check created_datetime parsing.")

    threshold = past["utt_score"].quantile(0.90)
    df = df.copy()
    df["is_popular"] = (df["utt_score"] >= threshold).astype(int)
    return df


# ----------------------------
#model
# ----------------------------
def build_model(cfg: TrainConfig) -> Pipeline:
    text_col = "conv_title"


    num_cols = ["day_of_week", "has_question_mark", "hour_sin", "title_len_log", "hour_cos"]
    cat_cols = ["conv_author_flair_text", "universe_tag"]

    preprocess = ColumnTransformer(
        transformers=[

            ("text", TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                min_df=cfg.word_min_df,
                max_features=cfg.word_max_features,
                ngram_range=(cfg.word_ngram_min, cfg.word_ngram_max),
                analyzer="word",
            ), text_col),


            ("char", TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(cfg.char_ngram_min, cfg.char_ngram_max),
                max_features=cfg.char_max_features,

            ), text_col),

            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False)),
            ]), num_cols),

            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), cat_cols),
        ],
        remainder="drop"
    )

    model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("clf", LogisticRegression(
            max_iter=cfg.max_iter,
            class_weight=cfg.class_weight,
        ))
    ])
    return model


# ----------------------------
# training + evaluation
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", dest="data_path", default="data/processed/posts.parquet",
                        help="Path to posts parquet. Falls back to ../posts.parquet if not found.")
    parser.add_argument("--train_end_year", type=int, default=2016)
    parser.add_argument("--live_year", type=int, default=2017)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)

    args = parser.parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists():
        fallback = Path("posts.parquet")
        if fallback.exists():
            data_path = fallback
        else:
            raise FileNotFoundError(
                f"Could not find {args.data_path} or ./posts.parquet. "
                "Save your dataframe to data/processed/posts.parquet (recommended)."
            )

    cfg = TrainConfig(
        data_path=str(data_path),
        train_end_year=args.train_end_year,
        live_year=args.live_year,
        val_size=args.val_size,
        random_state=args.random_state,
    )

    # load data
    df = pd.read_parquet(cfg.data_path)

    df = build_features(df)

    df = label_popularity(df, train_end_year=cfg.train_end_year)

    # split: past vs live
    past = df[df["created_datetime"].dt.year <= cfg.train_end_year].copy()
    live = df[df["created_datetime"].dt.year == cfg.live_year].copy()

    FEATURES = ["conv_title", "title_len", "universe_tag", "hour", "day_of_week",
                "title_len_log", "hour_sin", "hour_cos", "has_question_mark", "conv_author_flair_text"]


    X_past = past[FEATURES].copy()
    y_past = past["is_popular"].astype(int).copy()


    X_train, X_val, y_train, y_val = train_test_split(
        X_past, y_past,
        test_size=cfg.val_size,
        random_state=cfg.random_state,
        stratify=y_past
    )

    # build and fit model
    model = build_model(cfg)
    model.fit(X_train, y_train)

    # dvaluate on validation -
    val_proba = model.predict_proba(X_val)[:, 1]
    val_roc = float(roc_auc_score(y_val, val_proba))
    val_pr = float(average_precision_score(y_val, val_proba))


    live_metrics = {}
    if not live.empty:
        X_live = live[FEATURES].copy()
        y_live = live["is_popular"].astype(int).copy()
        live_proba = model.predict_proba(X_live)[:, 1]
        live_metrics = {
            "live_year": cfg.live_year,
            "live_roc_auc": float(roc_auc_score(y_live, live_proba)),
            "live_pr_auc": float(average_precision_score(y_live, live_proba)),
        }
    else:
        live_metrics = {"live_year": cfg.live_year, "note": "No rows found for live year."}


    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("artifacts") / f"run_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "model.joblib"
    metrics_path = out_dir / "metrics.json"
    config_path = out_dir / "config.json"

    joblib.dump(model, model_path)

    metrics = {
        "val_roc_auc": val_roc,
        "val_pr_auc": val_pr,
        "n_past": int(len(past)),
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "pos_rate_past": float(y_past.mean()),
        **live_metrics,
    }

    metrics_path.write_text(json.dumps(metrics, indent=2))
    config_path.write_text(json.dumps(asdict(cfg), indent=2))


    # (so drift later can compare fixed files)
    (out_dir / "posts_to_train_end_year.parquet").write_bytes(past.to_parquet(index=False))
    if not live.empty:
        (out_dir / "posts_live_year.parquet").write_bytes(live.to_parquet(index=False))

    print(f"[OK] Saved model:   {model_path}")
    print(f"[OK] Saved metrics: {metrics_path}")
    print(f"[OK] Val PR-AUC:   {val_pr:.4f}")
    if "live_pr_auc" in metrics:
        print(f"[OK] Live({cfg.live_year}) PR-AUC: {metrics['live_pr_auc']:.4f}")


if __name__ == "__main__":
    main()
