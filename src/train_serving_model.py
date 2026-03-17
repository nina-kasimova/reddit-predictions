from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from src.train import TrainConfig, build_features, build_model, label_popularity


FEATURES = [
    "conv_title",
    "title_len",
    "universe_tag",
    "hour",
    "day_of_week",
    "title_len_log",
    "hour_sin",
    "hour_cos",
    "has_question_mark",
    "conv_author_flair_text",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-data", default="posts-to-2016.parquet")
    parser.add_argument("--output-model", default="/models/model.joblib")
    parser.add_argument("--output-metadata", default="/models/model_metadata.json")
    parser.add_argument("--train-end-year", type=int, default=2016)
    parser.add_argument("--live-year", type=int, default=2017)
    args = parser.parse_args()

    reference_path = Path(args.reference_data)
    output_model = Path(args.output_model)
    output_metadata = Path(args.output_metadata)

    if not reference_path.exists():
        raise FileNotFoundError(f"Reference parquet not found: {reference_path}")

    output_model.parent.mkdir(parents=True, exist_ok=True)
    output_metadata.parent.mkdir(parents=True, exist_ok=True)

    cfg = TrainConfig(
        data_path=str(reference_path),
        train_end_year=args.train_end_year,
        live_year=args.live_year,
    )

    df = pd.read_parquet(reference_path)
    df = build_features(df)
    df = label_popularity(df, train_end_year=cfg.train_end_year)

    train_df = df[df["created_datetime"].dt.year <= cfg.train_end_year].copy()
    if train_df.empty:
        raise ValueError("No training rows found after feature engineering.")

    model = build_model(cfg)
    model.fit(train_df[FEATURES].copy(), train_df["is_popular"].astype(int).copy())
    joblib.dump(model, output_model)

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "reference_data": str(reference_path),
        "train_end_year": cfg.train_end_year,
        "live_year": cfg.live_year,
        "n_training_rows": int(len(train_df)),
        "feature_names": FEATURES,
    }
    output_metadata.write_text(json.dumps(metadata, indent=2))

    print(f"[OK] Saved serving model to {output_model}")
    print(f"[OK] Saved metadata to {output_metadata}")


if __name__ == "__main__":
    main()
