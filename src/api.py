from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.train import build_features


MODEL_PATH = Path(os.getenv("MODEL_PATH", "/models/model.joblib"))
MODEL_METADATA_PATH = Path(os.getenv("MODEL_METADATA_PATH", "/models/model_metadata.json"))

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


class PostPayload(BaseModel):
    conv_title: str = Field(..., description="Post title.")
    created_datetime: str = Field(..., description="ISO-8601 timestamp used for time features.")
    conv_author_flair_text: str | None = Field(default=None, description="Optional author flair text.")
    post_id: str | None = Field(default=None, description="Optional external identifier.")
    subreddit: str | None = Field(default=None, description="Optional subreddit name for tracing.")
    body: str | None = Field(default=None, description="Unused by the model today, but useful for future versions.")


class PredictResponse(BaseModel):
    popular_probability: float
    predicted_label: int
    model_path: str
    engineered_features: dict[str, object]


app = FastAPI(title="Reddit Popularity API", version="0.1.0")


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def load_model_metadata() -> dict[str, object]:
    if not MODEL_METADATA_PATH.exists():
        return {}
    try:
        return json.loads(MODEL_METADATA_PATH.read_text())
    except json.JSONDecodeError:
        return {"warning": f"Could not parse metadata file at {MODEL_METADATA_PATH}"}


def build_inference_frame(payload: PostPayload) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "conv_title": payload.conv_title,
                "created_datetime": payload.created_datetime,
                "conv_author_flair_text": payload.conv_author_flair_text,
                "utt_score": 0,
            }
        ]
    )
    return build_features(frame)


@app.get("/health")
def health() -> dict[str, object]:
    model_ready = MODEL_PATH.exists()
    metadata = load_model_metadata()
    return {
        "status": "ok" if model_ready else "degraded",
        "model_ready": model_ready,
        "model_path": str(MODEL_PATH),
        "metadata": metadata,
    }


@app.get("/ready")
def ready() -> dict[str, str]:
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model is not available yet.")
    return {"status": "ready"}


@app.get("/model/info")
def model_info() -> dict[str, object]:
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model is not available yet.")
    return {
        "model_path": str(MODEL_PATH),
        "metadata": load_model_metadata(),
        "features": FEATURES,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PostPayload) -> PredictResponse:
    try:
        model = load_model()
        frame = build_inference_frame(payload)
        feature_frame = frame[FEATURES].copy()
        probability = float(model.predict_proba(feature_frame)[0, 1])
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    engineered = frame.iloc[0][FEATURES].to_dict()
    return PredictResponse(
        popular_probability=probability,
        predicted_label=int(probability >= 0.5),
        model_path=str(MODEL_PATH),
        engineered_features=engineered,
    )
