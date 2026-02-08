# src/run_pipeline.py
"""
Live-replay pipeline for the Reddit popularity project.

Idea:
- Train baseline on reference window (<= train_end_year)
- "Deploy" into live_year, but only see data up to --as-of date (monitor window)
- Run drift(ref vs monitor) using pre-label signals (features + prediction scores)
- If drift -> retrain using all data available up to as-of (reference + monitor)
- Evaluate on the future window (after as-of) to measure how the chosen model behaves

Artifacts saved into a NEW run folder:
- model.joblib (final model used)
- drift_<asof>.json
- eval_future_<asof>.json
- config.json
- parquet snapshots: reference / monitor / future / train_used
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.train import TrainConfig, build_features, build_model, label_popularity

import mlflow
import mlflow.sklearn
from evidently import Report
from evidently.metrics import *
from evidently.presets import *
# from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
# from evidently import ColumnMapping

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


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if not np.issubdtype(df["created_datetime"].dtype, np.datetime64):
        df = df.copy()
        df["created_datetime"] = pd.to_datetime(df["created_datetime"], errors="coerce")
    return df


def _artifact_root() -> Path:
    if Path("src/artifacts").exists():
        return Path("src/artifacts")
    return Path("artifacts")


def _tvd(p: np.ndarray, q: np.ndarray) -> float:
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    return float(0.5 * np.abs(p - q).sum())


def _ks_statistic(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return float("nan")
    x_sorted = np.sort(x)
    y_sorted = np.sort(y)
    all_vals = np.sort(np.unique(np.concatenate([x_sorted, y_sorted])))
    cdf_x = np.searchsorted(x_sorted, all_vals, side="right") / x_sorted.size
    cdf_y = np.searchsorted(y_sorted, all_vals, side="right") / y_sorted.size
    return float(np.max(np.abs(cdf_x - cdf_y)))


def _score_summary(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "p10": float(np.quantile(arr, 0.10)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
    }


def run_drift(model, ref: pd.DataFrame, monitor: pd.DataFrame) -> dict:
    # 1) title length drift (log)
    ref_len = ref["title_len_log"].astype(float).values
    mon_len = monitor["title_len_log"].astype(float).values
    ref_mean = float(np.nanmean(ref_len))
    mon_mean = float(np.nanmean(mon_len))
    ref_std = float(np.nanstd(ref_len) + 1e-12)
    mon_std = float(np.nanstd(mon_len))

    mean_shift = float(mon_mean - ref_mean)
    mean_shift_z = float(abs(mean_shift) / ref_std)

    # 2) hour drift (TVD over 24 bins)
    ref_hours = ref["hour"].astype(int).clip(0, 23).values
    mon_hours = monitor["hour"].astype(int).clip(0, 23).values
    ref_hist = np.bincount(ref_hours, minlength=24).astype(float)
    mon_hist = np.bincount(mon_hours, minlength=24).astype(float)
    hour_tvd = _tvd(ref_hist, mon_hist)

    # 3) question mark rate drift
    ref_q = ref["has_question_mark"].astype(float).values
    mon_q = monitor["has_question_mark"].astype(float).values
    ref_q_rate = float(np.nanmean(ref_q))
    mon_q_rate = float(np.nanmean(mon_q))
    q_shift = float(mon_q_rate - ref_q_rate)

    # 4) prediction score drift (KS)
    ref_scores = model.predict_proba(ref[FEATURES].copy())[:, 1]
    mon_scores = model.predict_proba(monitor[FEATURES].copy())[:, 1]
    score_ks = _ks_statistic(ref_scores, mon_scores)

    # Simple thresholds (policy knobs)
    thresholds = {
        "title_len_mean_shift_z": 0.25,
        "hour_tvd": 0.10,
        "question_rate_abs_shift": 0.03,
        "score_ks": 0.10,
    }

    flags = {
        "title_len_shift": bool(mean_shift_z > thresholds["title_len_mean_shift_z"]),
        "hour_shift": bool(hour_tvd > thresholds["hour_tvd"]),
        "question_shift": bool(abs(q_shift) > thresholds["question_rate_abs_shift"]),
        "score_shift": bool(score_ks > thresholds["score_ks"]),
    }

    should_retrain = bool(any(flags.values()))

    return {
        "drift_signals": {
            "title_len_log": {
                "ref_mean": ref_mean,
                "monitor_mean": mon_mean,
                "mean_shift": mean_shift,
                "mean_shift_z": mean_shift_z,
                "ref_std": float(ref_std),
                "monitor_std": float(mon_std),
            },
            "hour_of_day": {"tvd_24bin": float(hour_tvd)},
            "question_mark_rate": {"ref_rate": ref_q_rate, "monitor_rate": mon_q_rate, "shift": q_shift},
            "prediction_scores": {
                "ks": float(score_ks),
                "ref_summary": _score_summary(ref_scores),
                "monitor_summary": _score_summary(mon_scores),
            },
        },
        "thresholds": thresholds,
        "flags": flags,
        "should_retrain": should_retrain,
        "note": "Drift uses pre-label-style signals (features + prediction distributions).",
    }


def eval_on(df: pd.DataFrame, model) -> dict:
    if df.empty:
        return {"note": "Empty evaluation window."}

    X = df[FEATURES].copy()
    y = df["is_popular"].astype(int).values
    s = model.predict_proba(X)[:, 1]

    pr_auc = float(average_precision_score(y, s))
    base = float(np.mean(y))
    try:
        roc_auc = float(roc_auc_score(y, s))
    except ValueError:
        roc_auc = None

    return {
        "n_samples": int(len(df)),
        "positive_rate": base,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
    }


def fit_model(cfg: TrainConfig, train_df: pd.DataFrame) -> tuple:
    """Fit a model and also report internal val metrics (like train.py)."""
    X_all = train_df[FEATURES].copy()
    y_all = train_df["is_popular"].astype(int).copy()

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_all,
        y_all,
        test_size=cfg.val_size,
        random_state=cfg.random_state,
        stratify=y_all,
    )

    model = build_model(cfg)
    model.fit(X_tr, y_tr)

    val = eval_on(pd.concat([X_val, y_val.rename("is_popular")], axis=1), model)
    return model, val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="src/data/processed/posts.parquet")
    parser.add_argument("--train_end_year", type=int, default=2016)
    parser.add_argument("--live_year", type=int, default=2017)


    parser.add_argument("--as-of", required=True, help="Replay 'current date' within live year. Example: 2017-03-31")


    parser.add_argument("--tag", default=None, help="Optional tag to include in run folder name (e.g. replay_Mar31)")

    args = parser.parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    as_of_dt = pd.to_datetime(args.as_of)

    cfg = TrainConfig(
        data_path=str(data_path),
        train_end_year=args.train_end_year,
        live_year=args.live_year,
    )

    # Load + feature + label
    df = pd.read_parquet(cfg.data_path)
    df = build_features(df)
    df = label_popularity(df, train_end_year=cfg.train_end_year)
    df = _ensure_datetime(df)

    mlflow.set_experiment("Reddit_Popularity")

    with mlflow.start_run(run_name=f"AsOf_{args.as_of}"):



        # split into reference and live year
        reference = df[df["created_datetime"].dt.year <= cfg.train_end_year].copy()
        live_full = df[df["created_datetime"].dt.year == cfg.live_year].copy()

        if reference.empty:
            raise ValueError("Reference window is empty.")
        if live_full.empty:
            raise ValueError("Live year is empty.")

        #  "monitor" and "future" windows inside live year
        monitor = live_full[live_full["created_datetime"] <= as_of_dt].copy()
        future = live_full[live_full["created_datetime"] > as_of_dt].copy()

        mlflow.log_params({
            "as_of_date": args.as_of,
            "n_reference": len(reference),
            "n_monitor": len(monitor)
        })

        if monitor.empty:
            raise ValueError("Monitor window is empty. Pick a later --as-of date within the live year.")
        if future.empty:
            raise ValueError("Future window is empty. Pick an earlier --as-of date within the live year.")


        baseline_model, baseline_val = fit_model(cfg, reference)


        drift = run_drift(baseline_model, reference, monitor)


        # create a visual HTML report comparing Ref vs Monitor
        try:

            drift_report = Report(metrics=[DataDriftPreset()])
            drift_report.run(
                reference_data=reference[FEATURES],
                current_data=monitor[FEATURES],
            )

            snapshot = drift_report.run(
                current_data=monitor[FEATURES],
                reference_data=reference[FEATURES],
            )

            report_path = "drift_report.html"
            snapshot.save_html(report_path)
            mlflow.log_artifact(report_path)
            print("[OK] Evidently drift report saved and logged.")
        except Exception as e:
            print(f"[WARN] Evidently report skipped: {e}")


        if drift["should_retrain"]:
            # "retrain with data available so far"
            train_used = pd.concat([reference, monitor], axis=0).copy()
            final_model, final_val = fit_model(cfg, train_used)
            decision = {"model_used": "retrained_on_reference_plus_monitor"}
            model_type = "retrained"
        else:
            train_used = reference
            final_model = baseline_model
            final_val = baseline_val
            decision = {"model_used": "baseline_reference_only"}
            model_type = "baseline"

        # Evaluate on future window (what happens after as-of)
        future_eval = eval_on(future, final_model)

        mlflow.log_param("model_decision", model_type)
        mlflow.log_metrics({
            "future_pr_auc": future_eval.get("pr_auc", 0),
            "future_pos_rate": future_eval.get("positive_rate", 0),
            "drift_detected": 1 if drift["should_retrain"] else 0
        })

        mlflow.sklearn.log_model(final_model, "model")

        # ----- Save artifacts -----
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = args.tag or f"asof_{as_of_dt.date()}"
        out_dir = _artifact_root() / f"pipeline_{tag}_{run_id}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        joblib.dump(final_model, out_dir / "model.joblib")

        # Save JSONs
        (out_dir / f"drift_{as_of_dt.date()}.json").write_text(json.dumps({
            "train_end_year": cfg.train_end_year,
            "live_year": cfg.live_year,
            "as_of": str(as_of_dt),
            "monitor_min_date": str(monitor["created_datetime"].min()),
            "monitor_max_date": str(monitor["created_datetime"].max()),
            "n_reference": int(len(reference)),
            "n_monitor": int(len(monitor)),
            **drift,
            **decision,
        }, indent=2))

        (out_dir / f"eval_future_{as_of_dt.date()}.json").write_text(json.dumps({
            "train_end_year": cfg.train_end_year,
            "live_year": cfg.live_year,
            "as_of": str(as_of_dt),
            "future_min_date": str(future["created_datetime"].min()),
            "future_max_date": str(future["created_datetime"].max()),
            **decision,
            "internal_val_metrics": final_val,
            "future_eval_metrics": future_eval,
        }, indent=2))

        (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

        # save snapshots
        reference.to_parquet(out_dir / "reference.parquet", index=False)
        monitor.to_parquet(out_dir / "monitor.parquet", index=False)
        future.to_parquet(out_dir / "future.parquet", index=False)
        train_used.to_parquet(out_dir / "train_used.parquet", index=False)

        print("[OK] Pipeline complete")
        print(f"[OK] Saved run: {out_dir}")
        print(json.dumps({
            "as_of": str(as_of_dt),
            "should_retrain": drift["should_retrain"],
            "flags": drift["flags"],
            "model_used": decision["model_used"],
            "future_pr_auc": future_eval.get("pr_auc", None),
            "future_n": future_eval.get("n_samples", None),
        }, indent=2))


if __name__ == "__main__":
    main()
