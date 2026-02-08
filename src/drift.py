# src/drift.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


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


def _resolve_run_dir(run_dir_str: str) -> Path:
    run_dir = Path(run_dir_str)
    if run_dir.exists():
        return run_dir
    alt = Path("src") / run_dir
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Run dir not found: {run_dir_str}")


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if not np.issubdtype(df["created_datetime"].dtype, np.datetime64):
        df = df.copy()
        df["created_datetime"] = pd.to_datetime(df["created_datetime"], errors="coerce")
    return df


def _slice_live_window(df_live: pd.DataFrame, live_year: int, months: int | None, start: str | None, end: str | None) -> pd.DataFrame:
    df_live = _ensure_datetime(df_live)

    if start or end:
        start_dt = pd.to_datetime(start) if start else pd.Timestamp.min
        end_dt = pd.to_datetime(end) if end else pd.Timestamp.max
        out = df_live[(df_live["created_datetime"] >= start_dt) & (df_live["created_datetime"] <= end_dt)].copy()
        return out

    if months is None:
        return df_live

    # first N months of live_year: Jan..N
    out = df_live[
        (df_live["created_datetime"].dt.year == live_year) &
        (df_live["created_datetime"].dt.month >= 1) &
        (df_live["created_datetime"].dt.month <= months)
    ].copy()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-name", default="drift_2017.json")

    # Live simulation knobs
    parser.add_argument("--months", type=int, default=None, help="Use only first N months of live year (e.g. 3 = Jan–Mar)")
    parser.add_argument("--start", type=str, default=None, help="Optional explicit start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="Optional explicit end date (YYYY-MM-DD)")

    args = parser.parse_args()
    run_dir = _resolve_run_dir(args.run_dir)

    ref_path = run_dir / "posts_to_train_end_year.parquet"
    live_path = run_dir / "posts_live_year.parquet"
    model_path = run_dir / "model.joblib"
    cfg_path = run_dir / "config.json"

    for p in [ref_path, live_path, model_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required artifact: {p}")

    ref = _ensure_datetime(pd.read_parquet(ref_path))
    live_full = _ensure_datetime(pd.read_parquet(live_path))
    model = joblib.load(model_path)

    cfg = {}
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
        except Exception:
            cfg = {}

    train_end_year = cfg.get("train_end_year", None)
    live_year = cfg.get("live_year", int(live_full["created_datetime"].dt.year.mode()[0]))

    live = _slice_live_window(live_full, live_year=live_year, months=args.months, start=args.start, end=args.end)

    if ref.empty:
        raise ValueError("Reference dataframe is empty.")
    if live.empty:
        raise ValueError("Selected live window is empty. Try a different months/start/end.")

    # 1) title length drift
    ref_len = ref["title_len_log"].astype(float).values
    live_len = live["title_len_log"].astype(float).values
    ref_mean = float(np.nanmean(ref_len))
    live_mean = float(np.nanmean(live_len))
    ref_std = float(np.nanstd(ref_len) + 1e-12)
    live_std = float(np.nanstd(live_len))

    mean_shift = float(live_mean - ref_mean)
    mean_shift_z = float(abs(mean_shift) / ref_std)

    # 2) hour distribution drift (TVD)
    ref_hours = ref["hour"].astype(int).clip(0, 23).values
    live_hours = live["hour"].astype(int).clip(0, 23).values
    ref_hist = np.bincount(ref_hours, minlength=24).astype(float)
    live_hist = np.bincount(live_hours, minlength=24).astype(float)
    hour_tvd = _tvd(ref_hist, live_hist)

    # 3) question mark rate drift
    ref_q = ref["has_question_mark"].astype(float).values
    live_q = live["has_question_mark"].astype(float).values
    ref_q_rate = float(np.nanmean(ref_q))
    live_q_rate = float(np.nanmean(live_q))
    q_shift = float(live_q_rate - ref_q_rate)

    # 4) prediction score distribution drift (KS)
    ref_scores = model.predict_proba(ref[FEATURES].copy())[:, 1]
    live_scores = model.predict_proba(live[FEATURES].copy())[:, 1]
    score_ks = _ks_statistic(ref_scores, live_scores)

    def score_summary(arr: np.ndarray) -> dict:
        arr = np.asarray(arr, dtype=float)
        return {
            "mean": float(np.mean(arr)),
            "p10": float(np.quantile(arr, 0.10)),
            "p50": float(np.quantile(arr, 0.50)),
            "p90": float(np.quantile(arr, 0.90)),
        }

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

    # window info for “live simulation”
    window = {
        "months": args.months,
        "start": args.start,
        "end": args.end,
        "live_min_date": str(live["created_datetime"].min()),
        "live_max_date": str(live["created_datetime"].max()),
    }

    out = {
        "train_end_year": train_end_year,
        "live_year": live_year,
        "window": window,
        "n_reference": int(len(ref)),
        "n_live_window": int(len(live)),
        "drift_signals": {
            "title_len_log": {
                "ref_mean": ref_mean,
                "live_mean": live_mean,
                "mean_shift": mean_shift,
                "mean_shift_z": mean_shift_z,
                "ref_std": float(ref_std),
                "live_std": float(live_std),
            },
            "hour_of_day": {"tvd_24bin": float(hour_tvd)},
            "question_mark_rate": {"ref_rate": ref_q_rate, "live_rate": live_q_rate, "shift": q_shift},
            "prediction_scores": {
                "ks": float(score_ks),
                "ref_summary": score_summary(ref_scores),
                "live_summary": score_summary(live_scores),
            },
        },
        "thresholds": thresholds,
        "flags": flags,
        "should_retrain": should_retrain,
        "note": "Drift uses pre-label signals only (features + prediction distributions).",
    }

    out_path = run_dir / args.out_name
    out_path.write_text(json.dumps(out, indent=2))

    print("[OK] Drift check complete")
    print(f"[OK] Saved: {out_path}")
    print(json.dumps({"should_retrain": should_retrain, "flags": flags, "window": window}, indent=2))


if __name__ == "__main__":
    main()
