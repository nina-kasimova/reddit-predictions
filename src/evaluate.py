# src/evaluate.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


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


def _slice_window(df: pd.DataFrame, year: int, months: int | None, start: str | None, end: str | None, mode: str) -> pd.DataFrame:
    df = _ensure_datetime(df)
    df = df[df["created_datetime"].dt.year == year].copy()

    if start or end:
        start_dt = pd.to_datetime(start) if start else pd.Timestamp.min
        end_dt = pd.to_datetime(end) if end else pd.Timestamp.max
        return df[(df["created_datetime"] >= start_dt) & (df["created_datetime"] <= end_dt)].copy()

    if months is None:
        return df

    if mode == "first":
        # first N months: Jan..N
        return df[df["created_datetime"].dt.month <= months].copy()

    if mode == "after":
        # after N months: (N+1)..Dec
        return df[df["created_datetime"].dt.month > months].copy()

    raise ValueError("mode must be 'first' or 'after'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-name", default="eval_2017.json")

    # Window controls
    parser.add_argument("--months", type=int, default=None, help="Use month split point N")
    parser.add_argument("--mode", choices=["all", "first", "after"], default="all",
                        help="all=entire year, first=Jan..N, after=N+1..Dec")
    parser.add_argument("--start", type=str, default=None, help="Optional start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="Optional end date YYYY-MM-DD")

    args = parser.parse_args()
    run_dir = _resolve_run_dir(args.run_dir)

    model_path = run_dir / "model.joblib"
    live_path = run_dir / "posts_live_year.parquet"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")
    if not live_path.exists():
        raise FileNotFoundError(f"Missing live data: {live_path}")

    model = joblib.load(model_path)
    live_full = _ensure_datetime(pd.read_parquet(live_path))

    # Determine year from data
    year = int(live_full["created_datetime"].dt.year.mode()[0])

    if args.mode == "all":
        live = live_full[live_full["created_datetime"].dt.year == year].copy()
    else:
        if args.months is None and not (args.start or args.end):
            raise ValueError("For mode=first/after, provide --months or --start/--end.")
        if args.start or args.end:
            live = _slice_window(live_full, year, months=None, start=args.start, end=args.end, mode="first")
        else:
            live = _slice_window(live_full, year, months=args.months, start=None, end=None, mode=args.mode)

    if live.empty:
        raise ValueError("Selected evaluation window is empty.")

    X = live[FEATURES].copy()
    y_true = live["is_popular"].astype(int).values
    y_score = model.predict_proba(X)[:, 1]

    pr_auc = float(average_precision_score(y_true, y_score))
    base_rate = float(np.mean(y_true))
    try:
        roc_auc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        roc_auc = None

    out = {
        "year": year,
        "mode": args.mode,
        "months_split": args.months,
        "start": args.start,
        "end": args.end,
        "window_min_date": str(live["created_datetime"].min()),
        "window_max_date": str(live["created_datetime"].max()),
        "n_samples": int(len(live)),
        "positive_rate": base_rate,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
    }

    out_path = run_dir / args.out_name
    out_path.write_text(json.dumps(out, indent=2))

    print("[OK] Evaluation complete")
    print(f"[OK] Saved: {out_path}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
