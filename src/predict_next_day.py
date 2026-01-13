from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from src.features import add_features, FEATURE_COLS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/raw/prices_spy.csv", help="latest price CSV")
    parser.add_argument("--model", default="models/logistic_model.pkl", help="trained model path")
    parser.add_argument("--out", default="predictions/history.csv", help="append predictions to this CSV")
    parser.add_argument("--ticker", default="SPY", help="ticker name to record in history")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load data + features
    df = pd.read_csv(args.csv)
    df = add_features(df)

    latest = df.dropna(subset=FEATURE_COLS).iloc[-1:].copy()

    if "date" not in latest.columns:
        raise ValueError("CSV must have 'date' column.")

    asof_date = pd.to_datetime(latest["date"].iloc[0]).date().isoformat()

    # Prevent duplicates
    if out_path.exists():
        hist = pd.read_csv(out_path)
        if (
            ("ticker" in hist.columns)
            and ("asof_date" in hist.columns)
            and ((hist["ticker"] == args.ticker) & (hist["asof_date"] == asof_date)).any()
        ):
            print(f"⚠️ Already predicted for {args.ticker} on {asof_date}. Skipping append.")
            return

    model = joblib.load(args.model)

    prob_up = float(model.predict_proba(latest[FEATURE_COLS])[:, 1][0])
    pred_up = int(prob_up >= 0.5)
    signal = "UP" if pred_up == 1 else "DOWN"

    row = pd.DataFrame(
        [{
            "run_time": datetime.now().replace(microsecond=0).isoformat(),
            "ticker": args.ticker,
            "asof_date": asof_date,
            "pred_up": pred_up,
            "prob_up": round(prob_up, 4),
            "signal": signal,
        }]
    )

    write_header = not out_path.exists()
    row.to_csv(out_path, mode="a", header=write_header, index=False)

    print(row.to_string(index=False))
    print(f"✅ Appended prediction: {out_path}")


if __name__ == "__main__":
    main()
