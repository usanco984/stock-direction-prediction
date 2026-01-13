# src/train.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

from src.features import add_features, FEATURE_COLS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/raw/prices_spy.csv", help="training CSV path")
    parser.add_argument("--out_model", default="models/logistic_model.pkl")
    parser.add_argument("--out_meta", default="models/metadata.json")
    args = parser.parse_args()

    # Load and feature engineering
    df = pd.read_csv(args.csv)
    df = add_features(df)

    X = df[FEATURE_COLS]
    y = df["target_up"]

    # Train
    model = LogisticRegression(max_iter=2000)
    model.fit(X, y)

    # Evaluate (in-sample; simple baseline)
    pred = model.predict(X)
    acc = accuracy_score(y, pred)

    # Save model
    out_model = Path(args.out_model)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_model)

    # Save metadata (good for portfolio)
    meta = {
        "model_type": "LogisticRegression",
        "feature_cols": FEATURE_COLS,
        "train_rows": int(len(df)),
        "train_start": str(pd.to_datetime(df["date"]).min().date()),
        "train_end": str(pd.to_datetime(df["date"]).max().date()),
        "in_sample_accuracy": float(acc),
    }
    out_meta = Path(args.out_meta)
    out_meta.parent.mkdir(parents=True, exist_ok=True)
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"✅ Saved model: {out_model}")
    print(f"✅ Saved meta : {out_meta}")
    print(f"📌 In-sample accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()


