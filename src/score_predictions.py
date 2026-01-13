# src/score_predictions.py
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", default="predictions/history.csv", help="prediction history csv")
    parser.add_argument("--prices", default="data/raw/prices_spy.csv", help="latest prices csv")
    args = parser.parse_args()

    pred_path = Path(args.pred)
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    preds = pd.read_csv(pred_path)

    # ✅ ここが重要：既に採点列があっても毎回やり直せるように一旦消す
    for col in ["actual_up_next_day", "is_correct"]:
        if col in preds.columns:
            preds = preds.drop(columns=[col])

    prices = pd.read_csv(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    # 実際の翌日上げ下げ
    prices["next_close"] = prices.groupby("ticker")["close"].shift(-1)
    prices["actual_up_next_day"] = (prices["next_close"] > prices["close"]).astype(int)

    # joinキーを作る
    prices["asof_date"] = prices["date"].dt.date.astype(str)
    prices_key = prices[["ticker", "asof_date", "actual_up_next_day"]]

    # merge
    merged = preds.merge(prices_key, on=["ticker", "asof_date"], how="left")

    # 採点（翌日データがない日はNAのまま）
    merged["is_correct"] = pd.NA
    mask = merged["actual_up_next_day"].notna()
    merged.loc[mask, "is_correct"] = (merged.loc[mask, "pred_up"] == merged.loc[mask, "actual_up_next_day"]).astype(float)

    # 保存（上書き）
    merged.to_csv(pred_path, index=False)

    # サマリ（採点できた行だけ）
    scored = merged.dropna(subset=["is_correct"])
    if len(scored) == 0:
        print("⚠️ No rows could be scored yet (next day price not available).")
        return

    acc = scored["is_correct"].astype(float).mean()
    print(f"✅ Scored rows: {len(scored)}  Accuracy so far: {acc:.4f}")


if __name__ == "__main__":
    main()

