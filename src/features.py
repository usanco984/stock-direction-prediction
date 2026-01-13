# src/features.py
from __future__ import annotations
import pandas as pd

FEATURE_COLS = ["ret_1d", "ma5_gap", "ma20_gap"]

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").reset_index(drop=True)

    out["ret_1d"] = out["close"].pct_change()
    out["ma_5"] = out["close"].rolling(5).mean()
    out["ma_20"] = out["close"].rolling(20).mean()
    out["ma5_gap"] = out["close"] / out["ma_5"] - 1
    out["ma20_gap"] = out["close"] / out["ma_20"] - 1

    out["target_up"] = (out["close"].shift(-1) > out["close"]).astype(int)
    out = out.dropna().reset_index(drop=True)

    return out



