from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yfinance as yf


def download_ohlcv(tickers: List[str], start: str, end: Optional[str] = None) -> pd.DataFrame:
    """
    Download OHLCV data from Yahoo Finance via yfinance.
    Returns tidy data with columns:
    date, ticker, open, high, low, close, adj_close, volume
    """
    df = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False,
    )

    tidy_frames: List[pd.DataFrame] = []

    # MultiIndex columns when multiple tickers are requested
    if isinstance(df.columns, pd.MultiIndex):
        for t in tickers:
            if t not in df.columns.get_level_values(0):
                continue
            sub = df[t].copy()
            sub.columns = [str(c).lower().replace(" ", "_") for c in sub.columns]
            sub = sub.reset_index()
            # Yahoo can return index name 'Date' (capital D)
            sub = sub.rename(columns={"Date": "date", "date": "date"})
            sub["ticker"] = t
            tidy_frames.append(sub)

        out = pd.concat(tidy_frames, ignore_index=True) if tidy_frames else pd.DataFrame()

    # Single ticker case
    else:
        out = df.copy()
        out.columns = [str(c).lower().replace(" ", "_") for c in out.columns]
        out = out.reset_index()
        out = out.rename(columns={"Date": "date", "date": "date"})
        out["ticker"] = tickers[0]

    if out.empty:
        raise RuntimeError("No data downloaded. Check ticker symbols or network access.")

    # Normalize adj close naming
    out = out.rename(columns={"adj close": "adj_close", "adj_close": "adj_close"})

    # Ensure required columns exist
    required = {"date", "ticker", "open", "high", "low", "close", "adj_close", "volume"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Missing columns after download: {missing}. Columns: {list(out.columns)}")

    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=["SPY"], help="e.g., SPY AAPL MSFT")
    parser.add_argument("--start", default="2018-01-01", help="YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD (optional)")
    parser.add_argument("--out", default="data/raw/prices.csv", help="output CSV path")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = download_ohlcv(args.tickers, args.start, args.end)
    data.to_csv(out_path, index=False)

    print(f"Saved: {out_path}  rows={len(data):,}  tickers={sorted(data['ticker'].unique())}")


if __name__ == "__main__":
    main()
