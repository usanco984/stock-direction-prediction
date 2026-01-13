# Stock Direction Prediction (SPY)

A minimal end-to-end data analytics project:
- Download daily OHLCV from Yahoo Finance
- Feature engineering (returns + moving-average gaps)
- Train a Logistic Regression classifier
- Predict next-day direction (UP/DOWN)
- Append predictions to a history file
- Score predictions when next-day actuals become available
- Run daily via macOS LaunchAgent (optional)

## Project structure
- `src/data_collection.py` : download price data to CSV
- `src/features.py` : feature engineering
- `src/train.py` : train model and save to `models/`
- `src/predict_next_day.py` : generate next-day prediction and append to history
- `src/score_predictions.py` : score prediction history vs actual outcomes
- `src/daily_run.py` : one-command daily pipeline

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install pandas numpy matplotlib scikit-learn yfinance joblib

./.venv/bin/python -m src.daily_run

## Quick start (run end-to-end)

```bash
# 1) create venv
python3 -m venv .venv
source .venv/bin/activate

# 2) install deps
pip install -r requirements.txt

# 3) download data
python src/data_collection.py --tickers SPY --start 2018-01-01 --out data/raw/prices_spy.csv

# 4) train
python -m src.train --csv data/raw/prices_spy.csv

# 5) predict next day and append history
python -m src.predict_next_day --out predictions/history.csv

# 6) score predictions (only rows where next-day actual exists will be scored)
python -m src.score_predictions --pred predictions/history.csv --prices data/raw/prices_spy.csv
