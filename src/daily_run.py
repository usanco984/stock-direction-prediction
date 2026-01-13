from __future__ import annotations

import subprocess
import sys


PYTHON = sys.executable  # ✅ launchdでも確実にこのpython(=venv)を使う


def run(args: list[str]) -> None:
    print("\n$ " + " ".join(args))
    subprocess.run(args, check=True)


def main() -> None:
    run([PYTHON, "src/data_collection.py", "--tickers", "SPY", "--start", "2018-01-01", "--out", "data/raw/prices_spy.csv"])
    run([PYTHON, "-m", "src.train", "--csv", "data/raw/prices_spy.csv"])
    run([PYTHON, "-m", "src.predict_next_day", "--out", "predictions/history.csv"])
    run([PYTHON, "-m", "src.score_predictions", "--pred", "predictions/history.csv", "--prices", "data/raw/prices_spy.csv"])
    print("\n✅ Daily run completed.")


if __name__ == "__main__":
    main()
