"""
Generate boxplots and averages from CSV results produced by perf_test.py.

Output files (under tests/results/):
  - performance_boxplot.png
  - averages.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_data() -> pd.DataFrame:
    frames = []
    for csv_path in RESULTS_DIR.glob("*.csv"):
        if csv_path.name == "averages.csv":
            continue
        case = csv_path.stem
        df = pd.read_csv(csv_path)
        df["case"] = case
        # coerce and drop invalid
        df["elapsed_ms"] = pd.to_numeric(df["elapsed_ms"], errors="coerce")
        df = df.dropna(subset=["elapsed_ms"])  # keep numeric rows only
        frames.append(df)
    if not frames:
        raise SystemExit("No CSV files found in tests/results. Run perf_test.py first.")
    return pd.concat(frames, ignore_index=True)


def make_boxplot(df: pd.DataFrame, out_png: Path) -> None:
    plt.figure(figsize=(8, 5))
    # Ensure order: sorted by case name
    order = sorted(df["case"].unique())
    data = [df.loc[df["case"] == c, "elapsed_ms"].values for c in order]
    plt.boxplot(data, labels=order, showmeans=True)
    plt.ylabel("Latency (ms)")
    plt.title("API Latency per Test Case (100 calls each)")
    plt.grid(axis="y", alpha=0.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def save_averages(df: pd.DataFrame, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    avg = df.groupby("case")["elapsed_ms"].mean().reset_index()
    avg.rename(columns={"elapsed_ms": "avg_ms"}, inplace=True)
    avg.to_csv(out_csv, index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(RESULTS_DIR / "performance_boxplot.png"))
    args = parser.parse_args()

    df = load_data()
    make_boxplot(df, Path(args.out))
    save_averages(df, RESULTS_DIR / "averages.csv")
    print(f"Saved: {args.out}")
    print(f"Saved: {RESULTS_DIR / 'averages.csv'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
