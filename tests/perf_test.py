"""
Performance test: send 100 requests per test case to the deployed REST API,
record per-call timestamps and latency to CSV, and print basic stats.

This script creates one CSV per test case under tests/results/ with exactly
100 rows each, as required by the assignment.

Usage:
  python tests/perf_test.py --base-url http://YOUR-EB-DOMAIN
"""
from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import requests


DEFAULT_URL = "http://444-env.eba-vkypmpy2.us-east-1.elasticbeanstalk.com"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_test_cases() -> Dict[str, str]:
    return {
        "real_1": (
            "The United Nations announced today a new climate agreement "
            "signed by over 100 countries, aiming to reduce emissions by 2035."
        ),
        "real_2": (
            "NASA's Artemis program has completed another successful test, "
            "moving closer to returning astronauts to the Moon."
        ),
        "fake_1": (
            "Scientists confirm that chocolate cures all diseases instantly, "
            "replacing the need for hospitals worldwide."
        ),
        "fake_2": (
            "A small town reportedly levitated into the sky after a mysterious "
            "sound; authorities claim gravity took a day off."
        ),
    }


def run_case(session: requests.Session, url: str, name: str, message: str, n: int = 100) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RESULTS_DIR / f"{name}.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iteration", "start_utc", "end_utc", "elapsed_ms", "status_code", "label"])
        for i in range(1, n + 1):
            start_ts = utc_now_iso()
            t0 = time.perf_counter()
            status = None
            label = None
            try:
                r = session.post(url, json={"message": message}, timeout=15)
                status = r.status_code
                if r.headers.get("content-type", "").startswith("application/json"):
                    label = (r.json() or {}).get("label")
            except Exception:
                status = 0
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            end_ts = utc_now_iso()
            writer.writerow([i, start_ts, end_ts, f"{elapsed_ms:.3f}", status, label or ""])  # exactly 100 rows
    return out_csv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.getenv("BASE_URL", DEFAULT_URL))
    parser.add_argument("--count", type=int, default=100, help="Calls per test case (default 100)")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    url = f"{base}/predict"
    tests = get_test_cases()

    print(f"Running performance test against: {url}")
    files: List[Path] = []
    with requests.Session() as s:
        for name, msg in tests.items():
            print(f"- Case '{name}' ...")
            files.append(run_case(s, url, name, msg, n=args.count))
    print("Done. CSV files:")
    for p in files:
        print(f"  {p}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
