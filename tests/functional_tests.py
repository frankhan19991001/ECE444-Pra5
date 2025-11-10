"""
Simple functional tests that call the deployed REST API with 4 inputs
and print the status and predicted label for each.

Usage:
  python tests/functional_tests.py --base-url http://YOUR-EB-DOMAIN
If --base-url is omitted, will use the BASE_URL env var, or default to
the sample domain from the README.
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap
from typing import Dict

import requests


DEFAULT_URL = "http://444-env.eba-vkypmpy2.us-east-1.elasticbeanstalk.com"


def get_test_cases() -> Dict[str, str]:
    """Return four test messages (2 real-ish, 2 fake-ish).

    Note: We don't assert correctness of the ML model's label here because
    models differ. We assert the API works and returns a non-empty label.
    """
    return {
        # real-ish
        "real_1": (
            "The United Nations announced today a new climate agreement "
            "signed by over 100 countries, aiming to reduce emissions by 2035."
        ),
        "real_2": (
            "US government shutdown enters 40th day:"
            "How is it affecting Americans?"
        ),
        # fake-ish
        "fake_1": (
            "Scientists confirm that chocolate cures all diseases instantly, "
            "replacing the need for hospitals worldwide."
        ),
        "fake_2": (
            "A small town reportedly levitated into the sky after a mysterious "
            "sound; authorities claim it must be the ufo."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default=os.getenv("BASE_URL", DEFAULT_URL),
        help="Base URL of the deployed service (e.g., http://xxx.elasticbeanstalk.com)",
    )
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    url = f"{base}/predict"
    print(f"Calling: {url}\n")

    tests = get_test_cases()
    ok = True
    for name, msg in tests.items():
        try:
            resp = requests.post(url, json={"message": msg}, timeout=10)
            data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            label = data.get("label")
            print(f"[{name}] status={resp.status_code} label={label}")
            if resp.status_code != 200 or not label:
                ok = False
        except Exception as e:  # pragma: no cover - network error
            ok = False
            print(f"[{name}] ERROR: {e}")

    if not ok:
        print(
            textwrap.dedent(
                f"""
                One or more functional tests failed. Verify the service is reachable at:
                  {base}
                and that POST /predict is responding with JSON {{"label": "..."}}.
                """
            ).strip()
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
