#!/usr/bin/env python
"""Score one loan application from the command line with the shipped model.

    python scripts/score.py examples/application_repeat.json
    python scripts/score.py my_application.json --model models/nkwapay-v1.joblib

The JSON file holds {"application": {...}, "history": [...]}. The fields are
listed in src/scoring.py (APPLICATION_FIELDS, HISTORY_FIELDS) and in
notebooks/12_scoring_service.ipynb. No training data is needed: the saved
model in models/ is enough.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.scoring import APPLICATION_FIELDS, Scorer  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Score one NkwaPay loan application.")
    ap.add_argument("request", type=Path, help="JSON file with 'application' and optional 'history'")
    ap.add_argument("--model", type=Path, default=None, help="saved scorer (default: models/nkwapay-v1.joblib)")
    args = ap.parse_args(argv)

    request = json.loads(args.request.read_text())
    application = request.get("application", request)
    missing = [f for f in APPLICATION_FIELDS if f not in application and f != "lenders"]
    if missing:
        print(f"The application is missing: {', '.join(missing)}", file=sys.stderr)
        return 2
    try:
        scorer = Scorer.load(args.model)
    except FileNotFoundError:
        print("Saved model not found. It ships in models/; to rebuild it run notebooks/12_scoring_service.ipynb.",
              file=sys.stderr)
        return 2
    print(json.dumps(scorer.score(application, request.get("history") or []), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
