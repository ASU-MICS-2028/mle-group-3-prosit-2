#!/usr/bin/env bash
# Rerun everything: data check, every notebook in order, the tests, the claim
# check, then rebuild the report (Word and PDF) and the slides.
# Usage: bash scripts/run_all.sh   or   make      (from the repo root, data in data/raw/)
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== data check"
python -m pytest -q tests/test_data_present.py || {
  echo "The raw data is missing or has the wrong shape. See data/README.md."; exit 1; }

for nb in notebooks/[0-9][0-9]_*.ipynb; do
  echo "== $nb"
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 "$nb"
done

echo "== tests"
python -m pytest -q tests

echo "== claim check"
python scripts/verify_claims.py

echo "== report and slides"
bash scripts/build_report.sh
python slides/build_slides.py
echo "Done."
