#!/usr/bin/env bash
# Rerun every notebook in order, then the tests, then rebuild the report and slides.
# Usage: bash scripts/run_all.sh   (from the repo root, data in data/raw/)
set -euo pipefail
cd "$(dirname "$0")/.."

for nb in notebooks/[0-9][0-9]_*.ipynb; do
  echo "== $nb"
  jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 "$nb"
done

python -m pytest -q tests

pandoc reports/report.md -o reports/NkwaPay_Report_Group3.docx --resource-path=reports
python slides/build_slides.py
echo "Done."
