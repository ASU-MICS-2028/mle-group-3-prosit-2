#!/usr/bin/env bash
# Build the report from reports/report.md: Word always, PDF if a LaTeX engine is installed.
set -euo pipefail
cd "$(dirname "$0")/.."

pandoc reports/report.md -o reports/NkwaPay_Report_Group3.docx --resource-path=reports
if command -v xelatex >/dev/null 2>&1; then
  pandoc reports/report.md -o reports/NkwaPay_Report_Group3.pdf --resource-path=reports \
    --pdf-engine=xelatex -V geometry:margin=2.2cm -V fontsize=10pt -V colorlinks=true
  echo "Wrote reports/NkwaPay_Report_Group3.docx and .pdf"
else
  echo "Wrote reports/NkwaPay_Report_Group3.docx (xelatex not found, so no PDF; export it from Word instead)"
fi
