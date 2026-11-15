#!/usr/bin/env bash
# Build the report and the audit (reports/report.md, reports/audit.md): Word always, PDF if a LaTeX engine is installed.
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/build_audit.py
for doc in report:NkwaPay_Report_Group3 audit:NkwaPay_Audit_Group3; do
  src="reports/${doc%%:*}.md"; out="reports/${doc##*:}"
  pandoc "$src" -o "$out.docx" --resource-path=reports
  if command -v xelatex >/dev/null 2>&1; then
    pandoc "$src" -o "$out.pdf" --resource-path=reports \
      --pdf-engine=xelatex -V geometry:margin=2.2cm -V fontsize=10pt -V colorlinks=true
    echo "Wrote $out.docx and .pdf"
  else
    echo "Wrote $out.docx (xelatex not found, so no PDF; export it from Word instead)"
  fi
done
