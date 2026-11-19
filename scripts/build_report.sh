#!/usr/bin/env bash
# Build the report and the audit (reports/report.md, reports/audit.md): Word always, PDF if a LaTeX engine
# (xelatex, or tectonic, which is XeTeX-based) is installed.
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/threshold_sweep.py >/dev/null
python scripts/build_audit.py
engine=""
for e in xelatex tectonic; do command -v "$e" >/dev/null 2>&1 && { engine="$e"; break; }; done
for doc in report:NkwaPay_Report_Group3 audit:NkwaPay_Audit_Group3; do
  src="reports/${doc%%:*}.md"; out="reports/${doc##*:}"
  pandoc "$src" -t docx+native_numbering -o "$out.docx" --resource-path=reports   # numbers tables and figures in Word
  if [ -n "$engine" ]; then
    pandoc "$src" -o "$out.pdf" --resource-path=reports \
      --pdf-engine="$engine" -V geometry:margin=2.2cm -V fontsize=10pt -V colorlinks=true
    echo "Wrote $out.docx and .pdf ($engine)"
  else
    echo "Wrote $out.docx (no xelatex or tectonic, so no PDF; export it from Word instead)"
  fi
done
